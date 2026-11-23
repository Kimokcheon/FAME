"""Official SINE (DINOv2 + matching transformer, FSS) over FAME episodes."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _common import add_common_args, pick_device, run_tasks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official SINE on FAME episodes.")
    parser.add_argument("--sine-root", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Defaults to <sine-root>/outputs/pt_sine/pytorch_model.bin.",
    )
    parser.add_argument(
        "--dinov2-weights",
        type=Path,
        default=None,
        help="Defaults to <sine-root>/../GF-SAM-master/models/dinov2_vitl14_pretrain.pth.",
    )
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--pad-size", type=int, default=896)
    parser.add_argument("--support-batch-size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--feat-chans", type=int, default=256)
    parser.add_argument("--dinov2-size", default="vit_large")
    parser.add_argument("--transformer-depth", type=int, default=6)
    parser.add_argument("--transformer-nheads", type=int, default=8)
    parser.add_argument("--transformer-mlp-dim", type=int, default=2048)
    parser.add_argument("--transformer-mask-dim", type=int, default=256)
    parser.add_argument("--transformer-fusion-layer-depth", type=int, default=1)
    parser.add_argument("--transformer-num-queries", type=int, default=200)
    parser.add_argument(
        "--transformer-pre-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatic mixed precision on CUDA (default: on).",
    )
    parser.add_argument("--device", default="auto")
    add_common_args(parser)
    args = parser.parse_args()

    root = args.sine_root.resolve()
    weights = args.weights or (root / "outputs" / "pt_sine" / "pytorch_model.bin")
    weights = weights.resolve()
    default_dinov2 = (
        root.parent / "GF-SAM-master" / "models" / "dinov2_vitl14_pretrain.pth"
    )
    dinov2_weights = (args.dinov2_weights or default_dinov2).resolve()
    if not (root / "inference_fss" / "model" / "model.py").is_file():
        raise SystemExit(f"SINE repo not found under {root}.")
    if not weights.is_file():
        raise SystemExit(f"Missing SINE checkpoint: {weights}")
    if not dinov2_weights.is_file():
        raise SystemExit(f"Missing DINOv2 weights: {dinov2_weights}")

    old_cwd = os.getcwd()
    os.chdir(root)
    sys.path.insert(0, str(root))
    try:
        from detectron2.modeling.postprocessing import sem_seg_postprocess
        from detectron2.structures import BitMasks, ImageList, Instances
        from dinov2.data.transforms import MaybeToTensor, make_normalize_transform
        from inference_fss.model.model import build_model
        from torchvision import transforms

        device = pick_device(args.device)
        model = build_model(
            SimpleNamespace(
                feat_chans=args.feat_chans,
                image_enc_use_fc=False,
                pt_model="dinov2",
                dinov2_size=args.dinov2_size,
                dinov2_weights=str(dinov2_weights),
                transformer_depth=args.transformer_depth,
                transformer_nheads=args.transformer_nheads,
                transformer_mlp_dim=args.transformer_mlp_dim,
                transformer_mask_dim=args.transformer_mask_dim,
                transformer_fusion_layer_depth=args.transformer_fusion_layer_depth,
                transformer_num_queries=args.transformer_num_queries,
                transformer_pre_norm=args.transformer_pre_norm,
                score_threshold=args.threshold,
            )
        )
        state_dict = torch.load(weights, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"[SINE] loaded | missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
        del state_dict
        model.to(device).eval()

        image_size = args.image_size
        pad_size = args.pad_size
        encoder_transform = transforms.Compose(
            [
                MaybeToTensor(),
                make_normalize_transform(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        def image_tensor(path: Path) -> torch.Tensor:
            image = Image.open(path).convert("RGB").resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32).copy()
            return torch.from_numpy(array).permute(2, 0, 1) / 255.0

        def mask_tensor(path: Path) -> torch.Tensor:
            mask = Image.open(path).convert("L").resize(
                (image_size, image_size), Image.Resampling.NEAREST
            )
            return torch.from_numpy((np.asarray(mask) > 0).astype(np.float32))

        def pad_img(value: torch.Tensor) -> torch.Tensor:
            height, width = value.shape[-2:]
            if height > pad_size or width > pad_size:
                raise ValueError(
                    f"SINE pad size {pad_size} is smaller than input {height}x{width}"
                )
            return F.pad(value, (0, pad_size - width, 0, pad_size - height))

        def make_instance(size: tuple[int, int], mask: torch.Tensor):
            instances = Instances(size)
            instances.gt_classes = torch.tensor([0], dtype=torch.int64)
            bit_masks = BitMasks(pad_img(mask[None, ...]))
            instances.gt_masks = bit_masks.tensor
            instances.gt_boxes = bit_masks.get_bounding_boxes()
            instances.ins_ids = torch.tensor([0], dtype=torch.int64)
            return instances

        def build_reference_cache(
            support_row: dict,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            support_img = image_tensor(support_row["image"])
            support_mask = mask_tensor(support_row["mask"])
            ref_inputs = [
                {
                    "image": pad_img(encoder_transform(support_img)),
                    "height": image_size,
                    "width": image_size,
                    "instances": make_instance(
                        (image_size, image_size), support_mask
                    ),
                }
            ]
            ref_enc_images = ImageList.from_tensors(
                [item["image"].to(model.device) for item in ref_inputs],
                size_divisibility=model.image_encoder.patch_size,
            )
            ref_features = torch.cat(
                [
                    model.get_enc_embs(
                        ref_enc_images.tensor[start : start + args.support_batch_size]
                    )
                    for start in range(0, len(ref_inputs), args.support_batch_size)
                ],
                dim=0,
            )
            ref_instances = [
                item["instances"].to(model.device) for item in ref_inputs
            ]
            ref_targets = model.prepare_targets(ref_instances, ref_enc_images)
            ref_masks = torch.cat(
                [target["masks"][None, ...] for target in ref_targets], dim=0
            )
            ref_labels = torch.cat(
                [target["labels"] for target in ref_targets], dim=0
            )
            return ref_features, ref_masks, ref_labels

        def predict_with_cache(reference, query_dict: dict) -> torch.Tensor:
            ref_features, ref_masks, ref_labels = reference
            tar_enc_images = ImageList.from_tensors(
                [query_dict["image"].to(model.device)],
                size_divisibility=model.image_encoder.patch_size,
            )
            tar_features = model.get_enc_embs(tar_enc_images.tensor)
            outputs = model.transformer_decoder(
                ref_features=ref_features,
                tar_features=tar_features,
                ref_masks=ref_masks,
                ref_labels=ref_labels,
            )
            output = outputs[0]
            pred_ins_logits = output["pred_ins_logits"]
            pred_ins_masks = output["pred_ins_masks"]
            ref_sem_labels = output["ref_sem_labels"]
            size = tar_enc_images.image_sizes[0]
            pred_ins_masks = F.interpolate(
                pred_ins_masks,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            height = query_dict.get("height", size[0])
            width = query_dict.get("width", size[1])
            new_size = (
                query_dict["instances"].image_size[0],
                query_dict["instances"].image_size[1],
            )
            if model.sem_seg_postprocess_before_inference:
                pred_ins_masks = sem_seg_postprocess(
                    pred_ins_masks[0], new_size, height, width
                )
                pred_ins_logits = pred_ins_logits.to(pred_ins_masks)[0]
                ref_sem_labels = ref_sem_labels.to(pred_ins_masks)[0]
            result = model.semantic_inference(
                pred_ins_masks, pred_ins_logits, ref_sem_labels
            )
            if not model.sem_seg_postprocess_before_inference:
                result = sem_seg_postprocess(result, new_size, height, width)
            return result

        cache: dict[str, object] = {"reference": None}

        def reset() -> None:
            cache["reference"] = None

        def predict_one(support_row, query_row):
            use_amp = args.amp and device.type == "cuda"
            with torch.inference_mode():
                if cache["reference"] is None:
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            cache["reference"] = build_reference_cache(support_row)
                    else:
                        cache["reference"] = build_reference_cache(support_row)
                query_img = image_tensor(query_row["image"])
                # The query mask is never fed to the model; the Instances
                # container only records the image size for output resize.
                query_dict = {
                    "image": pad_img(encoder_transform(query_img)),
                    "height": image_size,
                    "width": image_size,
                    "instances": make_instance(
                        (image_size, image_size),
                        torch.zeros((image_size, image_size)),
                    ),
                }
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        result = predict_with_cache(cache["reference"], query_dict)
                else:
                    result = predict_with_cache(cache["reference"], query_dict)
            scores = result.detach().float().cpu().numpy().squeeze()
            return scores > args.threshold

        run_tasks(args, "SINE", predict_one, reset=reset)
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
