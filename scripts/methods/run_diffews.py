"""Official DiffewS (diffusion-based few-shot segmentation) over FAME episodes."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _common import add_common_args, pick_device, run_tasks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official DiffewS on FAME episodes.")
    parser.add_argument("--diffews-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to <root>/weight/stable-diffusion-2-1-ref8inchannels-tag4inchannels.",
    )
    parser.add_argument(
        "--unet-checkpoint",
        type=Path,
        default=None,
        help="Defaults to <root>/weight/coco_fold0.",
    )
    parser.add_argument(
        "--scheduler",
        type=Path,
        default=None,
        help="Defaults to <root>/scheduler_1.0_1.0.",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--denoise-steps", type=int, default=1)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--relative-threshold",
        type=float,
        default=0.25,
        help="Per-image threshold as a fraction of max(score); 0 uses --threshold.",
    )
    parser.add_argument("--test-timestep", type=int, default=1)
    parser.add_argument(
        "--half-precision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FP16 weights (default: on).",
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

    root = args.diffews_root.resolve()
    checkpoint = (
        args.checkpoint
        or root / "weight" / "stable-diffusion-2-1-ref8inchannels-tag4inchannels"
    ).resolve()
    unet_checkpoint = (args.unet_checkpoint or root / "weight" / "coco_fold0").resolve()
    scheduler_checkpoint = (args.scheduler or root / "scheduler_1.0_1.0").resolve()
    required = {
        checkpoint / "vae" / "diffusion_pytorch_model.safetensors": "base VAE",
        checkpoint / "text_encoder" / "model.safetensors": "text encoder",
        unet_checkpoint / "unet" / "diffusion_pytorch_model.safetensors": "U-Net",
        scheduler_checkpoint / "scheduler_config.json": "scheduler",
    }
    missing = [f"{name}: {path}" for path, name in required.items() if not path.is_file()]
    if missing:
        raise SystemExit("DiffewS checkpoints are incomplete:\n" + "\n".join(missing))

    old_cwd = os.getcwd()
    os.chdir(root)
    sys.path.insert(0, str(root))
    try:
        from diffusers import AutoencoderKL
        from diffews.marigold_pipeline_rgb_latent_noise import (
            MarigoldPipelineRGBLatentNoise,
        )
        from diffews.models.unet_2d_condition import MyUNet2DConditionModel
        from marigold.util.scheduler_customized import DDIMSchedulerCustomized
        from transformers import CLIPTextModel, CLIPTokenizer

        device = pick_device(args.device)
        dtype = torch.float16 if args.half_precision else torch.float32
        print(f"[DiffewS] device={device} dtype={dtype}", flush=True)

        unet = MyUNet2DConditionModel.from_pretrained(
            str(unet_checkpoint),
            subfolder="unet",
            torch_dtype=dtype,
            local_files_only=True,
        )
        vae = AutoencoderKL.from_pretrained(
            str(checkpoint),
            subfolder="vae",
            torch_dtype=dtype,
            local_files_only=True,
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            str(checkpoint), subfolder="tokenizer", local_files_only=True
        )
        text_encoder = CLIPTextModel.from_pretrained(
            str(checkpoint),
            subfolder="text_encoder",
            torch_dtype=dtype,
            local_files_only=True,
        )
        scheduler = DDIMSchedulerCustomized.from_pretrained(
            str(scheduler_checkpoint), local_files_only=True
        )
        pipe = MarigoldPipelineRGBLatentNoise(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_embeds=None,
            text_encoder=text_encoder,
            image_encoder=None,
            image_projector=None,
            controlnet=None,
            customized_head=None,
        ).to(device)
        pipe.unet.eval()
        pipe.vae.eval()
        pipe.text_encoder.eval()
        pipe.test_timestep = args.test_timestep

        def image_tensor(path: Path) -> torch.Tensor:
            image = Image.open(path).convert("RGB").resize(
                (args.image_size, args.image_size), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
            array = array * 2.0 - 1.0
            return torch.from_numpy(array).permute(2, 0, 1).contiguous()

        def mask_tensor(path: Path) -> torch.Tensor:
            mask = Image.open(path).convert("L").resize(
                (args.image_size, args.image_size), Image.Resampling.NEAREST
            )
            array = (np.asarray(mask) > 0).astype(np.float32) * 2.0 - 1.0
            return torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1).contiguous()

        cache: dict[str, torch.Tensor] = {"image": None, "mask": None}

        def reset() -> None:
            cache["image"] = None
            cache["mask"] = None

        def predict_one(support_row, query_row):
            if cache["image"] is None:
                cache["image"] = image_tensor(support_row["image"]).unsqueeze(0).to(dtype=dtype)
                cache["mask"] = mask_tensor(support_row["mask"]).unsqueeze(0).to(dtype=dtype)
            query_img = image_tensor(query_row["image"]).unsqueeze(0).to(dtype=dtype)
            pipe_inputs = [cache["image"], query_img, cache["mask"]]
            call_kwargs = {
                "denoising_steps": args.denoise_steps,
                "ensemble_size": args.ensemble_size,
                "processing_res": args.image_size,
                "batch_size": args.batch_size,
                "show_progress_bar": False,
                "mode": "seg",
                "rgb_paths": [str(query_row["image"])],
                "seed": 0,
            }
            use_amp = args.amp and device.type == "cuda"
            with torch.inference_mode():
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pipe_out = pipe(pipe_inputs, **call_kwargs)
                else:
                    pipe_out = pipe(pipe_inputs, **call_kwargs)

            seg_image = pipe_out.seg_colored
            if isinstance(seg_image, list):
                seg_image = seg_image[0]
            scores = (
                np.asarray(seg_image.convert("RGB"), dtype=np.float32).mean(axis=2)
                / 255.0
            )
            threshold = args.threshold
            if args.relative_threshold > 0:
                threshold = float(scores.max()) * args.relative_threshold
            return scores > threshold if threshold > 0 else scores > 0

        run_tasks(args, "DiffewS", predict_one, reset=reset)
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
