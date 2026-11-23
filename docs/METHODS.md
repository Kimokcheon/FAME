# FAME Method Taxonomy

FAME reports 34 methods: 27 large-model methods and 7 conventional baselines. Every method is grouped by **where the one-shot support pair enters the pipeline**. Conventional baselines are evaluated inside the same three paradigms and serve as reference points rather than competing large-model entries.

## Similarity-Based

The support foreground is summarized as dense features or prototypes, and each query location is classified by its similarity to them.

- **Matching:** PANet, FAMNet, INSID3, RSRM.
- **CLIP head:** MaskCLIP, ProxyCLIP, SCLIP.

## Prompt-Based

The support pair is converted into a prompt (box, mask, point, or text) for a foundation segmentation model.

- **SAM:** GF-SAM, Matcher, SAM-MPA, MAUP, ProtoSAM, FoB, MedPLIB, LISA, GLaMM, F-LMM, Text4Seg, READ, STAMP, LENS.
- **SAM2:** SANSA, MedPro.
- **SAM3:** CG-ICS, MedSAM3.

## Interaction-Based

A segmentation network consumes the support pair and the query jointly and predicts the query mask directly.

- **Transformer:** SegGPT, SINE.
- **CNN:** UniverSeg, Tyche, DIFD, U-Net, nnU-Netv2.
- **Diffusion:** DiffewS, SD-FSMIS.

## Method Runners in This Repository

`scripts/methods/` provides thin adapters that read FAME episodes and write masks in the layout expected by `scripts/evaluate_split.py`. Third-party repositories and checkpoints are installed separately.

| Runner | Methods |
| --- | --- |
| `run_sam_family.py` | SAM, MedSAM, MobileSAM, SAM 2.1, MedSAM2 (box from support registration) |
| `run_sam_mpa.py` | SAM-MPA, MedSAM-MPA |
| `run_matcher.py` | Matcher |
| `run_gfsam.py` | GF-SAM |
| `run_clip_head.py` | MaskCLIP, SCLIP, ProxyCLIP |
| `run_sansa.py` | SANSA |
| `run_seggpt.py` | SegGPT (`BAAI/seggpt-vit-large` via Transformers) |
| `run_insid3.py` | INSID3 (frozen DINOv3 dense correspondence) |
| `run_sine.py` | SINE (DINOv2 + matching transformer) |
| `run_diffews.py` | DiffewS (diffusion-based few-shot segmentation) |
| `run_cgics.py` | CG-ICS (SAM3 + local Qwen3-VL, fast prompt cache) |
| `run_medsam3.py` | MedSAM3 (SAM3 + per-task MedSAM3 LoRA, target text prompt) |
| `run_maup.py` | MAUP |
| `run_medpro.py` | MedPro (SAM2 propagation + DINOv2 mask selection) |
| `run_tyche.py` | Tyche |
| `run_universeg.py` | UniverSeg |
| `run_unet.py` | Episodic U-Net |

Single-support runners use `--support-index 0` by default, which corresponds to the paper's one-shot (K=1) setting. The released `k10_seed0` episodes provide ten fixed support groups; pass another index to switch the pair. Methods that require task-specific adaptation (for example MedSAM3 LoRA-SFT, STAMP head fine-tuning, or MLLM LoRA training) follow their official repositories; the example wrappers under `scripts/examples/` show the FAME input/output contract.

`run_cgics.py` loads the local Qwen3-VL model from the CG-ICS repository (no API server), supports `--sam3-backend medsam3`, and reproduces the benchmark's fast prompt cache with `--bon 1 --num-loops 0 --max-restarts 0`; omitting those flags runs the full tree search. `run_medsam3.py` reads the per-task `target_display` from `task_info.json`; pass `--medsam3-lora-root` so each task loads its own `<root>/<task>/output/lora/last_lora_weights.pt` produced by the official LoRA training procedure (or `--medsam3-lora` for a single task). `run_insid3.py` needs the gated DINOv3 checkpoint under `<insid3-root>/pretrain/`; `run_sine.py` depends on detectron2 (the official SINE requirement); `run_diffews.py` needs the separate DiffewS environment (diffusers 0.25, transformers 4.36) and its local Stable Diffusion 2.1 checkpoints.

Methods without a bundled runner are not silently replaced: PANet, FAMNet, RSRM, ProtoSAM, FoB, MedPLIB, LISA, GLaMM, F-LMM, Text4Seg, READ, STAMP, LENS, DIFD, nnU-Netv2, and SD-FSMIS were run with their official repositories on a separate machine; their inference code is not part of this repository. Run those through their official repositories and write masks into the same `pred_masks/` layout. ProtoSAM and MedPLIB predictions in the paper came from the original qualitative package rather than a benchmark-machine rerun.

## Reporting

- Zero-shot uses only the query image and target prompt.
- IID few-shot uses the released support image-mask pairs.
- OOD uses source-task support masks and target-domain queries.
- Positive images report Dice.
- Target-absent negative images report image-level specificity, `TN/(TN+FP)`.
