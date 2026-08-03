# FAME Evaluation Method Groups

This public repository keeps a stable benchmark interface and avoids exposing local exploratory variants.

## Specialists

Task-specific supervised segmentation models trained or adapted with target-task masks. FAME reports them as specialist baselines, not as promptable zero-shot models.

## SAM-Based Methods

Promptable segmentation models that use boxes, points, masks, or support-derived prompts. For support-based variants, the 10 positive support masks are the only target-task adaptation signal unless the original method defines a trainable recipe.

## CLIP-Based Methods

Open-vocabulary or dense vision-language baselines. Training-free CLIP-style methods should stay separate from few-shot trainable heads.

## MLLM-Based Methods

Large multimodal models that consume a disease/target prompt and produce or condition a segmentation mask. For few-shot trainable variants, use official or paper-consistent trainable modules and label the adaptation clearly.

## Universal Few-Shot Segmenters

Methods that directly consume support image-mask pairs and a query image without task-specific gradient updates.

## Reporting

- Zero-shot uses only the query image and target prompt.
- IID few-shot uses 10 positive support image-mask pairs.
- OOD uses 10 source-task support masks and target-domain queries.
- Positive images report Dice.
- Target-absent negative images report image-level specificity, `TN/(TN+FP)`.

