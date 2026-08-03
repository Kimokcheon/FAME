# FAME Dataset Release

The benchmark data are released on Hugging Face:

https://huggingface.co/datasets/Kimokcheon/FAME_benchmark

The dataset repository contains:

- `protocols/FAME_benchmark_protocols.tar.gz`: JSON/JSONL protocols;
- `data_shards/*.parquet`: full image/mask pixel store;
- `viewer/fame_preview.parquet`: small visual preview;
- `metadata/*`: summaries and manifests;
- `tools/*`: dataset-side loading helpers.

The public code in this repository reads the extracted `FAME_benchmark/` protocol directory and the materialized image/mask files under `data/`.

