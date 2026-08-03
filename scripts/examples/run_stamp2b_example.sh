#!/usr/bin/env bash
set -euo pipefail

: "${FAME_RELEASE:?Set FAME_RELEASE to the downloaded FAME dataset root.}"
: "${STAMP_ROOT:?Set STAMP_ROOT to the official STAMP repository.}"
: "${STAMP_CKPT:?Set STAMP_CKPT to the STAMP-2B checkpoint.}"

TASK="${TASK:-ISIC2018__ISIC2018_skin_lesion}"
SETTING="${SETTING:-k10_seed0}"
METHOD="${METHOD:-STAMP-2B}"
OUT_ROOT="${OUT_ROOT:-outputs/${METHOD}/${SETTING}/${TASK}}"
TEST_JSONL="${FAME_RELEASE}/FAME_benchmark/episodes/${SETTING}/${TASK}/test.jsonl"
SUPPORT_JSONL="${FAME_RELEASE}/FAME_benchmark/episodes/${SETTING}/${TASK}/support.jsonl"

mkdir -p "${OUT_ROOT}/pred_masks"

cat <<EOF
Run the official STAMP inference/training code here.

Required FAME inputs:
  support_jsonl=${SUPPORT_JSONL}
  test_jsonl=${TEST_JSONL}
  checkpoint=${STAMP_CKPT}

Write one binary mask per query case to:
  ${OUT_ROOT}/pred_masks

Then evaluate with:
  python scripts/evaluate_predictions.py \\
    --test-jsonl "${TEST_JSONL}" \\
    --pred-dir "${OUT_ROOT}/pred_masks" \\
    --out-dir "${OUT_ROOT}/eval" \\
    --method "${METHOD}" \\
    --task "${TASK}" \\
    --support-k 10 \\
    --release-root "${FAME_RELEASE}"
EOF

