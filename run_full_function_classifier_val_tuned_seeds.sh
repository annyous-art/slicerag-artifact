#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-${SCRIPT_DIR}}
MODEL=${MODEL:-${SLICERAG_EMBED_MODEL:-microsoft/graphcodebert-base}}
OUT_ROOT=${OUT_ROOT:-${ROOT}/outdir/full_function_classifier_val_tuned}
TRAIN=${TRAIN:-${ROOT}/data/primevul_train_paired_labeled.jsonl}
TEST=${TEST:-${ROOT}/data/primevul_test_paired_labeled.jsonl}
SEEDS=${SEEDS:-"12345 23456 34567"}
EPOCHS=${EPOCHS:-5}
VAL_RATIO=${VAL_RATIO:-0.1}
THRESHOLD_MODE=${THRESHOLD_MODE:-val_f1}
SELECTION_METRIC=${SELECTION_METRIC:-val_f1}

mkdir -p "${OUT_ROOT}"

for seed in ${SEEDS}; do
  out_dir="${OUT_ROOT}/graphcodebert_fullfunc_${THRESHOLD_MODE}_seed${seed}"
  mkdir -p "${out_dir}"
  echo "[run] seed=${seed} threshold=${THRESHOLD_MODE} selection=${SELECTION_METRIC} out=${out_dir}"
  python "${ROOT}/train_full_function_classifier.py" \
    --train "${TRAIN}" \
    --test "${TEST}" \
    --model "${MODEL}" \
    --output-dir "${out_dir}" \
    --epochs "${EPOCHS}" \
    --batch-size 8 \
    --eval-batch-size 16 \
    --gradient-accumulation-steps 1 \
    --max-length 512 \
    --lr 2e-5 \
    --weight-decay 0.01 \
    --warmup-ratio 0.06 \
    --val-ratio "${VAL_RATIO}" \
    --threshold-mode "${THRESHOLD_MODE}" \
    --selection-metric "${SELECTION_METRIC}" \
    --seed "${seed}" \
    --device cuda \
    --fp16 \
    --save-model \
    > "${out_dir}/train.log" 2>&1
done
