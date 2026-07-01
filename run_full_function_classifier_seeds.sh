#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-${SCRIPT_DIR}}
MODEL=${MODEL:-${SLICERAG_EMBED_MODEL:-microsoft/graphcodebert-base}}
OUT_ROOT=${OUT_ROOT:-${ROOT}/outdir/full_function_classifier}
TRAIN=${TRAIN:-${ROOT}/data/primevul_train_paired_labeled.jsonl}
TEST=${TEST:-${ROOT}/data/primevul_test_paired_labeled.jsonl}
SEEDS=${SEEDS:-"12345 23456 34567"}

mkdir -p "${OUT_ROOT}"

for seed in ${SEEDS}; do
  out_dir="${OUT_ROOT}/graphcodebert_fullfunc_seed${seed}"
  mkdir -p "${out_dir}"
  echo "[run] seed=${seed} out=${out_dir}"
  python "${ROOT}/train_full_function_classifier.py" \
    --train "${TRAIN}" \
    --test "${TEST}" \
    --model "${MODEL}" \
    --output-dir "${out_dir}" \
    --epochs 3 \
    --batch-size 8 \
    --eval-batch-size 16 \
    --gradient-accumulation-steps 1 \
    --max-length 512 \
    --lr 2e-5 \
    --weight-decay 0.01 \
    --warmup-ratio 0.06 \
    --seed "${seed}" \
    --device cuda \
    --fp16 \
    --save-model \
    > "${out_dir}/train.log" 2>&1
done
