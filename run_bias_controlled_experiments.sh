#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-${SCRIPT_DIR}}"
INPUT="${INPUT:-${ROOT}/data/primevul_test_paired_labeled.jsonl}"
OUTDIR="${OUTDIR:-${ROOT}/outdir/empirical_study/bias_controlled_glm51_test870}"
MODEL="${MODEL:-glm-5.1}"
THREADS="${THREADS:-2}"
MAX_GEN_LENGTH="${MAX_GEN_LENGTH:-512}"
LIMIT="${LIMIT:-0}"

VARIANTS=(
  no_code_prior
  metadata_only
  skeleton_only
  code_author_fewshot
  code_author_flipped_fewshot
)

mkdir -p "${OUTDIR}"

for variant in "${VARIANTS[@]}"; do
  output="${OUTDIR}/${MODEL}_${variant}.jsonl"
  log="${OUTDIR}/${MODEL}_${variant}.log"

  if [[ -s "${output}" ]]; then
    echo "[skip] ${variant}: output exists"
    continue
  fi

  echo "[run] ${variant}"
  cmd=(
    python "${ROOT}/run_bias_controlled_prompt.py"
    --input "${INPUT}"
    --output-dir "${OUTDIR}"
    --variant "${variant}"
    --model "${MODEL}"
    --temperature 0
    --max_gen_length "${MAX_GEN_LENGTH}"
    --num-threads "${THREADS}"
  )
  if [[ "${LIMIT}" != "0" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi
  nohup "${cmd[@]}" > "${log}" 2>&1
  echo "[done] ${variant}"
done

python "${ROOT}/summarize_bias_controlled_results.py" \
  --input-dir "${OUTDIR}" \
  --output-csv "${OUTDIR}/bias_controlled_metrics.csv" \
  --summary-json "${OUTDIR}/bias_controlled_summary.json"

echo "[done] bias-controlled experiments"
