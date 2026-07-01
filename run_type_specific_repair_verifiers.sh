#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-${SCRIPT_DIR}}"
INPUT="${INPUT:-${ROOT}/outdir/repair_presence_verifier/leakage_free_repair_signature_best_ensemble_sameproject_samecwe_overlap2.jsonl}"
OUTDIR="${OUTDIR:-${ROOT}/outdir/repair_presence_verifier}"
MODEL="${MODEL:-glm-5.1}"
THREADS="${THREADS:-2}"
MAX_GEN_LENGTH="${MAX_GEN_LENGTH:-768}"
TAG="${TAG:-best_ensemble_sameproject_samecwe_overlap2_glm51}"

TYPES=(
  api_replacement
  error_handling
  null_check
  added_guard
  state_or_lifetime_repair
)

mkdir -p "${OUTDIR}"

for repair_type in "${TYPES[@]}"; do
  output="${OUTDIR}/${repair_type}_${TAG}.jsonl"
  summary="${OUTDIR}/${repair_type}_${TAG}_summary.json"
  log="${OUTDIR}/${repair_type}_${TAG}.log"

  if [[ -s "${output}" ]]; then
    echo "[skip] ${repair_type}: output exists: ${output}"
    continue
  fi

  echo "[run] ${repair_type}"
  nohup python "${ROOT}/run_type_specific_repair_verifier.py" \
    --input "${INPUT}" \
    --output "${output}" \
    --summary-json "${summary}" \
    --repair-type "${repair_type}" \
    --model "${MODEL}" \
    --temperature 0 \
    --max_gen_length "${MAX_GEN_LENGTH}" \
    --num-threads "${THREADS}" \
    > "${log}" 2>&1

  echo "[done] ${repair_type}: ${output}"
done

echo "[done] all type-specific repair verifiers"
