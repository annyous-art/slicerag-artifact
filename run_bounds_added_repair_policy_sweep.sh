#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-${SCRIPT_DIR}}"
OUTDIR="${OUTDIR:-${ROOT}/outdir/repair_presence_verifier}"
PREDICTIONS_CSV="${PREDICTIONS_CSV:-${ROOT}/outdir/ensemble_error_patch_contrast/per_sample_predictions.csv}"
BASE_PRED_COLUMN="${BASE_PRED_COLUMN:-best_ensemble_pred}"
MODEL="${MODEL:-glm-5.1}"
THREADS="${THREADS:-2}"
MAX_GEN_LENGTH="${MAX_GEN_LENGTH:-768}"

# Each entry is tag:path. Tags are used in output filenames.
CANDIDATES=(
  "strict_sameproject_samecwe_overlap2:${OUTDIR}/leakage_free_repair_signature_best_ensemble_sameproject_samecwe_overlap2.jsonl"
  "samecwe:${OUTDIR}/leakage_free_repair_signature_best_ensemble_minrelneg_samecwe.jsonl"
  "sameproject:${OUTDIR}/leakage_free_repair_signature_best_ensemble_minrelneg_sameproject.jsonl"
  "wide:${OUTDIR}/leakage_free_repair_signature_best_ensemble_minrelneg.jsonl"
)

mkdir -p "${OUTDIR}"

for item in "${CANDIDATES[@]}"; do
  tag="${item%%:*}"
  input="${item#*:}"

  if [[ ! -s "${input}" ]]; then
    echo "[skip] ${tag}: missing candidate file: ${input}"
    continue
  fi

  bounds_output="${OUTDIR}/bounds_path_variable_${tag}_${MODEL}.jsonl"
  bounds_summary="${OUTDIR}/bounds_path_variable_${tag}_${MODEL}_summary.json"
  bounds_log="${OUTDIR}/bounds_path_variable_${tag}_${MODEL}.log"

  added_output="${OUTDIR}/added_guard_${tag}_${MODEL}.jsonl"
  added_summary="${OUTDIR}/added_guard_${tag}_${MODEL}_summary.json"
  added_log="${OUTDIR}/added_guard_${tag}_${MODEL}.log"

  merged_output="${OUTDIR}/merged_bounds_added_${tag}_${MODEL}.jsonl"
  merged_summary="${OUTDIR}/merged_bounds_added_${tag}_${MODEL}_summary.json"

  applied_csv="${OUTDIR}/applied_bounds_added_${tag}_${MODEL}.csv"
  applied_summary="${OUTDIR}/applied_bounds_added_${tag}_${MODEL}_summary.json"

  if [[ -s "${bounds_output}" ]]; then
    echo "[skip] ${tag}: bounds output exists"
  else
    echo "[run] ${tag}: bounds_path_variable"
    nohup python "${ROOT}/run_bounds_path_variable_verifier.py" \
      --input "${input}" \
      --output "${bounds_output}" \
      --summary-json "${bounds_summary}" \
      --model "${MODEL}" \
      --temperature 0 \
      --max_gen_length "${MAX_GEN_LENGTH}" \
      --num-threads "${THREADS}" \
      > "${bounds_log}" 2>&1
  fi

  if [[ -s "${added_output}" ]]; then
    echo "[skip] ${tag}: added_guard output exists"
  else
    echo "[run] ${tag}: added_guard"
    nohup python "${ROOT}/run_type_specific_repair_verifier.py" \
      --input "${input}" \
      --output "${added_output}" \
      --summary-json "${added_summary}" \
      --repair-type added_guard \
      --model "${MODEL}" \
      --temperature 0 \
      --max_gen_length "${MAX_GEN_LENGTH}" \
      --num-threads "${THREADS}" \
      > "${added_log}" 2>&1
  fi

  echo "[merge] ${tag}"
  python "${ROOT}/merge_repair_verifier_outputs.py" \
    --inputs "${bounds_output}" "${added_output}" \
    --output "${merged_output}" \
    --summary-json "${merged_summary}"

  echo "[apply] ${tag}"
  python "${ROOT}/apply_repair_presence_verifier.py" \
    --predictions-csv "${PREDICTIONS_CSV}" \
    --verifier-jsonl "${merged_output}" \
    --base-pred-column "${BASE_PRED_COLUMN}" \
    --output-csv "${applied_csv}" \
    --summary-json "${applied_summary}" \
    --min-confidence medium \
    --require-repair-present \
    --allow-repair-types bounds_or_shape_check added_guard

  echo "[done] ${tag}: ${applied_summary}"
done

echo "[done] bounds+added_guard repair policy sweep"
