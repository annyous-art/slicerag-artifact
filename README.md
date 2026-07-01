# SliceRAG Empirical Study Artifact

This artifact is organized as an anonymized replication package for the SliceRAG empirical study. It is intended to let reviewers inspect the data flow, prompts, predictions, retrieval outputs, and analysis scripts behind the paper tables.

## Scope

The artifact supports the paper's diagnostic claims on a PrimeVul-style paired vulnerability benchmark. It does not contain API credentials or proprietary model endpoints. When a closed model is required, the scripts expect the user to configure the corresponding provider credentials locally.

## Main Inputs

- `data/primevul_test_paired_labeled.jsonl`: paired test data with diff-derived labels used by the analysis scripts.
- `outdir/`: cached retrieval outputs, parsed prompt predictions, verifier outputs, and analysis CSV/JSON summaries used by the paper.
- Raw train/test pair files and large model-output logs are not bundled in this lightweight artifact. They can be regenerated from the public benchmark and labeled with `build_diff_lines.py` and `mark_pair_diffs.py`; compact parsed predictions and summaries are included under `outdir/`.

## Repository Layout

- `data/`: compact benchmark input used by the analysis scripts.
- `label/`: sampled taxonomy annotations from the two manual annotators.
- `Baseline/`: fixed few-shot baseline runner and prompt utilities.
- `icl/`: function-level retrieved few-shot runner.
- `outdir/`: cached metrics, predictions, retrieval summaries, and table inputs.
- Top-level `build_*`, `run_*`, `analyze_*`, and `train_*` scripts: retrieval construction, prompting, verifier, router, and analysis entry points.

Top-level scripts are intentionally kept flat to preserve the exact command lines used in the experiments; the Core Scripts section groups them by function.

## Core Scripts

- `build_diff_lines.py`, `mark_pair_diffs.py`: generate diff-line labels for paired vulnerable/fixed functions.
- `build_chunk_index.py`, `build_query_chunks_sim_rank.py`: build balanced chunk indexes and query-time retrieval outputs.
- `build_patch_pair_index.py`, `build_query_patch_contrast.py`: build patch-pair retrieval and patch-contrast query outputs.
- `run_prompting_sliced_rag.py`, `run_patch_contrast_prompt.py`, `run_bias_controlled_prompt.py`: run prompt variants.
- `build_per_sample_feature_table.py`, `train_router.py`: construct per-sample features and router diagnostics.
- `run_verifier_prompt.py`, `run_repair_presence_verifier.py`, `run_type_specific_repair_verifier.py`: run verifier experiments.
- `analyze_pairwise_primevul_metrics.py`, `analyze_empirical_error_taxonomy.py`, `sample_empirical_taxonomy_cases.py`: reproduce pair-wise metrics, taxonomy, and manual-sample files.

## Model Registry

The paper uses the following model identifiers in the text and output filenames:

| Model label in paper | Role | Call period | Notes |
| --- | --- | --- | --- |
| `gpt-5.5` | motivating 100-sample observation, full baseline/RAG transfer check, disagreement verifiers | June 2026 | Closed provider endpoint; cached outputs are included under `data/` and `outdir/verifier_results/`. |
| `glm-5.1` | full 870-sample ablation matrix, bias controls, routers, repair verifier policy sweeps | June 2026 | Closed/open provider endpoint depending on deployment; cached outputs are included under `outdir/`. |
| `claude-opus-4-7` | auxiliary bias-control and retrieval sensitivity checks | June 2026 | Cached outputs are under `outdir/empirical_study/bias_controlled_claude47_test870/` and `outdir/icl_fixed_baseline_compatible_test870/`; not part of the full factorial replication. |
| `gemini-3.1-pro-preview` (`gemini3.1_pro` in legacy filenames) | auxiliary bias-control sensitivity checks and auxiliary pair-wise logs | June 2026 and prior experimental logs | Cached bias-control outputs are under `outdir/empirical_study/bias_controlled_gemini3.1pro_test870/`; the metadata-without-CWE run has refusal/unknown behavior and is not treated as main ablation evidence. |

All binary predictions are parsed from the first unambiguous `YES` or `NO` in the model response. Unknown or unparsable responses are tracked explicitly and explain the 868-row common parsed-prediction intersection used in some tables.

Auxiliary Claude/Gemini bias-control and retrieval logs are retained to check model-specific directional behavior. They are sensitivity checks, not a full cross-model factorial replication. Gemini metadata-without-CWE has refusal behavior and should not be compared as a fully parsed main result. Older subset/full logs are retained for historical context but are not used as main-paper ablation evidence.

## Representative Reproduction Commands

Build paired line labels:

```bash
python build_diff_lines.py \
  --input data/primevul_test_paired.jsonl \
  --output data/diff_lines.jsonl

python mark_pair_diffs.py \
  --file data/primevul_test_paired.jsonl \
  --diff-file data/diff_lines.jsonl
```

Analyze pair-wise metrics:

```bash
python analyze_pairwise_primevul_metrics.py \
  --features-csv outdir/ensemble_error_patch_contrast/per_sample_predictions.csv \
  --data-jsonl data/primevul_test_paired_labeled.jsonl \
  --output-dir outdir/empirical_study/pairwise_metrics
```

Analyze taxonomy and sample cases:

```bash
python analyze_empirical_error_taxonomy.py \
  --predictions-csv outdir/ensemble_error_patch_contrast/per_sample_predictions.csv \
  --base-pred-column best_ensemble_pred \
  --repair-verifier-jsonl outdir/repair_presence_verifier/merged_bounds_added_wide_glm-5.1.jsonl \
  --output-dir outdir/empirical_study/error_taxonomy

python sample_empirical_taxonomy_cases.py \
  --taxonomy-csv outdir/empirical_study/error_taxonomy/per_sample_error_taxonomy.csv \
  --data-jsonl data/primevul_test_paired_labeled.jsonl \
  --output-dir outdir/empirical_study/error_taxonomy/manual_samples \
  --per-taxonomy 10 \
  --max-func-chars 6000
```

Run bias-controlled prompts:

```bash
bash run_bias_controlled_experiments.sh
```

## Paper Table Mapping

- Main full-test table: `outdir/ensemble_error_patch_contrast/per_sample_predictions.csv` and method metric JSON/CSV files under `outdir/`.
- Data-flow table: produced from parsed prediction counts and pair validation in `analyze_pairwise_primevul_metrics.py`.
- Pair-wise table: `outdir/empirical_study/pairwise_metrics/*summary.json` and `*pairs.csv`.
- GPT-5.5 transfer and pair-wise check: `outdir/empirical_study/gpt55_pairwise/`.
- Function-level retrieved few-shot: `outdir/icl_fixed_baseline_compatible_test870/glm-5.1_std_cls_fewshotegTrue_top2_baseline_compatible.jsonl`, `outdir/icl_fixed_baseline_compatible_test870/claude-opus-4-7_std_cls_fewshotegTrue_top2_baseline_compatible.jsonl`, and `outdir/icl_fixed_baseline_compatible_test870/eval/`.
- Bias-control table: `outdir/empirical_study/bias_controlled_glm51_test870*/`, with GPT-5.5 sensitivity in `outdir/empirical_study/bias_controlled_gpt55_test870/`.
- Auxiliary Claude/Gemini bias controls: `outdir/empirical_study/bias_controlled_claude47_test870/` and `outdir/empirical_study/bias_controlled_gemini3.1pro_test870/`.
- Full-function GraphCodeBERT classifier: `outdir/full_function_classifier_val_f1/summary.json`, `outdir/full_function_classifier_val_pc/summary.json`, and the per-seed subdirectories under those folders.
- Router diagnostics: `outdir/router/`, `outdir/router_no_project_cwe/`, and `outdir/router_realistic/`.
- GPT-5.5 verifier all-disagreement run: `outdir/verifier_results/verifier_gpt55_v3_blinded_all_disagreement*`.
- Error taxonomy: `outdir/empirical_study/error_taxonomy/taxonomy_summary.csv`.
- Manual taxonomy sample set: `outdir/empirical_study/error_taxonomy/manual_samples/all_manual_samples.csv`.
- Final manual annotations: `label/all_manual_samples_annotator1*.csv`, `label/all_manual_samples_annotator2*.csv`, and `label/all_manual_samples_annotation_summary.json`.
- Repair verifier table: `outdir/repair_presence_verifier/*summary.json`.

## Manual Taxonomy Protocol

The raw sampled cases are stored under `outdir/empirical_study/error_taxonomy/manual_samples/`.
The final manual annotation files are stored under `label/` to avoid mixing generated samples with annotator outputs.
Each annotation file adds three reviewer-facing columns to sampled automatic categories:

- `manual_taxonomy_valid`: `yes`, `partial`, or `no`.
- `manual_refined_category`: a more specific semantic failure mode.
- `notes`: short explanation of the checked evidence.

The current paper reports this as a sanity check, not as a full inter-rater annotation study. The paper reports both annotators' direct/partial/invalid counts.

## Expected Environment

The experiments were run with Python 3.10/3.13 environments across local and server machines. A lightweight dependency list is provided in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Closed-model runs require user-provided API credentials through environment variables:

```bash
export SLICERAG_API_KEY=...
export SLICERAG_OPENAI_BASE_URL=https://api.example/v1
export SLICERAG_ANTHROPIC_BASE_URL=https://api.example/claude
```

Path-dependent commands use environment variables rather than machine-specific absolute paths:

```bash
export SLICERAG_ROOT="$(pwd)"
export SLICERAG_EMBED_MODEL=microsoft/graphcodebert-base
```

If using a local checkpoint, set `SLICERAG_EMBED_MODEL` to that local path before running retrieval or classifier scripts. Historical output files may contain the placeholder `${SLICERAG_ROOT}` or `${SLICERAG_EMBED_MODEL}` where the original machine path was redacted.

## Anonymization Checklist

- Remove API keys, usernames, hostnames, and absolute server paths before public release.
- Keep relative paths in commands where possible.
- Replace proprietary endpoint URLs with provider/model identifiers and dates.
- Exclude `.env`, `*.log`, `__pycache__/`, `*.pyc`, and LaTeX build intermediates from the submitted artifact.
- Include cached prediction outputs when model reruns are expensive or unavailable.
