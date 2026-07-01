

nohup bash -c '
cd ${SLICERAG_ROOT}

for variant in metadata_no_cwe; do
  echo "[run] ${variant}"
  python ${SLICERAG_ROOT}/run_bias_controlled_prompt.py \
    --input ${SLICERAG_ROOT}/data/claude-opus-4-7_metadata_no_cwe_EMPTY_remaining_round4_original.jsonl \
    --output-dir ${SLICERAG_ROOT}/outdir/empirical_study/bias_controlled_claude47_test870/rerun_empty_metadata_no_cwe_round5 \
    --variant ${variant} \
    --model claude-opus-4-7 \
    --temperature 0 \
    --max_gen_length 512 \
    --num-threads 2 \
    > ${SLICERAG_ROOT}/outdir/empirical_study/bias_controlled_claude47_test870/metadata_no_cwe_rerun_empty.log 2>&1
done
' > ${SLICERAG_ROOT}/outdir/empirical_study/bias_controlled_claude47_test870/run_bias_controls_empty2.log 2>&1 &




nohup bash -c '
cd ${SLICERAG_ROOT}

for variant in metadata_only; do
  echo "[run] ${variant}"
  python ${SLICERAG_ROOT}/run_bias_controlled_prompt.py \
    --input ${SLICERAG_ROOT}/data/claude-opus-4-7_metadata_only_EMPTY_remaining_round4.jsonl\
    --output-dir ${SLICERAG_ROOT}/outdir/empirical_study/bias_controlled_claude47_test870/rerun_empty_metadata_only_round5 \
    --variant ${variant} \
    --model claude-opus-4-7 \
    --temperature 0 \
    --max_gen_length 512 \
    --num-threads 2 \
    > ${SLICERAG_ROOT}/outdir/empirical_study/bias_controlled_claude47_test870/${variant}.log 2>&1 
done
' > ${SLICERAG_ROOT}/outdir/empirical_study/bias_controlled_claude47_test870/run_bias_controls_empty.log 2>&1 &


