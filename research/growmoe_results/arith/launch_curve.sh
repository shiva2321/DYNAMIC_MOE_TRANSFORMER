#!/bin/bash
# Data-size sweep (how much data does it take to generalise?): FULL lengths, seed 0, dense MLP and transformer, test grid identical to the main runs.
# Waits for the sequential GPU queue to finish so that the GPU is not oversubscribed.
cd "$(dirname "$0")"
until grep -q "queue gpu_seq finished" logs/gpu_seq.log 2>/dev/null; do sleep 30; done
for n in 500 2000 4000 16000; do
  python run_arith.py run --protocol iid --regimes FULL --seeds 0 --learners MLP-L,Transformer --n-per-op $n --per-cell 200 --seen-per-op 500 --out results_curve/n$n
done
echo "queue curve finished at $(date)"
