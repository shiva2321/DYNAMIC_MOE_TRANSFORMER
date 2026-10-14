#!/bin/bash
# Second CPU queue for the operator-sequential GrowMoE runs (runs the items the first CPU queue reaches last, in the opposite order; finished runs are skipped).
cd "$(dirname "$0")"
N=8000; CELL=200; SEEN=500; SEQCELL=40
RUN="python run_arith.py run --n-per-op $N --per-cell $CELL --seen-per-op $SEEN --seq-cell $SEQCELL --out results"
for regime in SHORT HOLE; do
  $RUN --protocol seq --regimes $regime --seeds 0 --learners GrowMoE-h512-s5-r20000+s3
done
$RUN --protocol seq --regimes FULL --seeds 0 --learners GrowMoE+s3
for regime in SHORT HOLE; do
  $RUN --protocol iid --regimes $regime --seeds 0 --learners GrowMoE
done
echo "queue cpu_seq2 finished at $(date)"
