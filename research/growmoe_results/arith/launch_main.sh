#!/bin/bash
# Main arithmetic runs (see run_arith.py). Four independent queues; each is resumable (finished runs are skipped).
#   usage: bash launch_main.sh gpu_iid | gpu_seq | cpu_iid | cpu_seq
cd "$(dirname "$0")"
N=8000; CELL=200; SEEN=500; SEQCELL=40
RUN="python run_arith.py run --n-per-op $N --per-cell $CELL --seen-per-op $SEEN --seq-cell $SEQCELL --out results"
case "$1" in
  gpu_iid)
    for regime in FULL HOLE SHORT; do
      for seed in 0 1; do
        $RUN --protocol iid --regimes $regime --seeds $seed --learners kNN,MLP-S,MLP-L,Oracle-op,Oracle-oppos,Transformer
      done
    done ;;
  gpu_seq)
    for regime in FULL HOLE SHORT; do
      for seed in 0 1; do
        $RUN --protocol seq --regimes $regime --seeds $seed --learners MLP-L-seq,MLP-L-replay,Oracle-op
      done
    done ;;
  cpu_iid)
    for seed in 0 1; do
      $RUN --protocol iid --regimes FULL --seeds $seed --learners GrowMoE,GrowMoE-h512-s5-r20000
    done
    for regime in HOLE SHORT; do
      $RUN --protocol iid --regimes $regime --seeds 0 --learners GrowMoE-h512-s5-r20000
    done ;;
  cpu_seq)
    for seed in 0 1; do
      $RUN --protocol seq --regimes FULL --seeds $seed --learners GrowMoE-h512-s5-r20000+s3,GrowMoE-h512-s5-r20000,ReplayNet,FineTuneNet
    done
    for regime in HOLE SHORT; do
      $RUN --protocol seq --regimes $regime --seeds 0 --learners GrowMoE-h512-s5-r20000+s3
    done
    $RUN --protocol seq --regimes FULL --seeds 0 --learners GrowMoE+s3 ;;
esac
echo "queue $1 finished at $(date)"
