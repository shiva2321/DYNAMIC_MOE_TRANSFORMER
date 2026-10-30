#!/bin/bash
cd /d/shm_um
export PYTHONPATH=/d/shm_um
for s in mix drift seq; do python -m unimoe.p4 small-tune $s > results/unimoe/p4_small_tune_$s.log 2>&1; done
python -m unimoe.p4 small-eval > results/unimoe/p4_small_eval.log 2>&1
