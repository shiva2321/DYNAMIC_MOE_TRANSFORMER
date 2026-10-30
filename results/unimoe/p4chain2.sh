#!/bin/bash
cd /d/shm_um
export PYTHONPATH=/d/shm_um
python -m unimoe.p4 oracle-lda > results/unimoe/p4_oracle_lda.log 2>&1
for s in seq mix drift; do python -m unimoe.p4 main-tune $s > results/unimoe/p4_main_tune_$s.log 2>&1; done
