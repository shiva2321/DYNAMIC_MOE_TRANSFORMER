#!/bin/bash
cd /d/shm_um
export PYTHONPATH=/d/shm_um
python -m unimoe.experiment tune --schedule mix > results/unimoe/tune_mix.log 2>&1
python -m unimoe.experiment tune --schedule drift > results/unimoe/tune_drift.log 2>&1
mkdir -p results/unimoe/runs_v2
cp -r results/unimoe/runs/unimoe-seq results/unimoe/runs_v2/
python -m unimoe.experiment run --seeds 500 501 502 --schedules mix drift --runs runs_v2 --per-schedule > results/unimoe/run_v2.log 2>&1
