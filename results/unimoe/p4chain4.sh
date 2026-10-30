#!/bin/bash
cd /d/shm_um
export PYTHONPATH=/d/shm_um
python -m unimoe.p4 main-run --seeds 503 504 > results/unimoe/p4_main_confirm.log 2>&1
python -m unimoe.experiment run --seeds 503 504 --schedules seq mix drift --systems U-real replay_eqmem knn_eqmem lda joint oracle_experts --runs runs_v2 --per-schedule > results/unimoe/run_v2_confirm.log 2>&1
python -m unimoe.p4_refusal run --systems UG0 UPLUS --seeds 503 504 > results/unimoe/p4_refusal_confirm.log 2>&1
