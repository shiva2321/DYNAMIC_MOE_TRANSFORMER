#!/bin/bash
cd /d/shm_um
export PYTHONPATH=/d/shm_um
until grep -q "drift chosen" results/unimoe/p4_main_tune_drift.log 2>/dev/null; do sleep 20; done
python -m unimoe.p4 main-run > results/unimoe/p4_main_run.log 2>&1
python -m unimoe.p4_refusal run --systems UG0 > results/unimoe/p4_refusal.log 2>&1
until grep -q "chosen threshold" results/unimoe/p4_capacity_tune.log 2>/dev/null; do sleep 30; done
python -m unimoe.p4 main-run >> results/unimoe/p4_main_run.log 2>&1
python -m unimoe.p4_refusal run --systems UPLUS >> results/unimoe/p4_refusal.log 2>&1
