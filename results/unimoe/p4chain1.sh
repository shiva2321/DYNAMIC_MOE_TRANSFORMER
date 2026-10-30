#!/bin/bash
cd /d/shm_um
export PYTHONPATH=/d/shm_um
python -m unimoe.p4 capacity-widths > results/unimoe/p4_capacity_widths.log 2>&1
python -m unimoe.p4 capacity-tune > results/unimoe/p4_capacity_tune.log 2>&1
python -m unimoe.p4 capacity-eval > results/unimoe/p4_capacity_eval.log 2>&1
