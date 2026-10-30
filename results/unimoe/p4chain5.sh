#!/bin/bash
cd /d/shm_um
export PYTHONPATH=/d/shm_um
python -m unimoe.p4_refusal run --systems UG0 UPLUS --seeds 500 501 502 > results/unimoe/p4_refusal_density.log 2>&1
python -m unimoe.p4 small-start > results/unimoe/p4_small_start.log 2>&1
