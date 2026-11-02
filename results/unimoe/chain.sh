#!/bin/bash
cd /d/shm_um
until [ -f results/unimoe/tuning.json ]; do sleep 20; done
PYTHONPATH=/d/shm_um python -m unimoe.experiment run --seeds 500 501 502 > results/unimoe/run.log 2>&1
