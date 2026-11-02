#!/bin/bash
cd /d/shm_um
PYTHONPATH=/d/shm_um python -m unimoe.p3 ablate > results/unimoe/p3_ablate.log 2>&1
PYTHONPATH=/d/shm_um python -m unimoe.p3 scale > results/unimoe/p3_scale.log 2>&1
