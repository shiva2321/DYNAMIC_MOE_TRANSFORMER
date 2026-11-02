#!/bin/bash
# Layer R on the report seeds: embed (cached, resumable), then the recognizer lab (one JSON per cell, resumable). Relaunch after a power cut: bash results/fsmoe/chain_R.sh
cd /d/spherical_hyperspace_memory || exit 1
SEEDS="710 711 712 713 714 715 716 717 718 719"
python -W ignore -m fsmoe.embed_bed --seeds $SEEDS >> results/fsmoe/logs/embed_report.log 2>&1
python -W ignore -m fsmoe.recog_lab run --seeds $SEEDS >> results/fsmoe/logs/R_report.log 2>&1
echo "chain_R done $(date)" >> results/fsmoe/logs/R_report.log
