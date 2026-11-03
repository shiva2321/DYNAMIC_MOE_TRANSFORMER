#!/bin/bash
# GPU repeat of the wide baselines for milestone M1 (Job G-1).
# Mirrors m1/scored_run.sh; runs after the CPU timed pass, alone on the machine.
#   bash m1/gpu_run.sh run       five report seeds (810-814), ONE process at a time on logical processors 0 and 2
#   bash m1/gpu_run.sh report    the comparative report from finished CPU and GPU cells
# Every step is idempotent: skips valid, current records; resumes seamlessly after interruptions.

cd "$(dirname "$0")/.." || exit 1
PY=${PY:-/c/Users/Asta/AppData/Local/Programs/Python/Python312/python.exe}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ARMS="A3,A3-2,A4,A4-2,A6,A8"
BEDS="B1,B2,B3seq,B3mix,B3drift"
OUT=results/m1/scored_gpu
CPU_DIR=results/m1/scored
LOGS=results/m1/scored_gpu_logs
REPORT_SEEDS="810 811 812 813 814"
mkdir -p "$LOGS"

$PY -m m1.freeze check || exit 2

queue() {   # queue SEED CPUS LOGNAME
    $PY -u -m m1.gpu_repeat --split report --beds "$BEDS" --ns 10,30,100 --arms "$ARMS" --seeds "$1" \
        --out "$OUT" --tuning results/m1/tuning --budgets-from "$CPU_DIR" --cpus "$2" --threads 2 >> "$LOGS/$3.log" 2>&1
}

case "$1" in
run|all)
    echo "gpu repeat start $(date +%H:%M:%S)"
    for s in $REPORT_SEEDS; do
        queue "$s" 0,2 "gpu_$s"; echo "gpu seed $s done $(date +%H:%M:%S) rc=$?"
    done
    echo "gpu repeat end $(date +%H:%M:%S)"
    ;;
report)
    $PY -m m1.gpu_report --cpu "$CPU_DIR" --gpu "$OUT" --out results/m1/scored_gpu_report
    ;;
*)
    echo "usage: bash m1/gpu_run.sh run|report"; exit 1
    ;;
esac
