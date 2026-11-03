#!/bin/bash
# Scored run of M1 on the report seeds (preregistration section 5; the plan is in docs/roadmap/M1_FREEZE.json "run_plan").
#   bash m1/scored_run.sh timed   five seeds with the full profile (timing), ONE process at a time on logical processors 0 and 2, nothing else running
#   bash m1/scored_run.sh det     five seeds with the deterministic profile, four workers of two threads each; never at the same time as "timed"
#   bash m1/scored_run.sh report  the report from the finished cells
# Every step is idempotent: the queue skips cells that are complete, valid and current, so the same command resumes after a power loss.
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-/c/Users/Asta/AppData/Local/Programs/Python/Python312/python.exe}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ALL="A1,A1q,A2,A3,A4,A5a,A5b,A6,A7,A7all,A8,A3-2,A4-2,A7-2,H"
BEDS="B2,B3seq,B3mix,B3drift,B1"
OUT=results/m1/scored
LOGS=results/m1/scored_logs
TIMED_SEEDS="810 811 812 813 814"
DET_SEEDS="815-819"
mkdir -p "$LOGS"

$PY -m m1.freeze check || exit 2          # the report seeds open only while the frozen files match the freeze

queue() {   # queue PROFILE SEEDS SHARD CPUS LOGNAME
    $PY -u -m m1.queue --split report --beds "$BEDS" --ns 10,30,100 --arms "$ALL" --seeds "$2" --out "$OUT" --profile "$1" \
        --tuning results/m1/tuning --shard "$3" --cpus "$4" --threads 2 >> "$LOGS/$5.log" 2>&1
}

case "$1" in
timed)
    echo "timed start $(date +%H:%M:%S)"
    for s in $TIMED_SEEDS; do
        queue full "$s" 0/1 0,2 "timed_$s"; echo "seed $s done $(date +%H:%M:%S) rc=$?"
    done
    echo "timed end $(date +%H:%M:%S)"
    ;;
det)
    echo "det start $(date +%H:%M:%S)"
    queue det "$DET_SEEDS" 0/4 0,2  det_w0 &
    queue det "$DET_SEEDS" 1/4 4,6  det_w1 &
    queue det "$DET_SEEDS" 2/4 8,10 det_w2 &
    queue det "$DET_SEEDS" 3/4 1,3  det_w3 &
    wait
    echo "det end $(date +%H:%M:%S)"
    ;;
report)
    $PY -m m1.report --cells "$OUT" --out results/m1/scored_report --split report
    ;;
*)
    echo "usage: bash m1/scored_run.sh timed|det|report"; exit 1
    ;;
esac
