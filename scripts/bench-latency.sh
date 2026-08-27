#!/usr/bin/env bash
# Run `vllm bench latency` and flag the two things that silently invalidate the result:
# CPU contention from outside this pod during the measured window, and a wide spread.
#
#   bash scripts/bench-latency.sh
#   bash scripts/bench-latency.sh --threads 192 --iters 20
#   bash scripts/bench-latency.sh -- --input-len 128 --output-len 128
#
# OMP_NUM_THREADS defaults to `nproc --all`, so a wide OpenMP barrier waits on whichever
# thread lands on a core a neighbouring container is busy with. That reads as a regression.

set -o pipefail

usage() { sed -e '1d' -e '/^[^#]/Q' "$0" | sed 's/^#\s\?//'; }

REPO="${REPO:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)}"
THREADS="${THREADS:-8}"
ITERS="${ITERS:-10}"
MODEL="${MODEL:-ibm-granite/granite-3.3-8b-instruct}"
BUSY_WARN="${BUSY_WARN:-4}"        # warn above this many externally-busy CPUs
SPREAD_WARN="${SPREAD_WARN:-1.5}"  # warn above this p10->p90 spread, percent
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads) THREADS="$2"; shift 2 ;;
        --iters)   ITERS="$2";   shift 2 ;;
        --model)   MODEL="$2";   shift 2 ;;
        --repo)    REPO="$2";    shift 2 ;;
        --)        shift; EXTRA=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

cd "$REPO" 2>/dev/null || { echo "repo root not found: ${REPO:-<unset>}" >&2; exit 1; }

NCPU=$(nproc --all)
LOG="$(mktemp -t bench-latency-XXXXXX.log)"
SAMPLES="$(mktemp -t bench-samples-XXXXXX)"

# The accelerator takes one process at a time; a concurrent run corrupts device state and
# the compile cache, not just the timings.
for fd in /proc/[0-9]*/fd/*; do
    [[ -L "$fd" ]] || continue
    case "$(readlink "$fd" 2>/dev/null)" in
        /dev/vfio/*)
            pid="${fd#/proc/}"; pid="${pid%%/*}"
            [[ "$pid" == "$$" ]] && continue
            echo "ERROR: pid $pid ($(cat "/proc/$pid/comm" 2>/dev/null)) already holds an AIU handle." >&2
            echo "Refusing to run: concurrent Spyre runs invalidate both results." >&2
            exit 1 ;;
    esac
done

# /proc/stat is not namespaced in these pods and reports the whole host; the cgroup
# reports only this pod. Both must span the same interval or their difference is noise.
sample_both() {   # $1 = seconds -> "<host_cpus_busy> <pod_cpus_busy>"
    local u n s i t1 b1 t2 b2 c1 c2
    read -r _ u n s i _ < /proc/stat; b1=$((u+n+s)); t1=$((u+n+s+i))
    c1=$(awk '/^usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null)
    sleep "$1"
    read -r _ u n s i _ < /proc/stat; b2=$((u+n+s)); t2=$((u+n+s+i))
    c2=$(awk '/^usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null)
    awk -v b=$((b2-b1)) -v t=$((t2-t1)) -v n="$NCPU" -v c1="$c1" -v c2="$c2" -v s="$1" \
        'BEGIN{printf "%.2f %.2f", t ? n*b/t : 0, (c2-c1)/1e6/s}'
}

echo "host      : $(cat /proc/sys/kernel/hostname 2>/dev/null)  (${NCPU} CPUs)"
echo "loadavg   : $(cut -d' ' -f1-3 /proc/loadavg)"
PRE=$(sample_both 5 | cut -d' ' -f1)
echo "pre-check : ${PRE} of ${NCPU} host CPUs busy"
awk -v p="$PRE" -v w="$BUSY_WARN" \
    'BEGIN{if (p>w) print "  WARNING: host is already busy before the run started."}'

( while :; do sample_both 5; echo; done ) > "$SAMPLES" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; rm -f "$SAMPLES"' EXIT

echo "running   : OMP_NUM_THREADS=$THREADS iters=$ITERS model=$MODEL"
(
    source /etc/profile.d/ibm-aiu-setup.sh >/dev/null 2>&1
    export OMP_NUM_THREADS="$THREADS"
    uv run --no-sync vllm bench latency \
        --model "$MODEL" \
        --input-len 64 --output-len 64 --batch-size 1 \
        --num-iters-warmup 2 --num-iters "$ITERS" --max-model-len 128 \
        -cc.compile_sizes='[64,1]' "${EXTRA[@]}"
) > "$LOG" 2>&1
RC=$?
kill $SAMPLER 2>/dev/null

AVG=$(grep -oP 'Avg latency: \K[0-9.]+' "$LOG" | tail -1)
P10=$(grep -oP '10% percentile latency: \K[0-9.]+' "$LOG" | tail -1)
P90=$(grep -oP '90% percentile latency: \K[0-9.]+' "$LOG" | tail -1)
if [[ $RC -ne 0 || -z "$AVG" || -z "$P10" || -z "$P90" ]]; then
    echo "FAILED (exit $RC). Tail of $LOG:" >&2
    tail -30 "$LOG" >&2
    exit 1
fi

echo
printf 'avg latency   : %.3f s\n' "$AVG"
awk -v a="$AVG" -v p1="$P10" -v p9="$P90" -v sw="$SPREAD_WARN" 'BEGIN {
    sp = p9 - p1; pct = 100*sp/a
    printf "p10 -> p90    : %.3f -> %.3f s  (spread %.3f s, %.1f%%)\n", p1, p9, sp, pct
    if (pct > sw)
        printf "  WARNING: spread %.1f%% > %.1f%% -- noisy window, treat the mean as unreliable.\n", pct, sw
    else
        printf "  spread is tight (<= %.1f%%): clean measurement window.\n", sw
}'
awk -v w="$BUSY_WARN" -v n="$NCPU" -v threads="$THREADS" '
{ h+=$1; if ($1>hm) hm=$1; p+=$2; c++ }
END {
    if (!c) { print "host CPU      : (no samples -- run too short)"; exit }
    printf "host CPU busy : mean %.2f / max %.2f of %d CPUs (during the run)\n", h/c, hm, n
    printf "this pod      : mean %.2f CPUs\n", p/c
    ext = h/c - p/c
    if (ext <= w)
        printf "  host was quiet (~%.1f external CPUs): comparable across pods.\n", ext
    else if (threads > 32) {
        printf "  WARNING: ~%.1f CPUs of load from OUTSIDE this pod, and OMP_NUM_THREADS=%s is\n", ext, threads
        printf "           wide enough that a barrier almost certainly waits on a thread sharing a\n"
        printf "           contended core. Expect an inflated mean; retry at 8-32 threads.\n"
    } else {
        printf "  NOTE: ~%.1f CPUs of load from OUTSIDE this pod, but OMP_NUM_THREADS=%s is narrow\n", ext, threads
        printf "        enough to mostly dodge it. Trust the spread above over this number.\n"
    }
}' "$SAMPLES"
echo "full log: $LOG"
