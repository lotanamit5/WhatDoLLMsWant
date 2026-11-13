#!/usr/bin/env bash
# Synthetic cars: send the cars jobs of scripts/slurms.sh to Heraclitus, right-sized and in chains.
#
# Run on the newton login node, from the repo root:
#     bash scripts/synthetic_cars_launch.sh dry    # print the plan, send nothing (`dry all`: the plan for all 8)
#     bash scripts/synthetic_cars_launch.sh        # use only the GPUs that are free now - nobody's job is removed
#     bash scripts/synthetic_cars_launch.sh all    # use all 8 GPUs - SLURM removes the `public` jobs running there
#
# Why not `bash scripts/slurms.sh`: the wrapper asks for 8 GPUs and 32 CPUs per job, which is the whole
# of Heraclitus, so the jobs would run one at a time (about 21 h) and every one of them would remove
# other people's jobs. A job is not faster on more GPUs (gemma-27B: 15 prompts/s on 1 GPU, 13.9 on 3).
# This script takes the SAME lines and adds to each only sbatch flags:
#     --gres=gpu:N   qwen-72B 2 GPUs (136 GB of weights), gemma-27B and olmo-32B 1 GPU (52 / 61 GB)
#     -c 8           CPUs
#     --dependency   a chain: the next job of a chain starts when the one before it ends
#     --export       only in `all`: SLURM_TMPDIR=/dev/shm, see THE DISK below
# A chain holds its GPUs from start to end, so nothing has to wait in the queue. That matters: a `bml`
# job that waits for GPUs does not wait - SLURM removes `public` jobs for it at once.
#
# THE DISK is the real limit, not the GPUs. Every job downloads its model into its own cache
# (the wrapper: ${SLURM_TMPDIR:-/tmp}/hf_<job>, deleted when the job ends): qwen-72B 139 GB, olmo-32B
# 65 GB, gemma-27B 55 GB. /tmp on Heraclitus is the ROOT disk, 444 GB (416 GB free on 2026-11-13).
# Three qwen-72B jobs at once would fill it. So:
#     default mode   caches in /tmp, and the plan must stay under 340 GB of caches
#     `all`          caches in /dev/shm (RAM: the server has 1.5 TB, /dev/shm holds 756 GB), under 600 GB
#
# Two plans:
#   side by side (fewer than 8 GPUs): Q chains of qwen-72B (2 GPUs each) + S chains of the two smaller
#       models (1 GPU each), 2Q + S <= F, picked so the slowest chain is as short as possible.
#       F = 3 (2026-11-13):  Q 1, S 1  -> about 10.5 h
#   stacked (8 GPUs): 4 qwen chains first (train job, then a compounds job), and when a qwen chain
#       ends, two chains of a smaller model start on its 2 GPUs (train job, then a compounds job).
#       About 4 h.
# Every job sent is written to data/<exp>/launch_log.tsv (time, job id, chain, the full sbatch line).
set -euo pipefail

MODE=${1:-free}
EXP=${EXP:-synthetic_cars}                 # which -n <exp_name> lines to take
LINES_FILE=${LINES_FILE:-scripts/slurms.sh}
NODE=heraclitus
TOTAL=8
CPUS=8
# Minutes of ONE chain doing all the jobs of its kind, from the test of 2026-11-13 (jobs 1498364-70,
# these same flags, two jobs running at once): 248,292 prompts per model, 8 jobs per model.
MIN_Q=630      # qwen-72B: 7.0 prompts/s, 5 min to load the model in each job
MIN_S=625      # gemma-27B 14.8 prompts/s + olmo-32B 13.3 prompts/s, 2-3 min to load in each job
GB_Q=140       # cache of one qwen-72B job
GB_S=65        # cache of one smaller job (olmo-32B, the bigger of the two)

mapfile -t LINES < <(grep -- "-n ${EXP}\$" "$LINES_FILE")
QW=(); SG=()
for l in "${LINES[@]}"; do
  if [[ $l == *'-m "qwen" -s "72"'* ]]; then QW+=("$l"); else SG+=("$l"); fi
done

USED=$(scontrol show node $NODE | grep -o 'AllocTRES=.*' | grep -o 'gres/gpu=[0-9]*' | cut -d= -f2 || true)
USED=${USED:-0}
EXTRA=""; LIMIT_GB=340
if [ "$MODE" = all ] || [ "${2:-}" = all ]; then
  F=$TOTAL; EXTRA="--export=ALL,SLURM_TMPDIR=/dev/shm"; LIMIT_GB=600
else
  F=$((TOTAL - USED))
fi
echo "$NODE: $USED of $TOTAL GPUs in use. mode=$MODE -> $F GPUs for this batch."
echo "${#LINES[@]} jobs for data/$EXP/: ${#QW[@]} qwen-72B (2 GPUs each), ${#SG[@]} smaller (1 GPU each)."
if [ "${#LINES[@]}" -eq 0 ]; then echo "ABORT: no '-n $EXP' lines in $LINES_FILE."; exit 1; fi
if [ "$F" -lt 3 ]; then echo "ABORT: fewer than 3 GPUs (one qwen-72B chain needs 2, the others 1)."; exit 1; fi

LAST=""
submit_chain() {      # $1 chain name, $2 GPUs per job, $3 job to wait for ("" = start now), then the lines
  local name=$1 g=$2 prev=$3 l extra cmd out
  shift 3
  for l in "$@"; do
    extra="--gres=gpu:$g -c $CPUS -J $name"
    [ -n "$EXTRA" ] && extra="$extra $EXTRA"
    [ -n "$prev" ] && extra="$extra --dependency=afterany:$prev"
    cmd=${l/sbatch -p bml -A bml/sbatch -p bml -A bml $extra}
    if [ "$MODE" = dry ]; then
      echo "  [$name] ${cmd:0:230}"
      prev="END-OF-$name"
      continue
    fi
    # a job that starts now, gentle mode: ask SLURM first, and stop if it would remove somebody's job
    if [ -z "$prev" ] && [ "$MODE" != all ]; then
      if eval "${cmd/sbatch /sbatch --test-only }" 2>&1 | grep -q "Preempts"; then
        echo "ABORT at chain $name: this job would remove another user's job. Nothing more is sent."
        exit 1
      fi
    fi
    out=$(eval "$cmd")                       # "Submitted batch job <id>"
    prev=${out##* }
    printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$prev" "$name" "$cmd" >> "data/$EXP/launch_log.tsv"
    echo "  [$name] job $prev"
  done
  LAST=$prev
}

if [ "$F" -ge 8 ] && [ "${#QW[@]}" -eq 8 ] && [ "${#SG[@]}" -eq 16 ]; then
  # ---- stacked: train jobs carry no -q, compounds jobs do
  QT=(); QC=(); ST=(); SC=()
  for l in "${QW[@]}"; do if [[ $l == *' -q "'* ]]; then QC+=("$l"); else QT+=("$l"); fi; done
  for l in "${SG[@]}"; do if [[ $l == *' -q "'* ]]; then SC+=("$l"); else ST+=("$l"); fi; done
  PEAK=$(( 4 * GB_Q > 8 * GB_S ? 4 * GB_Q : 8 * GB_S ))
  echo "plan: stacked. 4 qwen chains x 2 GPUs, then 8 single-GPU chains on the same GPUs. About 4 h."
  echo "caches: up to $PEAK GB at once, limit $LIMIT_GB GB${EXTRA:+ (in /dev/shm)}"
  if [ "$PEAK" -gt "$LIMIT_GB" ]; then echo "ABORT: the caches do not fit."; exit 1; fi
  QLAST=()
  for k in 0 1 2 3; do
    submit_chain "cars-q$((k + 1))" 2 "" "${QT[$k]}" "${QC[$k]}"
    QLAST+=("$LAST")
  done
  for j in 0 1 2 3 4 5 6 7; do
    submit_chain "cars-s$((j + 1))" 1 "${QLAST[$((j / 2))]}" "${ST[$j]}" "${SC[$j]}"
  done
else
  # ---- side by side: pick Q (qwen chains) and S (single-GPU chains)
  BEST=999999; Q=0; S=0
  for q in $(seq 1 $(( (F - 1) / 2 ))); do
    s=$((F - 2 * q))
    [ "$s" -gt "${#SG[@]}" ] && s=${#SG[@]}
    [ "$q" -gt "${#QW[@]}" ] && continue
    [ $((q * GB_Q + s * GB_S)) -gt "$LIMIT_GB" ] && continue        # the caches must fit
    tq=$((MIN_Q / q)); ts=$((MIN_S / s)); t=$(( tq > ts ? tq : ts ))
    if [ "$t" -lt "$BEST" ]; then BEST=$t; Q=$q; S=$s; fi
  done
  if [ "$Q" -eq 0 ]; then echo "ABORT: no plan fits $LIMIT_GB GB of caches."; exit 1; fi
  echo "plan: $Q qwen chain(s) x 2 GPUs + $S single-GPU chain(s) = $((2 * Q + S)) GPUs, about $((BEST / 60)) h $((BEST % 60)) min"
  echo "caches: up to $((Q * GB_Q + S * GB_S)) GB at once, limit $LIMIT_GB GB${EXTRA:+ (in /dev/shm)}"
  for k in $(seq 0 $((Q - 1))); do
    chain=()
    for i in "${!QW[@]}"; do [ $((i % Q)) -eq "$k" ] && chain+=("${QW[$i]}"); done
    submit_chain "cars-q$((k + 1))" 2 "" "${chain[@]}"
  done
  for k in $(seq 0 $((S - 1))); do
    chain=()
    for i in "${!SG[@]}"; do [ $((i % S)) -eq "$k" ] && chain+=("${SG[$i]}"); done
    [ "${#chain[@]}" -gt 0 ] && submit_chain "cars-s$((k + 1))" 1 "" "${chain[@]}"
  done
fi
[ "$MODE" = dry ] || squeue --me -o "%.9i %.10j %.9T %.10M %.12b %E %R"
