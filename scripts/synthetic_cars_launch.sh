#!/usr/bin/env bash
# Synthetic cars: send the cars jobs of scripts/slurms.sh to Heraclitus, right-sized and in chains.
#
# Run on the newton login node, from the repo root:
#     bash scripts/synthetic_cars_launch.sh dry    # print the plan, send nothing (`dry all`: as if all 8 GPUs were free)
#     bash scripts/synthetic_cars_launch.sh        # use only the GPUs that are free now - nobody's job is removed
#     bash scripts/synthetic_cars_launch.sh all    # count all 8 GPUs as ours - SLURM removes `public` jobs if it must
#
# Why not `bash scripts/slurms.sh`: the wrapper asks for 8 GPUs and 32 CPUs per job, which is the whole
# of Heraclitus, so the jobs would run one at a time (about 21 h) and every one of them would remove
# other people's jobs. A job is not faster on more GPUs (gemma-27B: 15 prompts/s on 1 GPU, 13.9 on 3).
# This script takes the SAME lines and adds to each only sbatch flags:
#     --gres=gpu:N   qwen-72B 2 GPUs (136 GB of weights), gemma-27B and olmo-32B 1 GPU (52 / 61 GB)
#     -c 8           CPUs
#     --dependency   a chain: the next job of a chain starts when the one before it ends
# A chain holds its GPUs from start to end, so nothing has to wait in the queue. That matters: a `bml`
# job that waits for GPUs does not wait - SLURM removes `public` jobs for it at once.
#
# THE DISK is the real limit, not the GPUs. Every job downloads its model into its own cache
# (the wrapper: /tmp/hf_<job>, deleted when the job ends): qwen-72B 139 GB, olmo-32B 65 GB,
# gemma-27B 55 GB. /tmp on Heraclitus is the ROOT disk, 444 GB (416 GB free on 2026-11-13). Three
# qwen-72B jobs at once would fill it. Every plan here keeps the caches under 340 GB, so at most
# 2 qwen-72B + 1 smaller job run together = 5 GPUs. More GPUs than that cannot be used.
#
# DO NOT move the caches to /dev/shm (RAM). Tried on 2026-11-13 (SLURM_TMPDIR=/dev/shm, jobs
# 1498452-1498475): one job alone works, but whenever one of our jobs ENDS on the server, all our
# files in /dev/shm are deleted, also those of the jobs still loading. 24 of 24 jobs died at model
# load. /data (3.5 TB, local) would be the right place, but only root can write there.
#
# Two plans:
#   two phases (5 GPUs or more): 2 qwen chains + 1 gemma chain from the start; when a qwen chain
#       ends, two olmo chains start on its 2 GPUs. About 6 h 40 min. Caches: 335 GB, then 260 GB.
#   side by side (3 or 4 GPUs): Q chains of qwen-72B (2 GPUs each) + S chains of the two smaller
#       models (1 GPU each), 2Q + S <= F, picked so the slowest chain is as short as possible.
#       F = 3:  Q 1, S 1  -> about 10.5 h
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
GB_G=55        # cache of one gemma-27B job
LIMIT_GB=340   # caches allowed in /tmp at once

mapfile -t LINES < <(grep -- "-n ${EXP}\$" "$LINES_FILE")
QW=(); SG=()
for l in "${LINES[@]}"; do
  if [[ $l == *'-m "qwen" -s "72"'* ]]; then QW+=("$l"); else SG+=("$l"); fi
done

USED=$(scontrol show node $NODE | grep -o 'AllocTRES=.*' | grep -o 'gres/gpu=[0-9]*' | cut -d= -f2 || true)
USED=${USED:-0}
if [ "$MODE" = all ] || [ "${2:-}" = all ]; then F=$TOTAL; else F=$((TOTAL - USED)); fi
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

if [ "$F" -ge 5 ] && [ "${#QW[@]}" -eq 8 ] && [ "${#SG[@]}" -eq 16 ]; then
  # ---- two phases. Train jobs carry no -q, compounds jobs do.
  QT=(); QC=(); GM=(); OT=(); OC=()
  for l in "${QW[@]}"; do if [[ $l == *' -q "'* ]]; then QC+=("$l"); else QT+=("$l"); fi; done
  for l in "${SG[@]}"; do
    if [[ $l == *'-m "gemma"'* ]]; then GM+=("$l")
    elif [[ $l == *' -q "'* ]]; then OC+=("$l")
    else OT+=("$l"); fi
  done
  PEAK=$((2 * GB_Q + GB_G))          # phase 1; phase 2 is 4 olmo = 260 GB, the hand-over at most 325 GB
  echo "plan: two phases on 5 GPUs. 2 qwen chains + 1 gemma chain; then 4 olmo chains. About 6 h 40 min."
  echo "caches: up to $PEAK GB at once, limit $LIMIT_GB GB"
  if [ "$PEAK" -gt "$LIMIT_GB" ]; then echo "ABORT: the caches do not fit."; exit 1; fi
  submit_chain "cars-q1" 2 "" "${QT[0]}" "${QT[2]}" "${QC[0]}" "${QC[2]}"; Q1=$LAST
  submit_chain "cars-q2" 2 "" "${QT[1]}" "${QT[3]}" "${QC[1]}" "${QC[3]}"; Q2=$LAST
  submit_chain "cars-g1" 1 "" "${GM[@]}"
  submit_chain "cars-o1" 1 "$Q1" "${OT[0]}" "${OC[0]}"
  submit_chain "cars-o2" 1 "$Q1" "${OT[1]}" "${OC[1]}"
  submit_chain "cars-o3" 1 "$Q2" "${OT[2]}" "${OC[2]}"
  submit_chain "cars-o4" 1 "$Q2" "${OT[3]}" "${OC[3]}"
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
  echo "caches: up to $((Q * GB_Q + S * GB_S)) GB at once, limit $LIMIT_GB GB"
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
