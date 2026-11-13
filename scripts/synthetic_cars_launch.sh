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
# A chain holds its GPUs from start to end, so nothing has to wait in the queue. That matters: a `bml`
# job that waits for GPUs does not wait - SLURM removes `public` jobs for it at once.
#
# The plan for F usable GPUs: Q chains of qwen-72B (2 GPUs each) + S chains of the two smaller models
# (1 GPU each), 2Q + S <= F, picked so the slowest chain is as short as possible.
#     F = 3 (2026-11-13):  Q 1, S 1  -> about 10.5 h
#     F = 8:               Q 2, S 4  -> about 5.3 h
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

# pick Q (qwen chains) and S (single-GPU chains)
BEST=999999; Q=1; S=1
for q in $(seq 1 $(( (F - 1) / 2 ))); do
  s=$((F - 2 * q))
  [ "$s" -gt "${#SG[@]}" ] && s=${#SG[@]}
  [ "$q" -gt "${#QW[@]}" ] && continue
  tq=$((MIN_Q / q)); ts=$((MIN_S / s)); t=$(( tq > ts ? tq : ts ))
  if [ "$t" -lt "$BEST" ]; then BEST=$t; Q=$q; S=$s; fi
done
echo "plan: $Q qwen chain(s) x 2 GPUs + $S single-GPU chain(s) = $((2 * Q + S)) GPUs, about $((BEST / 60)) h $((BEST % 60)) min"

submit_chain() {      # $1 chain name, $2 GPUs per job, then the lines of the chain in order
  local name=$1 g=$2 prev="" l extra cmd out
  shift 2
  for l in "$@"; do
    extra="--gres=gpu:$g -c $CPUS -J $name"
    [ -n "$prev" ] && extra="$extra --dependency=afterany:$prev"
    cmd=${l/sbatch -p bml -A bml/sbatch -p bml -A bml $extra}
    if [ "$MODE" = dry ]; then
      echo "  [$name] ${cmd:0:210}"
      prev="PREV"
      continue
    fi
    # first job of a chain, gentle mode: ask SLURM first, and stop if it would remove somebody's job
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
}

for k in $(seq 0 $((Q - 1))); do
  chain=()
  for i in "${!QW[@]}"; do [ $((i % Q)) -eq "$k" ] && chain+=("${QW[$i]}"); done
  submit_chain "cars-q$((k + 1))" 2 "${chain[@]}"
done
for k in $(seq 0 $((S - 1))); do
  chain=()
  for i in "${!SG[@]}"; do [ $((i % S)) -eq "$k" ] && chain+=("${SG[$i]}"); done
  [ "${#chain[@]}" -gt 0 ] && submit_chain "cars-s$((k + 1))" 1 "${chain[@]}"
done
[ "$MODE" = dry ] || squeue --me -o "%.9i %.10j %.9T %.10M %.12b %E %R"
