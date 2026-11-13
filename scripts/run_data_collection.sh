#!/bin/bash

#SBATCH -c 32
#SBATCH --gres=gpu:8
#SBATCH --output=out/%j.txt
#SBATCH --error=err/%j.txt
echo "Starting"

# The ONE SLURM wrapper (2026-11-13: run_real_laptops.sh and real_laptops_collect.py are gone,
# data_collection.py runs the real laptops too). It sets up the environment and the logs, then runs
#
#     python scripts/<collector> <flags> --exp_dir data/<exp_name>/<job_id>
#
# Two flags belong to the wrapper:
#     -n <exp_name>    the folder under data/
#     -y <collector>   which file in scripts/ to run; default data_collection.py
# Every other flag is handed to the collector under its long name. A new collector needs no new
# wrapper: it must accept --exp_dir, and any new flag gets one line in the table below.
#
# LOGS are live, in the run's own folder:
#     data/<exp_name>/<job_id>/output.log   progress: one line a minute (rows done, speed, time left)
#     data/<exp_name>/<job_id>/error.log    warnings, download bars, the traceback if it dies
# Watch one job:         tail -F data/<exp_name>/<job_id>/*.log
# Watch a whole batch:   tail -F data/<exp_name>/*/output.log
# (out/<job_id>.txt and err/<job_id>.txt are SLURM's own files. They only hold what happens before
#  the lines below take over - look there if a job leaves no folder at all.)
PY="data_collection.py"
ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -n|--exp_name)     EXP_NAME="$2" ;;
    -y|--py)           PY="$2" ;;
    -m|--models)       ARGS+=(--model_family "$2") ;;
    -s|--size)         ARGS+=(--model_size "$2") ;;
    -a|--alternatives) ARGS+=(--alternatives "$2") ;;   # several, comma-separated: one run each
    -c|--constraints)  ARGS+=(--constraints "$2") ;;    # "screen=14-inch,ram=8GB"; "" = no constraint;
                                                        # several, ";"-separated: one run each
    -f|--frame)        ARGS+=(--frame "$2") ;;
    -t|--n_templates)  ARGS+=(--n_templates "$2") ;;
    -p|--template_set) ARGS+=(--template_set "$2") ;;   # "options" (instruct) or "pretrained" (base)
    -q|--pairs)        ARGS+=(--pairs "$2") ;;          # CSV of item-id pairs; default = all pairs
    *) echo "Unknown option $1"; exit 1 ;;
  esac
  shift 2
done
if [ -z "$EXP_NAME" ]; then echo "ABORT: no -n <exp_name>."; exit 1; fi

# Absolute, so the collector can be run from anywhere (see the cd below).
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXP_DIR="${REPO_ROOT}/data/${EXP_NAME}/${SLURM_JOB_ID}"
mkdir -p "$EXP_DIR"

exec >"${EXP_DIR}/output.log" 2>"${EXP_DIR}/error.log"
# Python holds back its prints in an 8 KB buffer when they go to a file, so a quiet job showed
# nothing until it ended. This turns the buffer off.
export PYTHONUNBUFFERED=1

START=$(date +%s)
echo "started: $(date '+%F %T') on $(hostname), job ${SLURM_JOB_ID}"

source /home/lotan.amit/miniconda3/etc/profile.d/conda.sh
conda activate /home/lotan.amit/miniconda3/envs/whatdo-llms-want

# Use per-job Hugging Face cache to avoid parallel download collisions on SLURM.
CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/hf_${SLURM_JOB_ID:-$$}"
mkdir -p "$CACHE_ROOT"
# Delete this job's cache when the job ends, however it ends. Nothing else does: SLURM_TMPDIR is
# not set on these servers, so the cache is /tmp/hf_<job_id> and it used to stay there forever.
# Found 2026-11-13: 1.46 TB of old caches on plato1 (217 GB left free), 461 GB on plato2.
# A cache is named after its job id, so no later job can reuse it - keeping it buys nothing.
trap 'rm -rf "$CACHE_ROOT"' EXIT
trap 'exit 143' TERM INT
export HF_HOME="$CACHE_ROOT"
export HF_HUB_CACHE="$CACHE_ROOT/hub"
export TRANSFORMERS_CACHE="$CACHE_ROOT/transformers"

# Run from node-local scratch, NOT from the repo.
#
# `_load_model_and_tokenizer` in src/agent.py hardcodes its model cache to
# "$(pwd)/huggingface/.cache", which overrides every HF_* variable set above. On
# 2026-08-18 that put ~212 GB of OLMo weights on the shared repo filesystem, filled the
# disk, and killed 7 of 8 jobs. Changing the working directory redirects that hardcoded
# path onto scratch, where the trap above deletes it when the job ends.
#
# Safe because `--exp_dir` is absolute and is the ONLY path the collector resolves
# relative to anything: its imports go through __file__, and git_commit() cds itself.
WORKDIR="${CACHE_ROOT}/work"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
echo "repo:    $REPO_ROOT"
echo "workdir: $(pwd)"
echo "         (the hardcoded model cache lands in $(pwd)/huggingface/.cache)"

# Pre-flight: fail loudly instead of writing config.json and then dying mid-download.
AVAIL_MB=$(df -Pm "$REPO_ROOT" | awk 'NR==2 {print $4}')
echo "free on repo filesystem:    ${AVAIL_MB} MB"
echo "free on scratch (${CACHE_ROOT}): $(df -Pm "$CACHE_ROOT" | awk 'NR==2 {print $4}') MB"
if [ "${AVAIL_MB:-0}" -lt 2048 ]; then
  echo "ABORT: under 2 GB free on the repo filesystem - refusing to start."
  exit 1
fi

CMD=(python3 "${REPO_ROOT}/scripts/${PY}" "${ARGS[@]}" --exp_dir "$EXP_DIR")
echo "running: $(printf '%q ' "${CMD[@]}")"

# A sign of life while the model downloads and loads. That is usually 3 minutes but has taken 72
# (job 1497200), and the download bar only moves when a whole file lands. One line a minute with
# the size of the cache: growing = downloading, flat for long = hung. Stops by itself at the first
# config.json, which is when scoring starts and the collector's own progress lines take over.
(
  while sleep "${HEARTBEAT_SEC:-60}"; do
    if ls "${EXP_DIR}"/config.json "${EXP_DIR}"_*/config.json 2>/dev/null | grep -q .; then break; fi
    echo "[$(date +%T)] loading the model... cache is $(du -sh "$CACHE_ROOT" 2>/dev/null | cut -f1)"
  done
) &
HEARTBEAT=$!

"${CMD[@]}"
STATUS=$?
kill "$HEARTBEAT" 2>/dev/null

echo "model cache used: $(du -sh "$CACHE_ROOT" 2>/dev/null | cut -f1)"
# One run writes to EXP_DIR. A job with several runs (several domains and/or several contracts)
# writes EXP_DIR_<name>, one folder per run, and EXP_DIR itself keeps only the logs.
FOUND=0
for D in "${EXP_DIR}" "${EXP_DIR}"_*; do
  if [ -f "${D}/scores.csv" ]; then
    echo "scores.csv rows: $(( $(wc -l < "${D}/scores.csv") - 1 ))  (${D##*/})"
    FOUND=1
  elif [ -f "${D}/config.json" ]; then
    echo "WARNING: ${D##*/} has config.json but no scores.csv - incomplete."
  fi
done
if [ "$FOUND" -eq 0 ]; then
  echo "WARNING: no scores.csv written - this run is incomplete."
fi

# Propagate the exit code so a failed run shows as FAILED in sacct rather than COMPLETED.
echo "Finished with status $STATUS after $(( ($(date +%s) - START) / 60 )) min"
if [ "$STATUS" -ne 0 ]; then echo "FAILED - the reason is at the end of error.log"; fi
exit $STATUS
