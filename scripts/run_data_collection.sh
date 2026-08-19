#!/bin/bash

#SBATCH -c 32
#SBATCH --gres=gpu:8
#SBATCH --output=out/%j.txt
#SBATCH --error=err/%j.txt
echo "Starting"

# Merged 2026-08-10 with run_data_collection_robust.sh, which was the same file plus -c.
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--models)
      MODEL="$2"
      shift 2
      ;;
    -s|--size)
      SIZE="$2"
      shift 2
      ;;
    -a|--alternatives)
      ALTERNATIVES="$2"
      shift 2
      ;;
    -c|--constraints)
      # "screen=14-inch,ram=8GB"; empty string means no constraint
      CONSTRAINTS="$2"
      shift 2
      ;;
    -f|--frame)
      FRAME="$2"
      shift 2
      ;;
    -t|--n_templates)
      N_TEMPLATES="$2"
      shift 2
      ;;
    -p|--template_set)
      # "options" (instruct) or "pretrained" (base models)
      TEMPLATE_SET="$2"
      shift 2
      ;;
    -n|--exp_name)
      EXP_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Absolute, so the collector can be run from anywhere (see the cd below).
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ID="${SLURM_JOB_ID}"
EXP_DIR="${REPO_ROOT}/data/${EXP_NAME}/${RUN_ID}"
OUTPUT_PATH="${EXP_DIR}/output.log"

mkdir -p "$EXP_DIR"

exec >"$OUTPUT_PATH" 2>&1

source /home/lotan.amit/miniconda3/etc/profile.d/conda.sh
conda activate /home/lotan.amit/miniconda3/envs/whatdo-llms-want

# Use per-job Hugging Face cache to avoid parallel download collisions on SLURM.
CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/hf_${SLURM_JOB_ID:-$$}"
mkdir -p "$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT"
export HF_HUB_CACHE="$CACHE_ROOT/hub"
export TRANSFORMERS_CACHE="$CACHE_ROOT/transformers"

# Run from node-local scratch, NOT from the repo.
#
# `_load_model_and_tokenizer` in src/agent.py hardcodes its model cache to
# "$(pwd)/huggingface/.cache", which overrides every HF_* variable set above. On
# 2026-08-18 that put ~212 GB of OLMo weights on the shared repo filesystem, filled the
# disk, and killed 7 of 8 jobs. Changing the working directory redirects that hardcoded
# path onto scratch, where it is cleaned up with the job.
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

# Build command array
CMD=(python3 "${REPO_ROOT}/scripts/data_collection.py")

if [ -n "$MODEL" ]; then
    CMD+=(--model_family "$MODEL")
fi

if [ -n "$SIZE" ]; then
    CMD+=(--model_size "$SIZE")
fi

if [ -n "$ALTERNATIVES" ]; then
    CMD+=(--alternatives "$ALTERNATIVES")
fi

# Always pass --constraints, even when empty: "" is a meaningful value (no constraint)
# and the script's default is already "".
CMD+=(--constraints "$CONSTRAINTS")

if [ -n "$FRAME" ]; then
    CMD+=(--frame "$FRAME")
fi

if [ -n "$N_TEMPLATES" ]; then
    CMD+=(--n_templates "$N_TEMPLATES")
fi

if [ -n "$TEMPLATE_SET" ]; then
    CMD+=(--template_set "$TEMPLATE_SET")
fi

CMD+=(--exp_dir "$EXP_DIR")

echo "Running"
# Execute command
"${CMD[@]}"
STATUS=$?

echo "model cache used: $(du -sh "$CACHE_ROOT" 2>/dev/null | cut -f1)"
if [ -f "${EXP_DIR}/scores.csv" ]; then
  echo "scores.csv rows: $(( $(wc -l < "${EXP_DIR}/scores.csv") - 1 ))"
else
  echo "WARNING: no scores.csv written - this run is incomplete."
fi

# Propagate the exit code so a failed run shows as FAILED in sacct rather than COMPLETED.
echo "Finished with status $STATUS"
exit $STATUS
