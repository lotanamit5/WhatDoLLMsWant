#!/bin/bash

#SBATCH -c 32
#SBATCH --gres=gpu:8
#SBATCH --output=out/%j.txt
#SBATCH --error=err/%j.txt
echo "Starting"

# SLURM wrapper for scripts/real_laptops_collect.py. Same environment set-up as
# run_data_collection.sh (conda env, per-job HF cache, run from node-local scratch).
# One job = one model, all its contracts: runs land in data/<exp_name>/<job_id>_<constraints_id>/,
# and data/<exp_name>/<job_id>/ holds only this log.
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--models)   MODEL="$2";    shift 2 ;;
    -s|--size)     SIZE="$2";     shift 2 ;;
    -o|--mode)     MODE="$2";     shift 2 ;;   # main | sanity
    -x|--only)     ONLY="$2";     shift 2 ;;   # optional: ";"-separated contract specs, for reruns
    -n|--exp_name) EXP_NAME="$2"; shift 2 ;;
    *) echo "Unknown option $1"; exit 1 ;;
  esac
done

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXP_ROOT="${REPO_ROOT}/data/${EXP_NAME}"
LOG_DIR="${EXP_ROOT}/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"
exec >"${LOG_DIR}/output.log" 2>&1

source /home/lotan.amit/miniconda3/etc/profile.d/conda.sh
conda activate /home/lotan.amit/miniconda3/envs/whatdo-llms-want

CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/hf_${SLURM_JOB_ID:-$$}"
mkdir -p "$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT"
export HF_HUB_CACHE="$CACHE_ROOT/hub"
export TRANSFORMERS_CACHE="$CACHE_ROOT/transformers"

# Run from node-local scratch: src/agent.py puts its model cache in "$(pwd)/huggingface/.cache".
WORKDIR="${CACHE_ROOT}/work"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
echo "repo:    $REPO_ROOT"
echo "workdir: $(pwd)"

AVAIL_MB=$(df -Pm "$REPO_ROOT" | awk 'NR==2 {print $4}')
echo "free on repo filesystem: ${AVAIL_MB} MB"
if [ "${AVAIL_MB:-0}" -lt 2048 ]; then
  echo "ABORT: under 2 GB free on the repo filesystem - refusing to start."
  exit 1
fi

CMD=(python3 "${REPO_ROOT}/scripts/real_laptops_collect.py"
     --model_family "$MODEL" --model_size "$SIZE" --mode "$MODE"
     --exp_root "$EXP_ROOT" --job_id "$SLURM_JOB_ID")
if [ -n "$ONLY" ]; then
    CMD+=(--only "$ONLY")
fi

echo "Running"
"${CMD[@]}"
STATUS=$?

for D in "${EXP_ROOT}/${SLURM_JOB_ID}"_*; do
  if [ -f "${D}/scores.csv" ]; then
    echo "scores.csv rows: $(( $(wc -l < "${D}/scores.csv") - 1 ))  (${D##*/})"
  elif [ -f "${D}/config.json" ]; then
    echo "WARNING: ${D##*/} has config.json but no scores.csv - incomplete."
  fi
done
echo "Finished with status $STATUS"
exit $STATUS
