#!/bin/bash

#SBATCH -c 32
#SBATCH --gres=gpu:8
#SBATCH --output=out/%j.txt
#SBATCH --error=err/%j.txt

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

RUN_ID="${SLURM_JOB_ID}"
EXP_DIR="data/${EXP_NAME}/${RUN_ID}"
OUTPUT_PATH="${EXP_DIR}/output.log"

mkdir -p "$EXP_DIR"
mkdir -p "$(dirname "$OUTPUT_PATH")"

exec >"$OUTPUT_PATH" 2>&1

source /home/lotan.amit/miniconda3/etc/profile.d/conda.sh
conda activate /home/lotan.amit/miniconda3/envs/whatdo-llms-want

# Use per-job Hugging Face cache to avoid parallel download collisions on SLURM.
if [ -n "${SLURM_JOB_ID}" ]; then
  CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/hf_${SLURM_JOB_ID}"
  mkdir -p "$CACHE_ROOT"
  export HF_HOME="$CACHE_ROOT"
  export HF_HUB_CACHE="$CACHE_ROOT/hub"
  export TRANSFORMERS_CACHE="$CACHE_ROOT/transformers"
fi

# Build command array
CMD=(python3 scripts/data_collection.py)

if [ -n "$MODEL" ]; then
    CMD+=(--model_family "$MODEL")
fi

if [ -n "$SIZE" ]; then
    CMD+=(--model_size "$SIZE")
fi

if [ -n "$ALTERNATIVES" ]; then
    CMD+=(--alternatives "$ALTERNATIVES")
fi

CMD+=(--exp_dir "$EXP_DIR")

# Execute command
"${CMD[@]}"
