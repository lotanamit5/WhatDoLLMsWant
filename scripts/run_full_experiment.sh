#!/bin/bash

#SBATCH --gres=gpu:8
#SBATCH -o ./out/%j.txt
#SBATCH -e ./err/%j.txt

TASK=false

while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model)
      MODEL="$2"
      shift 2
      ;;
    -a|--alternatives)
      ALTERNATIVES="$2"
      shift 2
      ;;
    -t|--templates)
      TEMPLATES="$2"
      shift 2
      ;;
    -n|--exp_name)
      EXP_NAME="$2"
      shift 2
      ;;
    --task)
      TASK=true
      shift
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

source /home/lotan.amit/miniconda3/etc/profile.d/conda.sh
conda activate /home/lotan.amit/miniconda3/envs/whatdo-llms-want

# Build command array
CMD=(python3 scripts/run_experiment.py)

if [ -n "$MODEL" ]; then
    CMD+=(--model "$MODEL")
fi

if [ -n "$ALTERNATIVES" ]; then
    CMD+=(--alternatives "$ALTERNATIVES")
fi

if [ -n "$TEMPLATES" ]; then
    CMD+=(--templates "$TEMPLATES")
fi

if [ -n "$EXP_NAME" ]; then
    CMD+=(--exp_name "$EXP_NAME")
fi

if [ "$TASK" = true ]; then
    CMD+=(--task)
fi

CMD+=(--cluster_job "$SLURM_JOB_ID")

# Execute command
"${CMD[@]}"

# sbatch -p bml -A bml -w plato1 scripts/run_full_experiment.sh -m qwen0_5 -a colors -t colors -n exp_name