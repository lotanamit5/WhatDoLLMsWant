#!/bin/bash

#SBATCH -c 32
#SBATCH --gres=gpu:8
#SBATCH -o ./out/%j.txt
#SBATCH -e ./err/%j.txt

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
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

source /home/lotan.amit/miniconda3/etc/profile.d/conda.sh
conda activate /home/lotan.amit/miniconda3/envs/whatdo-llms-want
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Build command array
CMD=(python3 scripts/data_collection.py)

if [ -n "$MODEL" ]; then
    CMD+=(--model "$MODEL")
fi

if [ -n "$ALTERNATIVES" ]; then
    CMD+=(--alternatives "$ALTERNATIVES")
fi

CMD+=(--cluster_job "$SLURM_JOB_ID")

# Execute command
"${CMD[@]}"

# sbatch -p bml -A bml -w plato1 scripts/run_data_collection.sh -a {alternatives} -m {model}