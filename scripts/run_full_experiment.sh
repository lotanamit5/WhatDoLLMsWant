#!/bin/bash

#SBATCH --gres=gpu:8
#SBATCH -o ./out/%j.txt
#SBATCH -e ./err/%j.txt

# Default values
MODEL="qwen7I"
COLORS="en"
TEMPLATES="en"
EXP_NAME="exp_name"

while getopts "m:a:t:n:" opt; do
  case $opt in
    m) MODEL="$OPTARG" ;;
    a) ALTERNATIVES="$OPTARG" ;;
    t) TEMPLATES="$OPTARG" ;;
    n) EXP_NAME="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done
source /home/lotan.amit/miniconda3/etc/profile.d/conda.sh
conda activate /home/lotan.amit/miniconda3/envs/whatdo-llms-want
python3 scripts/run_experiment.py --model "$MODEL" --alternatives "$ALTERNATIVES" --templates "$TEMPLATES" --exp_name "$EXP_NAME"

# sbatch -p bml -A bml -w plato1 scripts/run_full_experiment.sh -m qwen7 -a en -t en -n exp_name