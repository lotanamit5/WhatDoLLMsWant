#!/bin/bash

#SBATCH --gres=gpu:16
#SBATCH -o ./out/%j.txt
#SBATCH -e ./err/%j.txt
source /home/lotan.amit/miniconda3/etc/profile.d/conda.sh
conda activate /home/lotan.amit/miniconda3/envs/whatdo-llms-want
python3 run_experiment.py --model qwen --colors en --templates en --exp_name full_experiment_en

# sbatch -p bml -A bml -w plotinus1 ./run_full_experiment.sh
