#!/bin/bash

#SBATCH --gres=gpu:16
#SBATCH -o ./out/%j.txt
#SBATCH -e ./err/%j.txt
source /home/lotan.amit/miniconda3/etc/profile.d/conda.sh
conda activate /home/lotan.amit/miniconda3/envs/whatdo-llms-want

python3 test_size.py

# sbatch -p bml -A bml -w plato1 ./test_size.sh