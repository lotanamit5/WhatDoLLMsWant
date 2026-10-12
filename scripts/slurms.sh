#!/usr/bin/env bash
# GENERATED FILE - do not edit by hand.
# Edit scripts/create_slurms.py and re-run it, then commit BOTH files together.
# (No timestamp on purpose: identical params must produce an identical file,
#  so `git diff` shows a changed batch and nothing else.)
sbatch -p bml -A bml -w plato1 scripts/run_data_collection.sh -m "qwen" -s "32" -a "laptops_robustness" -c "brand=Apple" -n laptops_robustness
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "qwen" -s "32" -a "laptops_robustness" -c "brand=Apple,screen=14-inch" -n laptops_robustness
sbatch -p bml -A bml -w plato1 scripts/run_data_collection.sh -m "qwen" -s "32" -a "laptops_robustness" -c "brand=Apple,ram=8GB" -n laptops_robustness
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "qwen" -s "32" -a "laptops_robustness" -c "brand=Apple,screen=14-inch,ram=8GB" -n laptops_robustness
sbatch -p bml -A bml -w plato1 scripts/run_data_collection.sh -m "qwen" -s "32" -a "laptops_robustness" -c "brand=Dell,ram=4GB" -n laptops_robustness
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "qwen" -s "32" -a "laptops_robustness" -c "brand=Dell,screen=14-inch,ram=4GB" -n laptops_robustness
