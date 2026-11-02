#!/usr/bin/env bash
# GENERATED FILE - do not edit by hand.
# Edit scripts/create_slurms.py and re-run it, then commit BOTH files together.
# (No timestamp on purpose: identical params must produce an identical file,
#  so `git diff` shows a changed batch and nothing else.)
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "gemma" -s "27" -a "laptops_robustness" -c "brand=Apple" -n laptops_robustness
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "gemma" -s "27" -a "laptops_robustness" -c "brand=Dell" -n laptops_robustness
sbatch -p bml -A bml -w plato1 scripts/run_data_collection.sh -m "olmo" -s "32" -a "laptops_robustness" -c "brand=Apple" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "olmo" -s "32" -a "laptops_robustness" -c "brand=Dell" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "gemma" -s "27" -a "laptops_robustness" -c "brand=Dell,screen=14-inch,ram=8GB" -n laptops_robustness
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "gemma" -s "27" -a "laptops_robustness" -c "brand=Apple,screen=14-inch,ram=8GB" -n laptops_robustness
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "gemma" -s "27" -a "laptops_robustness" -c "brand=Dell,screen=14-inch,ram=4GB" -n laptops_robustness
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "olmo" -s "1" -a "laptops_robustness" -c "screen=13-inch" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "olmo" -s "1" -a "laptops_robustness" -c "screen=16-inch" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "olmo" -s "1" -a "laptops_robustness" -c "ram=4GB" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "olmo" -s "1" -a "laptops_robustness" -c "ram=16GB" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "olmo" -s "7" -a "laptops_robustness" -c "screen=13-inch" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "olmo" -s "7" -a "laptops_robustness" -c "screen=16-inch" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "olmo" -s "7" -a "laptops_robustness" -c "ram=4GB" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "olmo" -s "7" -a "laptops_robustness" -c "ram=16GB" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "olmo" -s "13" -a "laptops_robustness" -c "screen=13-inch" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "olmo" -s "13" -a "laptops_robustness" -c "screen=16-inch" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "olmo" -s "13" -a "laptops_robustness" -c "ram=4GB" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "olmo" -s "13" -a "laptops_robustness" -c "ram=16GB" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plato1 scripts/run_data_collection.sh -m "olmo" -s "32" -a "laptops_robustness" -c "screen=13-inch" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "olmo" -s "32" -a "laptops_robustness" -c "screen=16-inch" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plato1 scripts/run_data_collection.sh -m "olmo" -s "32" -a "laptops_robustness" -c "ram=4GB" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "olmo" -s "32" -a "laptops_robustness" -c "ram=16GB" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plato1 scripts/run_data_collection.sh -m "qwen" -s "32" -a "laptops_robustness" -c "" -t "0" -n laptops_all_templates
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "qwen" -s "32" -a "laptops_robustness" -c "ram=4GB" -t "0" -n laptops_all_templates
