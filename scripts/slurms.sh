#!/usr/bin/env bash
# GENERATED FILE - do not edit by hand.
# Edit scripts/create_slurms.py and re-run it, then commit BOTH files together.
# (No timestamp on purpose: identical params must produce an identical file,
#  so `git diff` shows a changed batch and nothing else.)
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "olmo" -s "1" -a "laptops_robustness" -c "screen=14-inch" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "olmo" -s "1" -a "laptops_robustness" -c "ram=8GB" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "olmo" -s "1" -a "laptops_robustness" -c "screen=14-inch,ram=8GB" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "olmo" -s "7" -a "laptops_robustness" -c "screen=14-inch" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "olmo" -s "7" -a "laptops_robustness" -c "ram=8GB" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "olmo" -s "7" -a "laptops_robustness" -c "screen=14-inch,ram=8GB" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "olmo" -s "13" -a "laptops_robustness" -c "screen=14-inch" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "olmo" -s "13" -a "laptops_robustness" -c "ram=8GB" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "olmo" -s "13" -a "laptops_robustness" -c "screen=14-inch,ram=8GB" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "olmo" -s "32" -a "laptops_robustness" -c "screen=14-inch" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "olmo" -s "32" -a "laptops_robustness" -c "ram=8GB" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "olmo" -s "32" -a "laptops_robustness" -c "screen=14-inch,ram=8GB" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "olmo-pt" -s "1" -a "laptops_robustness" -c "screen=14-inch" -p "pretrained" -n laptops_olmo_pt
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "olmo-pt" -s "1" -a "laptops_robustness" -c "ram=8GB" -p "pretrained" -n laptops_olmo_pt
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "olmo-pt" -s "1" -a "laptops_robustness" -c "screen=14-inch,ram=8GB" -p "pretrained" -n laptops_olmo_pt
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "olmo-pt" -s "7" -a "laptops_robustness" -c "screen=14-inch" -p "pretrained" -n laptops_olmo_pt
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "olmo-pt" -s "7" -a "laptops_robustness" -c "ram=8GB" -p "pretrained" -n laptops_olmo_pt
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "olmo-pt" -s "7" -a "laptops_robustness" -c "screen=14-inch,ram=8GB" -p "pretrained" -n laptops_olmo_pt
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "olmo-pt" -s "13" -a "laptops_robustness" -c "screen=14-inch" -p "pretrained" -n laptops_olmo_pt
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "olmo-pt" -s "13" -a "laptops_robustness" -c "ram=8GB" -p "pretrained" -n laptops_olmo_pt
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "olmo-pt" -s "13" -a "laptops_robustness" -c "screen=14-inch,ram=8GB" -p "pretrained" -n laptops_olmo_pt
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "olmo-pt" -s "32" -a "laptops_robustness" -c "screen=14-inch" -p "pretrained" -n laptops_olmo_pt
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "olmo-pt" -s "32" -a "laptops_robustness" -c "ram=8GB" -p "pretrained" -n laptops_olmo_pt
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "olmo-pt" -s "32" -a "laptops_robustness" -c "screen=14-inch,ram=8GB" -p "pretrained" -n laptops_olmo_pt
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "qwen-pt" -s "72" -a "laptops_robustness" -c "screen=14-inch" -p "pretrained" -n laptops_robustness_pt
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "qwen" -s "72" -a "laptops_robustness" -c "screen=13-inch" -n laptops_robustness
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "qwen" -s "72" -a "laptops_robustness" -c "ram=16GB" -n laptops_robustness
