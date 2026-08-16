sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "qwen-pt" -s "7" -a "laptops_robustness" -c "" -p "pretrained" -n laptops_robustness_pt
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "qwen-pt" -s "7" -a "laptops_robustness" -c "screen=14-inch" -p "pretrained" -n laptops_robustness_pt
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "qwen-pt" -s "7" -a "laptops_robustness" -c "ram=8GB" -p "pretrained" -n laptops_robustness_pt
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "qwen-pt" -s "7" -a "laptops_robustness" -c "screen=14-inch,ram=8GB" -p "pretrained" -n laptops_robustness_pt
