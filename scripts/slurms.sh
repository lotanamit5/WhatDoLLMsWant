sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection_robust.sh -m qwen -s 7 -a laptops_robustness -n laptops_robustness
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection_robust.sh -m qwen -s 32 -a laptops_robustness -n laptops_robustness
