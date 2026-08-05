sbatch -p bml -A bml -w plato2 scripts/run_data_collection_robust.sh -m qwen -s 0.5 -a laptops_robustness -c  -n laptops_robustness
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection_robust.sh -m qwen -s 72 -a laptops_robustness -c  -n laptops_robustness
