sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection_robust.sh -m qwen -s 0.5 -a laptops_robustness -n laptops_num_vs_txt
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection_robust.sh -m qwen -s 32 -a laptops_robustness -n laptops_num_vs_txt
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection_robust.sh -m qwen -s 72 -a laptops_robustness -n laptops_num_vs_txt
