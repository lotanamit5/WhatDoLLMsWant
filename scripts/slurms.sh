sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m qwen -s 72 -a colors -n pmi_qwen
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m qwen -s 72 -a foods -n pmi_qwen
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m qwen -s 72 -a cars -n pmi_qwen
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m qwen -s 72 -a stocks -n pmi_qwen
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m qwen -s 72 -a laptops -n pmi_qwen
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m qwen -s 72 -a laptop_brands -n pmi_qwen
