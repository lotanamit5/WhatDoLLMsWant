sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "olmo" -s "1" -a "laptops_robustness" -c "" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "olmo" -s "7" -a "laptops_robustness" -c "" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "olmo" -s "13" -a "laptops_robustness" -c "" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "olmo-pt" -s "7" -a "laptops_robustness" -c "" -p "pretrained" -n laptops_olmo_pt
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "olmo-pt" -s "13" -a "laptops_robustness" -c "" -p "pretrained" -n laptops_olmo_pt
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "olmo" -s "32" -a "laptops_robustness" -c "" -p "options" -n laptops_olmo
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "olmo-pt" -s "32" -a "laptops_robustness" -c "" -p "pretrained" -n laptops_olmo_pt
