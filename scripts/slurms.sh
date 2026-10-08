#!/usr/bin/env bash
# GENERATED FILE - do not edit by hand.
# Edit scripts/create_slurms.py and re-run it, then commit BOTH files together.
# (No timestamp on purpose: identical params must produce an identical file,
#  so `git diff` shows a changed batch and nothing else.)
sbatch -p bml -A bml -w plato1 scripts/run_data_collection.sh -m "qwen" -s "32" -a "laptops_robustness" -c "" -f "student_only" -n laptops_persona
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "qwen" -s "32" -a "laptops_robustness" -c "" -f "no_money" -n laptops_persona
sbatch -p bml -A bml -w plato1 scripts/run_data_collection.sh -m "qwen" -s "32" -a "laptops_robustness" -c "" -f "windows" -n laptops_persona
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "qwen" -s "32" -a "laptops_robustness" -c "" -f "designer" -n laptops_persona
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "qwen" -s "72" -a "laptops_robustness" -c "" -f "student_only" -n laptops_persona
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "qwen" -s "72" -a "laptops_robustness" -c "" -f "no_money" -n laptops_persona
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "qwen" -s "72" -a "laptops_robustness" -c "" -f "windows" -n laptops_persona
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "qwen" -s "72" -a "laptops_robustness" -c "" -f "designer" -n laptops_persona
sbatch -p bml -A bml -w plato1 scripts/run_data_collection.sh -m "qwen" -s "32" -a "laptops_robustness" -c "" -f "role_none" -n laptops_persona
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "qwen" -s "32" -a "laptops_robustness" -c "" -f "role_student" -n laptops_persona
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "qwen" -s "72" -a "laptops_robustness" -c "" -f "role_none" -n laptops_persona
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "qwen" -s "72" -a "laptops_robustness" -c "" -f "role_student" -n laptops_persona
