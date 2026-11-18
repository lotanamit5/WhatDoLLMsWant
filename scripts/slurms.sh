#!/usr/bin/env bash
# GENERATED FILE - do not edit by hand.
# Edit scripts/create_slurms.py and re-run it, then commit BOTH files together.
# (No timestamp on purpose: identical params must produce an identical file,
#  so `git diff` shows a changed batch and nothing else.)
sbatch -p bml -A bml -w heraclitus scripts/run_data_collection.sh -m "qwen" -s "72" -a "real_laptops_45_named" -c "" -t "5" -n real_laptops_45_named
sbatch -p bml -A bml -w heraclitus scripts/run_data_collection.sh -m "qwen" -s "72" -a "real_laptops_45_hidden" -c "" -t "5" -n real_laptops_45_hidden
sbatch -p bml -A bml -w heraclitus scripts/run_data_collection.sh -m "gemma" -s "27" -a "real_laptops_45_named" -c "" -t "5" -n real_laptops_45_named
sbatch -p bml -A bml -w heraclitus scripts/run_data_collection.sh -m "gemma" -s "27" -a "real_laptops_45_hidden" -c "" -t "5" -n real_laptops_45_hidden
sbatch -p bml -A bml -w heraclitus scripts/run_data_collection.sh -m "olmo" -s "32" -a "real_laptops_45_named" -c "" -t "5" -n real_laptops_45_named
sbatch -p bml -A bml -w heraclitus scripts/run_data_collection.sh -m "olmo" -s "32" -a "real_laptops_45_hidden" -c "" -t "5" -n real_laptops_45_hidden
