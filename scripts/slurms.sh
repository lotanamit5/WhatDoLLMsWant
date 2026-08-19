#!/usr/bin/env bash
# GENERATED FILE - do not edit by hand.
# Edit scripts/create_slurms.py and re-run it, then commit BOTH files together.
# (No timestamp on purpose: identical params must produce an identical file,
#  so `git diff` shows a changed batch and nothing else.)
sbatch -p bml -A bml -w plato2 scripts/run_data_collection.sh -m "olmo-pt" -s "32" -a "laptops_robustness" -c "" -p "pretrained" -n laptops_olmo_pt
