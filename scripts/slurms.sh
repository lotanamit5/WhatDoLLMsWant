#!/usr/bin/env bash
# GENERATED FILE - do not edit by hand.
# Edit scripts/create_slurms.py and re-run it, then commit BOTH files together.
# (No timestamp on purpose: identical params must produce an identical file,
#  so `git diff` shows a changed batch and nothing else.)
sbatch -p bml -A bml -w plotinus1 scripts/run_data_collection.sh -m "qwen" -s "7" -a "books_small" -c "" -f "reading" -n books_pilot
sbatch -p bml -A bml -w plotinus2 scripts/run_data_collection.sh -m "qwen" -s "7" -a "books_small" -c "fantasy=yes" -f "reading" -n books_pilot
