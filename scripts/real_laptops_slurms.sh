#!/usr/bin/env bash
# Real laptops, stage 2: the main runs. DO NOT LAUNCH before the stage-1 report is approved.
# One job per model, 27 contracts each, 44,780 prompts per model -> data/real_laptops/
sbatch -p bml -A bml -w plotinus1 scripts/run_real_laptops.sh -m "qwen" -s "72" -o "main" -n real_laptops
sbatch -p bml -A bml -w plato2 scripts/run_real_laptops.sh -m "gemma" -s "27" -o "main" -n real_laptops
sbatch -p bml -A bml -w plato1 scripts/run_real_laptops.sh -m "olmo" -s "32" -o "main" -n real_laptops
