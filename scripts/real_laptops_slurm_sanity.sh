#!/usr/bin/env bash
# Real laptops, stage 1: the template sanity check. ONE job, 2,000 prompts
# (no contract, 200 pairs, both orders, 5 templates) -> data/real_laptops_template_check/
sbatch -p bml -A bml -w plato2 scripts/run_real_laptops.sh -m "gemma" -s "27" -o "sanity" -n real_laptops_template_check
