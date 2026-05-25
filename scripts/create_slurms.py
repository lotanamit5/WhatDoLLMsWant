import os
import sys
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import qwen2_5_sizes, gemma3_sizes

exp_name = "pmi_qwen" # CHANGE WHEN RUNNING
nodes = ['plotinus1', 'plotinus2']

parameters = {
    'm': ['qwen'],
    's':  ['72'],
    'a': ['colors', 'foods', 'cars', 'stocks', 'laptops', 'laptop_brands'],
}

dst_path = "scripts/slurms.sh"
prefix = "sbatch -p bml -A bml"
script_path = "scripts/run_data_collection.sh"

with open(dst_path, 'w') as f:
    for i, combo in enumerate(itertools.product(*parameters.values())):
        flags = " ".join(f"-{k} {v}" for k, v in zip(parameters.keys(), combo))
        node = nodes[i % len(nodes)]
        cmd = f"{prefix} -w {node} {script_path} {flags} -n {exp_name}"
        f.write(cmd + "\n")