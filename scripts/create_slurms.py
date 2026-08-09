import os
import sys
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

exp_name = "laptops_robustness_gemma" # CHANGE WHEN RUNNING
nodes = [
    'plato2',
    'plotinus1',
    'plotinus2',
]

parameters = {
    'm': ['gemma'],
    's': ['1'],
    'a': [
        # 'colors',
        #   'foods', 'cars', 'stocks', 'laptops',
        #   'laptop_brands'
        'laptops_robustness',
        # 'laptops_txt_ram_screen','laptops_num_ram_screen',
          ],
    'c': [''],
}

dst_path = "scripts/slurms.sh"
prefix = "sbatch -p bml -A bml"
script_path = "scripts/run_data_collection_robust.sh"

with open(dst_path, 'w') as f:
    for i, combo in enumerate(itertools.product(*parameters.values())):
        # Quote each value - an empty string (the no-constraint case) must
        # still produce a real, non-empty shell token ("" not nothing), or
        # the flag before it silently swallows the NEXT flag's value once
        # this line is word-split by the shell.
        flags = " ".join(f'-{k} "{v}"' for k, v in zip(parameters.keys(), combo))
        node = nodes[i % len(nodes)]
        cmd = f"{prefix} -w {node} {script_path} {flags} -n {exp_name}"
        f.write(cmd + "\n")