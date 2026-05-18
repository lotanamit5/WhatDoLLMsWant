import itertools

dst_path = "scripts/slurms.sh"
prefix = "sbatch -p bml -A bml"
script_path = "scripts/run_data_collection.sh"

nodes = ['plato1', 'plato2', 'plotinus1', 'plotinus2']

parameters = {
    'm': ['gemma'],
    's':  ['1', '4', '12', '27'],
    'a': ['colors', 'foods', 'cars', 'stocks', 'laptops', 'laptop_brands'],
}

with open(dst_path, 'w') as f:
    for i, combo in enumerate(itertools.product(*parameters.values())):
        flags = " ".join(f"-{k} {v}" for k, v in zip(parameters.keys(), combo))
        node = nodes[i % len(nodes)]
        cmd = f"{prefix} -w {node} {script_path} {flags}"
        f.write(cmd + "\n")