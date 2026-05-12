dst_path = "scripts/slurms.sh"
prefix = "sbatch -p bml -A bml"
script_path = "scripts/run_data_collection.sh"

nodes = ['plato1', 'plato2', 'plotinus1', 'plotinus2']

parameters = {
    'm': ['qwen'],
    ('s','w'): [(0.5, 'plato1'), (7, 'plato2'), (32, 'plotinus1'), (72, 'plotinus2')],
    'a': ['colors', 'foods', 'cars', 'stocks', 'laptops', 'laptop_brands'],
}

with open(dst_path, 'w') as f:
    for model in parameters['m']:
        for size, node in parameters[('s','w')]:
            for alternatives in parameters['a']:
                cmd = f"{prefix} -w {node} {script_path} -a {alternatives} -m {model} -s {size}"
                f.write(cmd + "\n")
