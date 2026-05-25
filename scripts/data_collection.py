import os
import sys
import argparse
import itertools
import json
import huggingface
import pandas as pd

from datetime import datetime
from tqdm.auto import tqdm

# Add project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))
import src.alternatives
import src.prompts
from src.agent import load_gemma3_agent, load_qwen2_5_agent

MODEL_FAMILY_ALIASES = {
    'qwen': load_qwen2_5_agent,
    'gemma': load_gemma3_agent,
}

ALTERNATIVES_ALIASES = {
    'colors': src.alternatives.colors,
    'foods': src.alternatives.foods,
    'cars': src.alternatives.cars,
    'stocks': src.alternatives.stocks,
    'laptops': src.alternatives.laptops,
    'laptop_brands': src.alternatives.laptop_brands,
}

def collect_data(model_family, model_size, alternatives_alias,
                 exp_dir):
    # Parameters
    items = ALTERNATIVES_ALIASES[alternatives_alias]
    templates = src.prompts.options_comparisons
    agent_loader = MODEL_FAMILY_ALIASES[model_family]
    agent = agent_loader(model_size)
    labels = ['Option 1', 'Option 2']
    print('model_id:', agent.tokenizer.name_or_path)
    print('labels:', labels)
    print('items:')
    for item in items:
        print('-', item)
    print('templates:')
    for template in templates:
        print('-', template)

    # Experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = {
        "model_family": model_family,
        "model_size": model_size,
        "alternatives": alternatives_alias,
        "timestamp": timestamp,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        'templates': {i:item for i, item in enumerate(templates)},
    }
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    records = []
    for idx, template in tqdm(enumerate(templates)):
        for option_a, option_b in itertools.permutations(items, 2):
            prompt = template.format(A=option_a, B=option_b)
            score_a, score_b = agent.query(prompt, labels=['Option 1', 'Option 2'])
            records.append({
                'template_idx': idx,
                'option_a': option_a,
                'option_b': option_b,
                'score_a': score_a,
                'score_b': score_b,
            })
            
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(exp_dir, "scores.csv"), index=False)
    print("Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Data")
    parser.add_argument("--model_family", type=str, required=True, help="Model family")
    parser.add_argument("--model_size", type=str, required=True, help="Model size")
    parser.add_argument("--alternatives", type=str, required=True, help="Alternatives alias")
    parser.add_argument("--exp_dir", type=str, required=False, help="Experiment name (optional)")
    
    args = parser.parse_args()
    collect_data(args.model_family, args.model_size, args.alternatives, args.exp_dir)