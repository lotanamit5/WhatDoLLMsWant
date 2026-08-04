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
from src.agent import agent_factory


ALTERNATIVES_ALIASES = {
    'colors': src.alternatives.colors,
    'foods': src.alternatives.foods,
    'cars': src.alternatives.cars,
    'stocks': src.alternatives.stocks,
    'laptops': src.alternatives.laptops,
    'laptop_brands': src.alternatives.laptop_brands,
    'laptops_robustness': src.alternatives.laptops_robustness
}

def format_features_to_text(item_dict):
    return f"{item_dict.get('screen', '')} {item_dict.get('brand', '')} laptop with {item_dict.get('ram', '')} RAM".strip()

def collect_data(model_family, model_size, alternatives_alias,
                 exp_dir):
    # Parameters
    items = ALTERNATIVES_ALIASES[alternatives_alias]
    templates = src.prompts.options_comparisons[:5]
    agent = agent_factory(model_family, model_size)
    print('model_id:', agent.tokenizer.name_or_path)
    print('items:')
    for item in items:
        print('-', item)
    print('templates:')
    for template in templates:
        print('-', template)

    # Experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    constraint = "I am looking to buy a laptop.\n"
    # constraint = "I am looking to buy a laptop. I prefer a 14-inch screen.\n"
    # constraint = "I am looking to buy a laptop. I prefer a 14-inch screen and 8 GB ram.\n"
    
    config = {
        "model_family": model_family,
        "model_size": model_size,
        "alternatives": alternatives_alias,
        "timestamp": timestamp,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        'templates': {i:item for i, item in enumerate(templates)},
        'constraint': constraint,
    }
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    records = []
    for idx, template in tqdm(enumerate(templates)):
        template = constraint + template
        for option_a, option_b in itertools.permutations(items, 2):
            text_a = format_features_to_text(option_a)
            text_b = format_features_to_text(option_b)
            prompt = template.format(A=text_a, B=text_b)
            score_a, score_b = agent.query(prompt)
            record = {
                'template_idx': idx,
                'score_a': score_a,
                'score_b': score_b,
            }
            record.update({f"a_{k}": v for k, v in option_a.items()})
            record.update({f"b_{k}": v for k, v in option_b.items()})
            records.append(record)
            
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
    collect_data(
        args.model_family,
        args.model_size,
        args.alternatives,
        args.exp_dir
    )
