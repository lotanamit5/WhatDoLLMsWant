import os
import sys
import argparse
import itertools
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

def collect_data(model_family, model_size, alternatives_alias):
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
    exp_name = f'{model_family}-{model_size}-{alternatives_alias}-{timestamp}'
    exp_dir = os.path.join("data",'prev_prompts', exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    records = []
    for template in tqdm(templates):
        for option_a, option_b in itertools.permutations(items, 2):
            prompt = template.format(A=option_a, B=option_b)
            score_a, score_b = agent.query(prompt, labels=[option_a, option_b])
            records.append({
                'template': rf'{template}',
                'option_a': option_a,
                'option_b': option_b,
                'prompt': rf'{prompt}',
                'score_a': score_a.item(),
                'score_b': score_b.item(),
            })
            
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(exp_dir, f"scores.csv"), index=False)
    print("Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Data")
    parser.add_argument("--model_family", type=str, required=True, help="Model family")
    parser.add_argument("--model_size", type=str, required=True, help="Model size")
    parser.add_argument("--alternatives", type=str, required=True, help="Alternatives alias")

    args = parser.parse_args()
    collect_data(args.model_family, args.model_size, args.alternatives)