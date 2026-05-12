import os
import sys
import argparse
import itertools
import pandas as pd

from datetime import datetime
from tqdm.auto import tqdm

# Add project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))
from src import prompts, alternatives
from src.agent import InstructedHFAgent

def collect_data():
    # Parameters
    items = alternatives.colors
    model_id = 'qwen/qwen2.5-0.5B-instruct'
    templates = prompts.options_comparisons
    agent = InstructedHFAgent(model_id)
    labels = ['Option 1', 'Option 2']
    print({
        'items': items,
        'model_id': model_id,
        'templates': templates,
        'labels': labels,
    })

    # Experiment directory
    exp_name = 'qwen0_5I-colors'
    exp_dir = os.path.join("data", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(exp_dir, f"{timestamp}.csv")


    records = []
    for template in tqdm(templates):
        for option_a, option_b in itertools.permutations(items, 2):
            prompt = template.format(A=option_a, B=option_b)
            score_a, score_b = agent.query(prompt, labels)
            records.append({
                'template': template,
                'option_a': option_a,
                'option_b': option_b,
                'prompt': prompt,
                'score_a': score_a,
                'score_b': score_b,
            })
            
    df = pd.DataFrame(records)
    df.to_csv(file_path)
    print("Finished!")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run Data")
    # parser.add_argument("--cluster_job", type=str, default="", help="Job name")
    # parser.add_argument("--model", type=str, required=True, help="Model alias")
    # parser.add_argument("--alternatives", type=str, required=True, help="Alternatives alias")

    # args = parser.parse_args()
    collect_data()