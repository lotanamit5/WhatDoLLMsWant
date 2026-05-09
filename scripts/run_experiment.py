import argparse
import itertools
import json
import sys
import os
from datetime import datetime

import pandas as pd
from tqdm import tqdm
# Add project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import prompts, alternatives
# from src.experiment import collect_preference_data, fit_pref_models
from src.agent import InstructedHFAgent, PretrainedAgent, agent_factory


MODEL_ALIASES = {
    "qwen0_5": ["qwen25", 0.5, False],
    "qwen0_5I": ["qwen25", 0.5, True],
    "qwen1_5": ["qwen25", 1.5, False],
    "qwen3": ["qwen25", 3, False],
    "qwen7": ["qwen25", 7, False],
    "qwen7I": ["qwen25", 7, True],
    "qwen14": ["qwen25", 14, False],
    "qwen14I": ["qwen25", 14, True],
    "qwen32": ["qwen25", 32, False],
    "qwen32I": ["qwen25", 32, True],
    "qwen72": ["qwen25", 72, False],
    "qwen72I": ["qwen25", 72, True],
    "gemma1": ["gemma3", 1, False],
    "gemma1I": ["gemma3", 1, True],
    "gemma4": ["gemma3", 4, False],
    "gemma4I": ["gemma3", 4, True],
    "gemma12": ["gemma3", 12, False],
    "gemma12I": ["gemma3", 12, True],
    "gemma27": ["gemma3", 27, False],
    "gemma27I": ["gemma3", 27, True],
}

SET_ALIASES = {
    "colors": alternatives.colors,
    "colorsX": alternatives.colors_X,
    "colors_es": alternatives.colors_es,
    "colors_zh": alternatives.colors_zh,
    "stocks": alternatives.stocks,
    "tickers": alternatives.tickers,
    "foods": alternatives.foods,
    "cars": alternatives.cars,
    "gifts": alternatives.gifts_v1,
    "gifts_small": alternatives.gifts_small,
    "colored_cars": alternatives.colored_cars,
    'laptops': alternatives.laptops,
}

TEMPLATE_ALIASES = {
    'default': prompts.general_comparisons,
    'sanity': prompts.sanity_check_colors,
    "colors": prompts.colors_templates,
    "colors_es": prompts.colors_es_templates,
    "colors_zh": prompts.colors_zh_templates,
    "stocks": prompts.stocks_templates,
}

TASKS_ALIASES = {
    "gifts_vague": prompts.gifts_vague_general,
    "gifts_feature": prompts.gifts_feature_general,
    "gifts_constrained": prompts.gifts_constrained_general,
    "cc_general": prompts.cc_general,
    "cc_red": prompts.cc_red,
    "cc_purple": prompts.cc_purple,
    "cc_red_audi": prompts.cc_red_audi,
    "cc_purple_audi": prompts.cc_purple_audi,
    "cc_red_tesla": prompts.cc_red_tesla,
    "cc_purple_tesla": prompts.cc_purple_tesla,
}

def main(
    exp_name,
    model,
    alternatives,
    templates,
    pairwise=True,
    cluster_job=None
    ):

    print(f"Running experiment with model: {model}")
    print(f"Alternatives: {alternatives}")
    print(f"Templates: {templates}")
    
    dir_path = os.path.join("experiments", exp_name)
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, "config.json"), "w") as f:
        json.dump({
            "cluster_job": cluster_job,
            "model_name": model,
            "alternatives": alternatives,
            "templates": templates,
        }, f, indent=4)

    model_cls, model_size, model_instruct = MODEL_ALIASES.get(model, model)
    alternatives = SET_ALIASES.get(alternatives, alternatives.split(","))
    templates = TEMPLATE_ALIASES[templates] if pairwise else TASKS_ALIASES[templates]
    print(f"Number of templates: {len(templates)}")

    agent = agent_factory(model_cls, model_size, model_instruct)
    print(f"Loaded model: {agent.model_id}")
    
    def query_all_pairs(agent, alternatives, template):
        results = []
        for a1, a2 in itertools.combinations(alternatives, 2):
            scores = agent.query(template, [a1, a2])
            # Store or process scores as needed
            results.append({
                "template": template,
                "item_a": a1,
                "item_b": a2,
                "score_a": scores[0],
                "score_b": scores[1]
            })
        return pd.DataFrame(results)

    pref_data = pd.DataFrame()
    for i, template in tqdm(enumerate(templates)):
        current_data = query_all_pairs(agent, alternatives, template)
        pref_data = pd.concat([pref_data, current_data.assign(iteration=i)], ignore_index=True)
        
    pref_data.to_csv(f"experiments/{exp_name}/scores.csv", index=False)
    print(f"Finished with {len(pref_data)} preference data points.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full experiment")
    parser.add_argument("--cluster_job", type=str, default="", help="Job name or alias")
    parser.add_argument("--model", type=str, required=True, help="Model name or alias")
    parser.add_argument("--alternatives", type=str, required=True, help="Alternatives alias (en, es, zh, stocks) or comma-separated list")
    parser.add_argument("--templates", type=str, default="default", help="Templates alias (en, es, zh, stocks)")
    parser.add_argument("--pairwise", action="store_true", default=True, help="Use pairwise comparisons (default: True)")
    parser.add_argument("--task", action="store_true", help="Run in task mode (pairwise=False)")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")

    args = parser.parse_args()

    if not args.exp_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"{args.alternatives}-{args.model}-{timestamp}"
    
    # Logic: pairwise is false if task is set, otherwise true (unless explicitly pairwise logic was intended otherwise, but the user requested pairwise=!task)
    pairwise = not args.task
    
    main(
        args.exp_name,
        args.model,
        args.alternatives,
        args.templates,
        pairwise,
        args.cluster_job
    )
