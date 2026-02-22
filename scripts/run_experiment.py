import argparse
import json
import sys
import os
from datetime import datetime
# Add project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import prompts, alternatives
# from src.experiment import collect_preference_data, fit_pref_models
from agent import InstructedHFAgent, PretrainedAgent
from src.experiment import run_experiment


MODEL_ALIASES = {
    "qwen0_5": "Qwen/Qwen2.5-0.5B",
    "qwen0_5I": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1_5": "Qwen/Qwen2.5-1.5B",
    "qwen3": "Qwen/Qwen2.5-3B",
    "qwen7": "Qwen/Qwen2.5-7B",
    "qwen7I": "Qwen/Qwen2.5-7B-Instruct",
    "qwen14": "Qwen/Qwen2.5-14B",
    "qwen14I": "Qwen/Qwen2.5-14B-Instruct",
    "qwen32": "Qwen/Qwen2.5-32B",
    "qwen32I": "Qwen/Qwen2.5-32B-Instruct",
    "qwen72": "Qwen/Qwen2.5-72B",
    "qwen72I": "Qwen/Qwen2.5-72B-Instruct",
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
}

TEMPLATE_ALIASES = {
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

    model_name = MODEL_ALIASES.get(model, model)
    alternatives = SET_ALIASES.get(alternatives, alternatives.split(","))
    templates = TEMPLATE_ALIASES[templates] if pairwise else TASKS_ALIASES[templates]
    print(f"Number of templates: {len(templates)}")

    if 'I' in model_name:
        agent = InstructedHFAgent(model_name)
    else:
        agent = PretrainedAgent(model_name)
    
    run_experiment(agent, alternatives, templates, exp_name, pairwise=pairwise)
    # collect_preference_data(model_name, alternatives, templates, exp_name)
    # fit_pref_models(exp_name, alternatives)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full experiment")
    parser.add_argument("--cluster_job", type=str, default="", help="Job name or alias")
    parser.add_argument("--model", type=str, required=True, help="Model name or alias")
    parser.add_argument("--alternatives", type=str, required=True, help="Alternatives alias (en, es, zh, stocks) or comma-separated list")
    parser.add_argument("--templates", type=str, default="stocks", help="Templates alias (en, es, zh, stocks)")
    parser.add_argument("--pairwise", action="store_true", default=True, help="Use pairwise comparisons (default: True)")
    parser.add_argument("--task", action="store_true", help="Run in task mode (pairwise=False)")
    parser.add_argument("--exp_name", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), help="Experiment name")

    args = parser.parse_args()
    
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
