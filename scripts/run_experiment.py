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
from src.experiment2 import run_experiment


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

MODEL_PARAMS_ALIASES = {
    "qwen0_5": (0.5, False),
    "qwen0_5I": (0.5, True),
    "qwen1_5": (1.5, False),
    "qwen3": (3, False),
    "qwen7": (7, False),
    "qwen7I": (7, True),
    "qwen14": (14, False),
    "qwen14I": (14, True),
    "qwen32": (32, False),
    "qwen32I": (32, True),
    "qwen72": (72, False),
    "qwen72I": (72, True),
}

SET_ALIASES = {
    "colors": alternatives.colors,
    "colorsX": alternatives.colors_X,
    "colors_es": alternatives.colors_es,
    "colors_zh": alternatives.colors_zh,
    "stocks": alternatives.stocks,
    "tickers": alternatives.tickers,
}

TEMPLATE_ALIASES = {
    "colors": prompts.colors_templates,
    "colors_es": prompts.colors_es_templates,
    "colors_zh": prompts.colors_zh_templates,
    "stocks": prompts.stocks_templates,
}

def main(
    exp_name,
    model,
    alternatives,
    templates,
):

    model_name = MODEL_ALIASES.get(model, model)
    alternatives = SET_ALIASES.get(alternatives, alternatives.split(","))
    templates = TEMPLATE_ALIASES.get(templates, [templates])

    print(f"Running experiment with model: {model_name}")
    print(f"Alternatives: {alternatives}")
    print(f"Number of templates: {len(templates)}")
    
    dir_path = os.path.join("experiments", exp_name)
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, "config.json"), "w") as f:
        json.dump({
            "model_name": model_name,
            "alternatives": alternatives,
            "templates": templates,
        }, f, indent=4)

    if 'I' in model_name:
        agent = InstructedHFAgent(model_name)
    else:
        agent = PretrainedAgent(model_name)
    
    run_experiment(agent, alternatives, templates, exp_name)
    # collect_preference_data(model_name, alternatives, templates, exp_name)
    # fit_pref_models(exp_name, alternatives)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full experiment")
    parser.add_argument("--model", type=str, required=True, help="Model name or alias")
    parser.add_argument("--alternatives", type=str, required=True, help="Alternatives alias (en, es, zh, stocks) or comma-separated list")
    parser.add_argument("--templates", type=str, default="stocks", help="Templates alias (en, es, zh, stocks)")
    parser.add_argument("--exp_name", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), help="Experiment name")

    args = parser.parse_args()
    main(
        args.model,
        args.alternatives,
        args.templates,
        args.exp_name
    )
