import argparse
import sys
import os

# Add project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.experiment import collect_preference_data, fit_pref_models
from src.prompts import (
    color_en_dict, color_en_extended_dict, templates_en, templates_en_instruct,
    color_es_dict, templates_es,
    color_zh_dict, templates_zh,
    stocks_names_dict, stocks_templates
)

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
    "en": list(color_en_dict.keys()),
    "en_X": list(color_en_extended_dict.keys()),
    "es": list(color_es_dict.keys()),
    "zh": list(color_zh_dict.keys()),
    "stocks": list(stocks_names_dict.keys()),
}

TEMPLATE_ALIASES = {
    "en": templates_en,
    "enI": templates_en_instruct,
    "es": templates_es,
    "zh": templates_zh,
    "stocks": stocks_templates,
}

def main():
    parser = argparse.ArgumentParser(description="Run full experiment")
    parser.add_argument("--model", type=str, required=True, help="Model name or alias")
    parser.add_argument("--alternatives", type=str, required=True, help="Alternatives alias (en, es, zh, stocks) or comma-separated list")
    parser.add_argument("--templates", type=str, required=True, help="Templates alias (en, es, zh, stocks)")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")

    args = parser.parse_args()

    model_name = MODEL_ALIASES.get(args.model, args.model)
    
    if args.alternatives in SET_ALIASES:
        alternatives = SET_ALIASES[args.alternatives]
    else:
        alternatives = args.alternatives.split(",")

    if args.templates in TEMPLATE_ALIASES:
        templates = TEMPLATE_ALIASES[args.templates]
    else:
        if "{A}" in args.templates and "{B}" in args.templates:
             templates = [args.templates]
        else:
             raise ValueError(f"Unknown template alias: {args.templates}")

    print(f"Running experiment with model: {model_name}")
    print(f"Alternatives: {alternatives}")
    print(f"Number of templates: {len(templates)}")

    collect_preference_data(model_name, alternatives, templates, args.exp_name)
    fit_pref_models(args.exp_name, alternatives)
if __name__ == "__main__":
    main()
