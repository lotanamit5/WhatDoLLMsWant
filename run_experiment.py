import argparse
import sys
import os

# Add src to sys.path to allow imports within src modules to work
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.experiment import run_full_experiment, measure_top5_occurrences, run_perplexity_experiment
from src.prompts import (
    color_en_dict, templates_en, templates_en_instruct,
    color_es_dict, templates_es,
    color_zh_dict, templates_zh
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

COLOR_ALIASES = {
    "en": list(color_en_dict.keys()),
    "es": list(color_es_dict.keys()),
    "zh": list(color_zh_dict.keys()),
}

TEMPLATE_ALIASES = {
    "en": templates_en,
    "enI": templates_en_instruct,
    "es": templates_es,
    "zh": templates_zh,
}

def main():
    parser = argparse.ArgumentParser(description="Run full experiment")
    parser.add_argument("--model", type=str, required=True, help="Model name or alias")
    parser.add_argument("--colors", type=str, required=True, help="Colors alias (en, es, zh) or comma-separated list")
    parser.add_argument("--templates", type=str, required=True, help="Templates alias (en, es, zh)")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")

    args = parser.parse_args()

    model_name = MODEL_ALIASES.get(args.model, args.model)
    
    if args.colors in COLOR_ALIASES:
        colors = COLOR_ALIASES[args.colors]
    else:
        colors = args.colors.split(",")

    if args.templates in TEMPLATE_ALIASES:
        templates = TEMPLATE_ALIASES[args.templates]
    else:
        if "{A}" in args.templates and "{B}" in args.templates:
             templates = [args.templates]
        else:
             raise ValueError(f"Unknown template alias: {args.templates}")

    print(f"Running experiment with model: {model_name}")
    print(f"Colors: {colors}")
    print(f"Number of templates: {len(templates)}")

    run_perplexity_experiment(model_name, colors, templates, args.exp_name)

if __name__ == "__main__":
    main()
