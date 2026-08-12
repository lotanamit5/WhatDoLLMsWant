"""Collect pairwise preference scores for one (model, item set, frame, constraints) run.

Merged 2026-08-10 from the old `data_collection.py` and `data_collection_robustness.py`,
which had drifted apart (one had constraints, the other had all templates and all item sets).

Writes `<exp_dir>/config.json` and `<exp_dir>/scores.csv`. See
`docs/config_schema.md` for what goes in the config and why.

The prompt sent to the model is:

    <frame text> <constraints text>
    <template with {A} and {B} filled in>

and `prompt_prefix` in the config records that first line verbatim.
"""
import os
import sys
import json
import argparse
import itertools
import subprocess

import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm

# Add project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
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
    'laptops_num_vs_txt': src.alternatives.laptops_num_vs_txt,
    'laptops_txt_ram_screen': src.alternatives.laptops_txt_ram_screen,
    'laptops_num_ram_screen': src.alternatives.laptops_num_ram_screen,
    'laptops_robustness': src.alternatives.laptops_robustness,
}

# The sentence that sets the scene, before any spec is named. "shopping" is what every
# run up to 2026-08-10 used. The others exist to measure how much the frame itself moves
# the answer - we have never run without one.
FRAMES = {
    'bare': "",
    'shopping': "I am looking to buy a laptop.",
    'self': "You are choosing a laptop for yourself.",
    'third_person': "A person is looking to buy a laptop.",
}

# level string (exactly as it appears in alternatives.py / scores.csv) -> prose
CONSTRAINT_PHRASE = {
    'screen': lambda v: f"a {v} screen",      # "14-inch" -> "a 14-inch screen"
    'ram': lambda v: f"{v[:-2]} GB ram",      # "8GB"     -> "8 GB ram"
    'brand': lambda v: f"a {v} laptop",
}

# Word order of the generated sentence. Fixed here rather than taken from the dict's
# insertion order, because json.dump(sort_keys=True) reorders the dict on the way to disk -
# so insertion order does not survive a round trip and must not be load-bearing.
# This order reproduces the historical strings ("a 14-inch screen and 8 GB ram").
FEATURE_ORDER = ['brand', 'screen', 'ram']


def parse_constraints(spec):
    """'screen=14-inch,ram=8GB' -> {'screen': '14-inch', 'ram': '8GB'}.

    An empty/missing spec means no constraint at all, which is `{}` - never None.
    The order you write them in does not matter; FEATURE_ORDER decides the sentence.
    """
    if not spec:
        return {}
    out = {}
    for part in spec.split(','):
        feature, _, level = part.partition('=')
        if not level:
            raise ValueError(f"constraint {part!r} must look like feature=level, "
                             f"e.g. screen=14-inch")
        out[feature.strip()] = level.strip()
    return out


def constraints_text(constraints):
    if not constraints:
        return ""
    ordered = sorted(constraints, key=FEATURE_ORDER.index)
    parts = [CONSTRAINT_PHRASE[f](constraints[f]) for f in ordered]
    return "I prefer " + " and ".join(parts) + "."


def constraints_id(constraints):
    """Unambiguous slug for filtering and folder names. Derived, never typed by hand -
    which is the whole point: `screen=16-inch` and `ram=16GB` used to both be "16"."""
    return "+".join(f"{f}={v}" for f, v in sorted(constraints.items())) or "none"


def validate_constraints(constraints, items):
    """A constraint naming a level no item has would silently produce a run where nothing
    satisfies the contract. That has happened; it is worth the six lines."""
    for feature, level in constraints.items():
        if feature not in CONSTRAINT_PHRASE:
            raise ValueError(f"no prose template for feature {feature!r}; "
                             f"add one to CONSTRAINT_PHRASE")
        levels = {item[feature] for item in items if feature in item}
        if not levels:
            raise ValueError(f"items have no feature {feature!r}")
        if level not in levels:
            raise ValueError(f"{feature}={level!r} is not a level of {feature}; "
                             f"choose one of {sorted(levels)}")


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def format_features_to_text(item_dict):
    return (f"{item_dict.get('screen', '')} {item_dict.get('brand', '')} "
            f"laptop with {item_dict.get('ram', '')} RAM").strip()


def collect_data(model_family, model_size, alternatives_alias, exp_dir,
                 constraints=None, frame='shopping', n_templates=5):
    items = ALTERNATIVES_ALIASES[alternatives_alias]
    constraints = constraints or {}
    validate_constraints(constraints, items)

    templates = src.prompts.options_comparisons
    if n_templates:
        templates = templates[:n_templates]

    frame_text = FRAMES[frame]
    con_text = constraints_text(constraints)
    prompt_prefix = " ".join(p for p in (frame_text, con_text) if p)

    agent = agent_factory(model_family, model_size)
    print('model_id:', agent.tokenizer.name_or_path)
    print('prompt_prefix:', repr(prompt_prefix))
    print('constraints:', constraints)
    print('items:')
    for item in items:
        print('-', item)
    print('templates:')
    for template in templates:
        print('-', template)

    config = {
        "model_family": model_family,
        "model_size": model_size,
        "alternatives": alternatives_alias,

        "frame": {"name": frame, "text": frame_text},
        "constraints": constraints,
        "constraints_text": con_text,
        "constraints_id": constraints_id(constraints),
        "prompt_prefix": prompt_prefix,

        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": git_commit(),
        "collection_script": "scripts/data_collection.py",
        "templates": {i: t for i, t in enumerate(templates)},
    }
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    records = []
    for idx, template in tqdm(enumerate(templates)):
        full_template = f"{prompt_prefix}\n{template}" if prompt_prefix else template
        for option_a, option_b in itertools.permutations(items, 2):
            prompt = full_template.format(A=format_features_to_text(option_a),
                                          B=format_features_to_text(option_b))
            score_a, score_b = agent.query(prompt)
            record = {'template_idx': idx, 'score_a': score_a, 'score_b': score_b}
            record.update({f"a_{k}": v for k, v in option_a.items()})
            record.update({f"b_{k}": v for k, v in option_b.items()})
            records.append(record)

    pd.DataFrame(records).to_csv(os.path.join(exp_dir, "scores.csv"), index=False)
    print("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect pairwise preference scores")
    parser.add_argument("--model_family", type=str, required=True)
    parser.add_argument("--model_size", type=str, required=True)
    parser.add_argument("--alternatives", type=str, required=True)
    parser.add_argument("--exp_dir", type=str, required=False)
    parser.add_argument("--constraints", type=str, default="",
                        help='"screen=14-inch,ram=8GB". Empty means no constraint.')
    parser.add_argument("--frame", type=str, default="shopping", choices=sorted(FRAMES),
                        help="the scene-setting sentence before any spec")
    parser.add_argument("--n_templates", type=int, default=5,
                        help="use the first N prompt templates; 0 means all")

    args = parser.parse_args()
    collect_data(
        args.model_family,
        args.model_size,
        args.alternatives,
        args.exp_dir,
        constraints=parse_constraints(args.constraints),
        frame=args.frame,
        n_templates=args.n_templates,
    )
