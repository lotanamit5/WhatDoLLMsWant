"""Collect pairwise scores for the "real laptops" domain: one model, many contracts, ONE model load.

A new domain, kept apart from the synthetic laptop runs. Reuses the existing pipeline unchanged:
the agent (src/agent.py), the template (src/prompts.py), and the frame + contract wording
(scripts/data_collection.py). Only the items and the pair lists are new; both come from
data/real_laptops/design/, written by scripts/real_laptops_prepare.py.

    --mode main    27 contracts, template 0 only.
                   none + 6 singles: all 1,770 pairs, both orders (3,540 prompts each).
                   20 compounds:     the fixed 500 pairs, both orders (1,000 prompts each).
    --mode sanity  no contract, the fixed 200 pairs, both orders, the first 5 templates (2,000 prompts).

Each contract is one run:  <exp_root>/<job_id>_<constraints_id>/{config.json, scores.csv}
scores.csv: template_idx, score_a, score_b, a_id, a_brand, a_screen, a_ram, b_id, b_brand, b_screen, b_ram
(`id` is laptop_ID; every other spec of an item is in design/items.csv).

    --dry_run      print the first prompt of each selected contract and stop. Needs no GPU and no torch.
"""
import argparse
import contextlib
import json
import os
import sys
import types
from datetime import datetime

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--model_family", default="qwen")
parser.add_argument("--model_size", default="72")
parser.add_argument("--mode", choices=["main", "sanity"], default="main")
parser.add_argument("--exp_root", default=None, help="absolute folder that receives the run folders")
parser.add_argument("--job_id", default=os.environ.get("SLURM_JOB_ID", "local"))
parser.add_argument("--only", default="", help='run only these contracts, ";"-separated specs as in '
                                               'contracts.csv ("none" for no contract). For reruns.')
parser.add_argument("--dry_run", action="store_true")
args = parser.parse_args()

if args.dry_run:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:           # the analysis machine: wording code only, no agent
        sys.modules["src.agent"] = types.SimpleNamespace(agent_factory=None)
import data_collection as dc               # FRAMES, constraints_text, constraints_id, TEMPLATE_SETS

DESIGN = os.path.join(ROOT, "data", "real_laptops", "design")
FRAME, TEMPLATE_SET, FEATS = "shopping", "options", ["brand", "screen", "ram"]

items = pd.read_csv(os.path.join(DESIGN, "items.csv")).set_index("id")
contracts = pd.read_csv(os.path.join(DESIGN, "contracts.csv"), keep_default_na=False)
all_pairs = [(x, y) for i, x in enumerate(items.index) for y in items.index[i + 1:]]
compound_pairs = list(pd.read_csv(os.path.join(DESIGN, "compound_pairs.csv")).itertuples(index=False, name=None))
sanity_pairs = list(pd.read_csv(os.path.join(DESIGN, "sanity_pairs.csv")).itertuples(index=False, name=None))
assert len(items) == 60 and len(all_pairs) == 1770 and len(compound_pairs) == 500 and len(sanity_pairs) == 200

templates_all = dc.TEMPLATE_SETS[TEMPLATE_SET]["templates"]
if args.mode == "main":
    jobs = [(r.spec, "all_1770" if r.set == "train" else "compound_500",
             all_pairs if r.set == "train" else compound_pairs, 1) for r in contracts.itertuples()]
else:
    jobs = [("", "sanity_200", sanity_pairs, 5)]
if args.only:
    keep = {"" if s == "none" else s for s in args.only.split(";")}
    assert keep <= {j[0] for j in jobs}, f"unknown contract in --only: {keep - {j[0] for j in jobs}}"
    jobs = [j for j in jobs if j[0] in keep]
print(f"{args.model_family}-{args.model_size} | mode {args.mode} | {len(jobs)} runs | "
      f"{sum(len(p) * 2 * t for _, _, p, t in jobs)} prompts")


def columns(prefix, i):
    return {f"{prefix}id": i, **{f"{prefix}{f}": items.at[i, f] for f in FEATS}}


agent = None
devnull = open(os.devnull, "w")
if not args.dry_run:
    from src.agent import agent_factory
    agent = agent_factory(args.model_family, args.model_size, dc.TEMPLATE_SETS[TEMPLATE_SET]["labels"])

for spec, pairs_name, pairs, n_templates in jobs:
    constraints = dc.parse_constraints(spec)
    for f, v in constraints.items():
        assert v in set(items[f]), f"{f}={v} is not a level of any item"
    templates = templates_all[:n_templates]
    frame_text, con_text = dc.FRAMES[FRAME], dc.constraints_text(constraints)
    prompt_prefix = " ".join(p for p in (frame_text, con_text) if p)
    cid = dc.constraints_id(constraints)

    if args.dry_run:
        x, y = pairs[0]
        print(f"\n===== {cid} ({pairs_name}, {len(pairs) * 2 * len(templates)} prompts) =====")
        print(f"{prompt_prefix}\n{templates[0]}".format(A=items.at[x, "text"], B=items.at[y, "text"]))
        continue

    run_dir = os.path.join(args.exp_root, f"{args.job_id}_{cid}")
    os.makedirs(run_dir, exist_ok=True)
    config = {
        "model_family": args.model_family, "model_size": args.model_size,
        "alternatives": "real_laptops", "items_file": "data/real_laptops/design/items.csv",
        "pairs": pairs_name, "n_pairs": len(pairs),
        "frame": {"name": FRAME, "text": frame_text, "voice": dc.frame_voice(FRAME)},
        "constraints": constraints, "constraints_text": con_text, "constraints_id": cid,
        "prompt_prefix": prompt_prefix,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"), "slurm_job_id": args.job_id,
        "git_commit": dc.git_commit(), "collection_script": "scripts/real_laptops_collect.py",
        "template_set": TEMPLATE_SET, "labels": agent.labels,
        "templates": {i: t for i, t in enumerate(templates)},
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    print(f"=== {cid}: {len(pairs) * 2 * len(templates)} prompts -> {run_dir}", flush=True)
    records = []
    for idx, template in enumerate(templates):
        full_template = f"{prompt_prefix}\n{template}"
        for x, y in pairs:
            for a, b in ((x, y), (y, x)):
                prompt = full_template.format(A=items.at[a, "text"], B=items.at[b, "text"])
                # the agent prints every prompt it scores; keep only the first one of each run in the log
                if records:
                    with contextlib.redirect_stdout(devnull):
                        score_a, score_b = agent.query(prompt)
                else:
                    score_a, score_b = agent.query(prompt)
                records.append({"template_idx": idx, "score_a": score_a, "score_b": score_b,
                                **columns("a_", a), **columns("b_", b)})
    pd.DataFrame(records).to_csv(os.path.join(run_dir, "scores.csv"), index=False)
    print(f"    done: {len(records)} rows", flush=True)
print("Finished!")
