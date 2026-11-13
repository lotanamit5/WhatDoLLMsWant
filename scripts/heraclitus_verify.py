"""Check a server / environment: re-score a few prompts of an EXISTING run and compare the margins
with the stored ones. Written 2026-11-13 for Heraclitus after the torch upgrade (2.5.1+cu124 ->
2.9.1+cu128); works for any server.

It goes through the normal wrapper and uses the collector's own code for the prompt and the model
(prompt_prefix + template from the stored config.json, item text, agent_factory, query), so what is
compared is the real pipeline:

    sbatch -p bml -A bml -w heraclitus --gres=gpu:1 -c 8 scripts/run_data_collection.sh \
        -y heraclitus_verify.py -m gemma -s 27 -n heraclitus_check

Writes <exp_dir>/verify.csv (one row per prompt, stored vs new) and <exp_dir>/verify_summary.json.
No config.json, so no loader ever takes the folder for a run.
"""
import os
import sys
import json
import argparse

import pandas as pd

import data_collection as dc          # scripts/ is on sys.path when this file is run

# One stored run per model: the same contract (brand=Dell) for all three, all collected after PMI
# was switched off (2026-08-12), all with the `options` templates. An older run would differ by the
# PMI constant, which has nothing to do with the server.
REFERENCE = {
    ('qwen', '72'): 'data/laptops_robustness/1445296',     # 2026-10-11
    ('gemma', '27'): 'data/laptops_robustness/1475800',    # 2026-11-02
    ('olmo', '32'): 'data/laptops_olmo/1475802',           # 2026-11-02
}
N_PER_TEMPLATE = 4                    # x 5 templates = 20 prompts
SEED = 0

parser = argparse.ArgumentParser()
parser.add_argument("--model_family", required=True)
parser.add_argument("--model_size", required=True)
parser.add_argument("--exp_dir", required=True)
args = parser.parse_args()

ref_dir = os.path.join(dc.ROOT, REFERENCE[(args.model_family, args.model_size)])
config = json.load(open(os.path.join(ref_dir, "config.json")))
stored = pd.read_csv(os.path.join(ref_dir, "scores.csv"))
assert (config["model_family"], str(config["model_size"])) == (args.model_family, args.model_size)
templates = [config["templates"][str(i)] for i in range(len(config["templates"]))]
assert templates == dc.TEMPLATE_SETS["options"]["templates"][:len(templates)], "templates changed"
prefix = config["prompt_prefix"]

rows = pd.concat([g.sample(N_PER_TEMPLATE, random_state=SEED + int(t))
                  for t, g in stored.groupby("template_idx")]).sort_index()
features = [c[2:] for c in stored.columns if c.startswith("a_")]

import torch
from src.agent import agent_factory
agent = agent_factory(args.model_family, args.model_size, None)
print("torch", torch.__version__, "| cuda", torch.version.cuda, "|",
      torch.cuda.device_count(), "x", torch.cuda.get_device_name(0))
print("reference run:", REFERENCE[(args.model_family, args.model_size)], "|", config["constraints_id"],
      "| collected", config["timestamp"], "| prompt_prefix:", repr(prefix))

out = []
for i, r in rows.iterrows():
    a = {f: r[f"a_{f}"] for f in features}
    b = {f: r[f"b_{f}"] for f in features}
    prompt = f"{prefix}\n{templates[int(r.template_idx)]}".format(A=dc.item_text(a), B=dc.item_text(b))
    score_a, score_b = dc.query(agent, prompt, show=not out)
    out.append({"row": i, "template_idx": int(r.template_idx), "a": dc.item_text(a), "b": dc.item_text(b),
                "stored_a": r.score_a, "stored_b": r.score_b, "new_a": score_a, "new_b": score_b})
out = pd.DataFrame(out)
out["stored_margin"] = out.stored_a - out.stored_b
out["new_margin"] = out.new_a - out.new_b
out["abs_diff"] = (out.new_margin - out.stored_margin).abs()

os.makedirs(args.exp_dir, exist_ok=True)
out.to_csv(os.path.join(args.exp_dir, "verify.csv"), index=False)
summary = {
    "model": f"{args.model_family}-{args.model_size}",
    "reference_run": REFERENCE[(args.model_family, args.model_size)],
    "n_prompts": len(out),
    "max_abs_diff_margin": float(out.abs_diff.max()),
    "mean_abs_diff_margin": float(out.abs_diff.mean()),
    "max_abs_diff_score": float(max((out.new_a - out.stored_a).abs().max(),
                                    (out.new_b - out.stored_b).abs().max())),
    "mean_abs_stored_margin": float(out.stored_margin.abs().mean()),
    "same_sign": int((out.new_margin * out.stored_margin > 0).sum()),
    "pearson_r": float(out.new_margin.corr(out.stored_margin)),
    "torch": torch.__version__,
    "gpu": f"{torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}",
    "host": os.uname().nodename,
}
with open(os.path.join(args.exp_dir, "verify_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(out[["template_idx", "stored_margin", "new_margin", "abs_diff"]].round(4).to_string())
print(json.dumps(summary, indent=2))
