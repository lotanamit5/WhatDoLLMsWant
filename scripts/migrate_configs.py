"""Rewrite every data/*/*/config.json into the schema in docs/config_schema.md.

Dry run by default. `--apply` writes.

Touches config.json ONLY. Never scores.csv. Every config is tracked in git, so a bad run
is undone with `git checkout -- data`.

Usage:
    python scripts/migrate_configs.py            # show what would change
    python scripts/migrate_configs.py --apply    # write it
"""
import os
import sys
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.data_collection import constraints_id, constraints_text, FRAMES

DATA_ROOT = "data"

# The historical slug -> what it actually meant. Taken from CONSTRAINT_ALIASES as it stood
# in scripts/data_collection_robustness.py before the 2026-08-10 merge.
OLD_SLUG = {
    "": {},
    "None": {},
    "14": {"screen": "14-inch"},
    "8": {"ram": "8GB"},
    "14_8": {"screen": "14-inch", "ram": "8GB"},
}

# The exact prefix strings that were sent, so we backfill history rather than regenerate it.
OLD_PREFIX = {
    "": "I am looking to buy a laptop.",
    "None": "I am looking to buy a laptop.",
    "14": "I am looking to buy a laptop. I prefer a 14-inch screen.",
    "8": "I am looking to buy a laptop. I prefer 8 GB ram.",
    "14_8": "I am looking to buy a laptop. I prefer a 14-inch screen and 8 GB ram.",
}

# Experiment folders collected before any prefix existed: the prompt was the bare template.
BARE_SETS = {"qwen_pt", "pmi_qwen", "eos_fix_qwen", "num_vs_txt",
             "laptops_num_vs_txt", "new_prompts", "prev_prompts", "gemma"}

# Runs whose config never recorded the truth and had to be recovered by hand.
SPECIAL = {
    ("laptops_robustness_50prompts", "68337011"): {
        "constraints": {"screen": "14-inch"},
        "frame": "shopping",
        "prompt_prefix": "I am looking to buy a laptop. I prefer a 14-inch screen.",
        "note": ("constraints recovered 2026-08-10 by row-matching against run 68337596 "
                 "(agree to 0.25 log-odds; the true unconstrained run differs by up to "
                 "25.8). The run predates the constraints field. 43 templates, not 5."),
    },
}


def migrate_one(exp_set, run_id, config):
    """Return the new config dict, or None if it is already migrated."""
    if isinstance(config.get("constraints"), dict) and "frame" in config:
        return None

    special = SPECIAL.get((exp_set, run_id))
    if special:
        constraints = special["constraints"]
        frame_name = special["frame"]
        prefix = special["prompt_prefix"]
        note = special["note"]
    elif exp_set in BARE_SETS:
        constraints, frame_name, prefix, note = {}, "bare", "", None
    else:
        slug = config.get("constraints") or ""
        if slug not in OLD_SLUG:
            raise ValueError(f"{exp_set}/{run_id}: unknown constraint slug {slug!r}")
        constraints = OLD_SLUG[slug]
        frame_name = "shopping"
        prefix = OLD_PREFIX[slug]
        note = None
        # If the old config recorded the text, trust it over our table.
        if config.get("constraints_text"):
            prefix = config["constraints_text"]

    frame_text = FRAMES[frame_name]
    con_text = prefix[len(frame_text):].strip() if frame_text else prefix

    new = dict(config)
    new["frame"] = {"name": frame_name, "text": frame_text}
    new["constraints"] = constraints
    new["constraints_text"] = con_text
    new["constraints_id"] = constraints_id(constraints)
    new["prompt_prefix"] = prefix
    # Deliberately NOT setting collection_script/git_commit here: these runs were made by
    # older scripts at commits we no longer know. Only new runs record their provenance.
    if note:
        new["note"] = note

    # Sanity: the text we would generate today must match what was actually sent.
    if not note and constraints and con_text != constraints_text(constraints):
        raise ValueError(f"{exp_set}/{run_id}: regenerated text {constraints_text(constraints)!r} "
                         f"!= historical {con_text!r}")
    return new


def main(apply):
    changed = skipped = 0
    for exp_set in sorted(os.listdir(DATA_ROOT)):
        set_dir = os.path.join(DATA_ROOT, exp_set)
        if not os.path.isdir(set_dir):
            continue
        for run_id in sorted(os.listdir(set_dir)):
            path = os.path.join(set_dir, run_id, "config.json")
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                config = json.load(f)
            new = migrate_one(exp_set, run_id, config)
            if new is None:
                skipped += 1
                continue
            changed += 1
            print(f"{exp_set}/{run_id}: {config.get('constraints')!r} -> "
                  f"{new['constraints_id']}  frame={new['frame']['name']}"
                  f"{'  [SPECIAL]' if 'note' in new else ''}")
            print(f"    prompt_prefix: {new['prompt_prefix']!r}")
            if apply:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(new, f, indent=2, sort_keys=True)
    print(f"\n{'wrote' if apply else 'would change'}: {changed}   already migrated: {skipped}")
    if not apply:
        print("dry run - pass --apply to write")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="write the files")
    main(p.parse_args().apply)
