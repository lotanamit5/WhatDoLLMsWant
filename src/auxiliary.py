import pandas as pd
import numpy as np
import os
import json

from src.pref_models import fit_bradley_terry, fit_item_bradley_terry

def analyze_bt_convergence(exp_name):
    scores_path = f"experiments/{exp_name}/scores.csv"
    if not os.path.exists(scores_path):
        print(f"File not found: {scores_path}")
        return

    df_scores = pd.read_csv(scores_path)

    # In case the file is empty or malformed
    if df_scores.empty:
        print("Scores file is empty.")
        return

    templates = df_scores['template'].unique()
    methods = [
        ('single', 'score_single_a', 'score_single_b'),
        ('group', 'score_group_a', 'score_group_b'),
        ('ppl', 'score_ppl_a', 'score_ppl_b')
    ]

    # Get all unique items from the dataset
    items = np.unique(np.concatenate([df_scores['item_a'].unique(), df_scores['item_b'].unique()]))

    all_results = []

    for method_name, col_a, col_b in methods:
        print(f"Processing method: {method_name}")
        for i in range(1, len(templates) + 1):
            # Select templates 0 to i-1
            current_templates = templates[:i]
            
            # Filter data
            df_subset = df_scores[df_scores['template'].isin(current_templates)].copy()
            
            # Determine winner (Higher score wins)
            # Handle potential tie or same scores? Usually > is fine, or >=. Previous code used >=.
            df_subset['winner'] = np.where(df_subset[col_a] >= df_subset[col_b], df_subset['item_a'], df_subset['item_b'])
            
            # Fit BT
            ranking, _ = fit_bradley_terry(df_subset, items)
            
            # Collect results
            for _, row in ranking.iterrows():
                all_results.append({
                    'method': method_name,
                    'iteration': i,
                    'num_templates': i,
                    'item': row['Color'],
                    'score': row['BT_Score']
                })

    df_results = pd.DataFrame(all_results)
    output_path = f"experiments/{exp_name}/bt_convergence.csv"
    df_results.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
    
    return df_results

def load_experiment_rankings(exp_name):
    """
    Loads all method rankings from an experiment folder into a single DataFrame.
    """
    methods = ['single', 'group', 'ppl']
    dfs = []
    
    for method in methods:
        path = f"experiments/{exp_name}/rankings_{method}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['method'] = method
            dfs.append(df)
        else:
            print(f"Warning: {path} not found.")
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)



# ============================================================================
# Loading runs from data/<exp_name>/<slurm_job_id>/
# ============================================================================

def load_scores_by_run(basedir, filters, constraint_order, fit_fn=fit_item_bradley_terry):
    """
    Loads and fits one Bradley-Terry model per experiment run under `basedir`.

    filters: dict of config.json keys that must match exactly to select a
        family of runs, e.g. {"model_size": "7"} or
        {"model_family": "gemma", "model_size": "27"}.
    constraint_order: canonical list of `constraints_id` values defining the run
        order, e.g. ["none", "screen=14-inch", "ram=8GB+screen=14-inch"].

    Returns (scores_by_run, run_order, constraints_by_run). scores_by_run is
    keyed by f"{'-'.join(filters.values())}-{constraints}" (matching the
    original CANONICAL_RUN_ORDER naming, e.g. "7-14"); run_order is
    constraint_order filtered down to the runs actually found on disk, in
    canonical order - so the plot/correlations grow automatically as more
    experiments land. constraints_by_run maps each run_key back to its raw
    constraints string (e.g. "7-14_8" -> "14_8"), for callers (like
    plot_laptop_ranking_shift) that need to know which constraints are
    actually active in each run.
    """
    prefix = "-".join(str(v) for v in filters.values())
    scores_by_run = {}
    constraints_by_run = {}

    for filename in os.listdir(basedir):
        config_path = os.path.join(basedir, filename, "config.json")
        scores_path = os.path.join(basedir, filename, "scores.csv")
        if not os.path.isfile(config_path) or not os.path.isfile(scores_path):
            continue
        with open(config_path, "r") as f:
            config = json.load(f)

        if any(config.get(key, "") != value for key, value in filters.items()):
            continue

        # Since the 2026-08-10 schema, `constraints` is a dict {feature: level};
        # `constraints_id` is its unambiguous slug ("none", "screen=14-inch", ...).
        constraints = config.get("constraints_id", "") or ""
        if constraints not in constraint_order:
            continue

        run_key = f"{prefix}-{constraints}"
        print(f"Loaded {filename} -> {run_key}")
        df = pd.read_csv(scores_path)
        scores_by_run[run_key] = fit_fn(df)
        constraints_by_run[run_key] = constraints

    run_order = [f"{prefix}-{c}" for c in constraint_order if f"{prefix}-{c}" in scores_by_run]
    missing = [f"{prefix}-{c}" for c in constraint_order if f"{prefix}-{c}" not in scores_by_run]
    if missing:
        print(f"Note: still missing {missing} (job may still be running) - rerun this cell once it lands.")

    return scores_by_run, run_order, constraints_by_run


def find_any_scores_csv(basedir, filters):
    """
    Returns the scores.csv path of the first run under `basedir` whose
    config.json matches `filters` - used to build an item_feature_map
    without caring which specific run it comes from, since the item
    universe is identical across every run of a given `alternatives` set.
    """
    for filename in os.listdir(basedir):
        config_path = os.path.join(basedir, filename, "config.json")
        scores_path = os.path.join(basedir, filename, "scores.csv")
        if not os.path.isfile(config_path) or not os.path.isfile(scores_path):
            continue
        with open(config_path, "r") as f:
            config = json.load(f)
        if all(config.get(key, "") == value for key, value in filters.items()):
            return scores_path
    raise FileNotFoundError(f"No run under {basedir} matches {filters}")


def get_item_feature_map(scores_csv_path, a_prefix="a_", b_prefix="b_"):
    """
    Maps every unique item (laptop) seen in `scores_csv_path` to its raw
    feature values, e.g. "ASUS-13-inch-4GB" -> {"brand": "ASUS", "screen":
    "13-inch", "ram": "4GB"}. The item universe is the same across every run
    of a given `alternatives` set, so this only needs to be built once (from
    any one run's scores.csv) and reused for every shift.

    Returns (item_feature_map, base_features).
    """
    df = pd.read_csv(scores_csv_path)
    cols_a = [col for col in df.columns if col.startswith(a_prefix)]
    cols_b = [col.replace(a_prefix, b_prefix) for col in cols_a]
    base_features = [col[len(a_prefix):] for col in cols_a]

    df_a = df[cols_a].rename(columns=dict(zip(cols_a, base_features)))
    df_b = df[cols_b].rename(columns=dict(zip(cols_b, base_features)))
    unique_items = pd.concat([df_a, df_b], axis=0).drop_duplicates()

    item_feature_map = {}
    for _, row in unique_items.iterrows():
        item_label = "-".join(str(row[feat]) for feat in base_features)
        item_feature_map[item_label] = {feat: row[feat] for feat in base_features}

    return item_feature_map, base_features
