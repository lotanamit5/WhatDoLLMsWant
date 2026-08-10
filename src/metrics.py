import pandas as pd
from scipy.stats import kendalltau, spearmanr

def calculate_kendall_distance(ranking_a, ranking_b):
    """
    Calculates the Normalized Kendall Tau Distance between two lists of items.
    Returns a value between 0.0 (identical) and 1.0 (completely reversed).
    """
    # Ensure both lists have the same items
    assert set(ranking_a) == set(ranking_b), "Rankings must contain the same items"

    # Calculate Tau correlation (-1 to 1)
    tau, _ = kendalltau(ranking_a, ranking_b)

    # Convert correlation to normalized distance
    # Distance = (1 - tau) / 2
    # If tau = 1 (identical), distance = 0
    # If tau = -1 (reversed), distance = 1
    return (1 - tau) / 2

# ============================================================================
# Comparing two runs of the same experiment
# ============================================================================

def compute_adherence_rank_correlations(scores_by_run, run_order, screen_marker_value="14-inch"):
    """
    For each consecutive pair of experiments in run_order, computes the Spearman
    rank correlation of laptop utility separately within the laptops that adhere
    to `screen_marker_value` and within the ones that don't - i.e. whether each
    group's own internal relative ordering (driven by brand/ram) survives the
    shift, independent of the constraint's main effect. Automatically covers
    every consecutive shift, so adding a third (or later) experiment to run_order
    adds its shift's correlations without any other changes.

    Returns a list of per-shift result dicts; also prints a readable summary.
    """
    results = []
    for run_a, run_b in zip(run_order, run_order[1:]):
        common_items = sorted(set(scores_by_run[run_a]) & set(scores_by_run[run_b]))
        adhering = [item for item in common_items if screen_marker_value in item]
        violating = [item for item in common_items if screen_marker_value not in item]

        def group_corr(items):
            if len(items) < 2:
                return float("nan"), len(items)
            a = [scores_by_run[run_a][item] for item in items]
            b = [scores_by_run[run_b][item] for item in items]
            rho, _ = spearmanr(a, b)
            return rho, len(items)

        adhere_rho, n_adhere = group_corr(adhering)
        violate_rho, n_violate = group_corr(violating)

        results.append({
            "shift": f"{run_a} -> {run_b}",
            "adhere_rho": adhere_rho, "n_adhere": n_adhere,
            "violate_rho": violate_rho, "n_violate": n_violate,
        })

        print(f"Shift {run_a} -> {run_b}:")
        print(f"  Adhering  ({screen_marker_value}, n={n_adhere}): rho = {adhere_rho:.3f}")
        print(f"  Violating (not {screen_marker_value}, n={n_violate}): rho = {violate_rho:.3f}")

    return results


def decompose_item_shift(feature_results_by_run, run_a, run_b, item_feature_map, base_features):
    """
    Decomposes each item's utility delta between run_a and run_b into its
    per-feature-dimension contributions. Exact, not approximate: since item
    utility is the sum of its feature weights, delta_total here always equals
    the same item's utility delta from fit_item_bradley_terry.

    feature_results_by_run: {run_key: fit_feature_based_bradley_terry(...) output}.

    Returns a DataFrame with one row per item (delta_<feature>, delta_total),
    sorted by |delta_total| descending - i.e. the biggest movers first.
    """
    scores_a = feature_results_by_run[run_a]["feature_scores"]
    scores_b = feature_results_by_run[run_b]["feature_scores"]

    records = []
    for item, feats in item_feature_map.items():
        per_feature_delta = {
            feat: scores_b.get(f"{feat}_{feats[feat]}", 0.0) - scores_a.get(f"{feat}_{feats[feat]}", 0.0)
            for feat in base_features
        }
        record = {"item": item, **{f"delta_{feat}": d for feat, d in per_feature_delta.items()}}
        record["delta_total"] = sum(per_feature_delta.values())
        records.append(record)

    return pd.DataFrame(records).sort_values(
        "delta_total", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)
