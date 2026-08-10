
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from src.plot_utils import better_color_map


def plot_convergence(df_weights,
                     title="BT Convergence",
                     color_map=None,
                     IQR=False,
                     start_iteration=1):
    """
    :param df_weights: index is 'iteration', columns are alternatives, values are BT probabilities
    """
    df_weights = df_weights.copy()[df_weights.index >= start_iteration]
    
    plt.figure(figsize=(15, 5))
    columns = [c for c in df_weights.columns if c not in ['iteration', 'num_templates']]
    if color_map:
        assert set(columns).issubset(set(color_map.keys())) \
            , f"Data columns do not match color map keys.\nColumns:{columns}\nColor Map Keys: {color_map.keys()}"
    for alternative in columns:
        if color_map:
            color = color_map[alternative]
            plt.plot(df_weights.index, df_weights[alternative], label=alternative, 
                color=better_color_map.get(color, color), linewidth=2)
        else:
            plt.plot(df_weights.index, df_weights[alternative], label=alternative, linewidth=2)
    plt.legend()
    plt.xlabel("Iteration (Templates)")
    plt.ylabel("BT Probability ($e^\\beta$)")
    if IQR:
        all_values = df_weights.values.flatten()
        q1 = np.percentile(all_values, 25)
        q3 = np.percentile(all_values, 75)
        iqr = q3 - q1
        plt.ylim(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_final_weights(df_weights_list,
                       title="BT Probabilities",
                       color_map=None,
                    ):
   all_final_probs = []

   for df in df_weights_list:
      last_iter = df['iteration'].max()
      final_df = df[df['iteration'] == last_iter].copy()
      final_df['prob'] = np.exp(final_df['BT_Score'])
      all_final_probs.append(final_df[['Item', 'prob']])

   summary_df = (
      pd.concat(all_final_probs)
      .groupby('Item')['prob'].agg(['mean', 'std'])
      .reset_index()
      .sort_values('mean', ascending=False)
   )

   plt.figure(figsize=(14, 6))
   items = summary_df['Item']
   means = summary_df['mean']
   stds = summary_df['std'].fillna(0)
   
   x_pos = np.arange(len(items))
   # Handle colors
   colors = 'skyblue'
   if color_map:
      assert set(items).issubset(set(color_map.keys())), \
         f"All items must have a corresponding color in the color_map.\nItems: {items}\nColor Map: {color_map.keys()}"
      colors = [better_color_map[color_map.get(item, 'grey')] for item in items]
      plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10, color=colors)
   else:
      plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10)
   
   plt.xticks(x_pos, items, rotation=45, ha='right')
   plt.ylabel('BT Probability ($e^{\\beta}$)')
   plt.title(title)
   plt.tight_layout()
   plt.grid(axis='y', linestyle='--', alpha=0.7)
   plt.show()

# ============================================================================
# Laptops robustness experiment plots
# ============================================================================

def plot_preference_shift(df: pd.DataFrame, 
                          feature_col: str = 'Brand',
                          w_base_col: str = 'Weight_32_Base', 
                          p_base_col: str = 'Pval_32_Base',
                          w_constrained_col: str = 'Weight_32_14', 
                          p_constrained_col: str = 'Pval_32_14'):
    """
    Plots the shift in LLM latent utility weights between an unconstrained 
    and constrained condition. Automatically fades bars that are not 
    statistically significant (p > 0.05).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Setup X-axis positions
    x = np.arange(len(df))
    width = 0.35  
    
    # Define primary colors
    color_base = '#1f77b4'      # Deep Blue
    color_constrained = '#ff7f0e' # Bright Orange
    
    # Significance threshold
    alpha_sig = 1.0  # Solid color if significant
    alpha_insig = 0.25 # Faded color if mostly noise (p > 0.05)

    # Plot each bar individually so we can apply specific opacity based on p-value
    for i in range(len(df)):
        # Base Condition (32-)
        alpha1 = alpha_sig if df.loc[i, p_base_col] <= 0.05 else alpha_insig
        ax.bar(x[i] - width/2, df.loc[i, w_base_col], width, 
               color=color_base, alpha=alpha1, edgecolor='black', linewidth=0.5)
        
        # Constrained Condition (32-14)
        alpha2 = alpha_sig if df.loc[i, p_constrained_col] <= 0.05 else alpha_insig
        ax.bar(x[i] + width/2, df.loc[i, w_constrained_col], width, 
               color=color_constrained, alpha=alpha2, edgecolor='black', linewidth=0.5)

    # Add a horizontal zero-line for the baseline reference
    ax.axhline(0, color='black', linewidth=1.2, linestyle='-')

    # Formatting and Labels
    ax.set_ylabel('Latent Utility (Log-Odds Weight)', fontsize=12, fontweight='bold')
    ax.set_title('Preference Flattening: Impact of "14-inch" Constraint on Brand Bias', 
                 fontsize=14, pad=15)
    
    ax.set_xticks(x)
    ax.set_xticklabels(df[feature_col], rotation=45, ha='right', fontsize=11)
    
    # Custom Legend
    legend_elements = [
        Patch(facecolor=color_base, label='Unconstrained (32GB RAM)', alpha=alpha_sig),
        Patch(facecolor=color_constrained, label='Constrained (+14-inch Screen)', alpha=alpha_sig),
        Patch(facecolor='gray', label='Not Significant (p > 0.05)', alpha=alpha_insig)
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.show()


def plot_laptop_ranking_shift(scores_by_run, run_order, constraints_by_run,
                               constraint_markers=None, figsize=None, label_fontsize=7):
    """
    Bump chart of per-laptop preference RANK across experiment conditions.
    Each laptop (a full brand-screen-ram item) is one line; x = run key, y = rank
    (1 = most preferred, at the top).

    Color/linestyle are computed PER SEGMENT (run_a -> run_b), not once per
    laptop, since which constraints are "previous" vs "new" changes from one
    shift to the next: `prev` is the full set of constraint tokens already
    active at run_a (vacuously satisfied by everyone if run_a is
    unconstrained), and `new` is whatever token(s) run_b adds on top of that.
    A laptop is classified by whether it matches ALL of `prev` and ALL of
    `new`:
      - matches prev and new     -> solid, bold deep pink  (adheres to all)
      - matches prev, not new    -> solid, light pink       (adheres to previous only)
      - matches new, not prev    -> dashed, deep pink       (adheres to new only)
      - matches neither          -> solid, muted gray

    constraints_by_run: {run_key: raw constraints string} as returned by
        load_scores_by_run, e.g. {"32-": "", "32-14": "14", "32-14_8": "14_8"}.
    constraint_markers: {constraint token: substring to look for in the item
        label} - defaults to {"14": "14-inch", "8": "8GB"}. A laptop
        "adheres" to a token if that substring is in its item label.

    Every laptop is labeled at its rank row on both sides, colored to match
    its first/last segment respectively. Works for any number of experiments
    in run_order (>= 2), and run_order doesn't need to start from an
    unconstrained baseline run.

    scores_by_run: {run_key: {item_name: score}}
    """
    if constraint_markers is None:
        constraint_markers = {"14": "14-inch", "8": "8GB"}

    # Only laptops present in every run can be drawn as a connected line
    common_items = set(scores_by_run[run_order[0]])
    for rk in run_order[1:]:
        common_items &= set(scores_by_run[rk])
    items = sorted(common_items)
    n = len(items)

    # Rank within each run: 1 = highest utility (most preferred)
    ranks_by_run = {}
    for rk in run_order:
        s = pd.Series({item: scores_by_run[rk][item] for item in items})
        ranks_by_run[rk] = s.rank(ascending=False, method="first").astype(int)

    x = np.arange(len(run_order))

    color_all = "#d6336c"        # adheres to all (previous + new)
    color_prev_only = "#ff8fab"  # adheres to previous only (lighter pink)
    color_new_only = "#9d174d"   # adheres to new only (deep pink, dashed)
    color_none = "#adb5bd"       # adheres to neither (muted gray)

    def tokens_of(run_key):
        return set(t for t in constraints_by_run[run_key].split("_") if t)

    def matches_all(item, tokens):
        return all(constraint_markers[t] in item for t in tokens)

    def style_for_segment(item, run_a, run_b):
        prev_tokens = tokens_of(run_a)
        new_tokens = tokens_of(run_b) - prev_tokens
        adheres_prev = matches_all(item, prev_tokens)
        adheres_new = matches_all(item, new_tokens)
        if adheres_prev and adheres_new:
            return dict(color=color_all, linestyle="-", linewidth=1.8, alpha=0.95, zorder=4)
        if adheres_prev:
            return dict(color=color_prev_only, linestyle="-", linewidth=1.4, alpha=0.9, zorder=3)
        if adheres_new:
            return dict(color=color_new_only, linestyle="--", linewidth=1.4, alpha=0.9, zorder=3)
        return dict(color=color_none, linestyle="-", linewidth=1.0, alpha=0.5, zorder=2)

    if figsize is None:
        figsize = (4 + 3 * len(run_order), max(6, n * 0.32))

    fig, ax = plt.subplots(figsize=figsize)

    n_segments = len(run_order) - 1
    for item in items:
        y = [ranks_by_run[rk][item] for rk in run_order]
        for seg_idx in range(n_segments):
            style = style_for_segment(item, run_order[seg_idx], run_order[seg_idx + 1])
            ax.plot(x[seg_idx:seg_idx + 2], y[seg_idx:seg_idx + 2], marker="o", markersize=4,
                    markeredgecolor="white", markeredgewidth=0.4, **style)

        # Row labels aligned to this laptop's rank at each end, colored to match
        # the segment touching that end - no overlap since rank is a unique
        # integer y-position per line
        left_color = style_for_segment(item, run_order[0], run_order[1])["color"]
        right_color = style_for_segment(item, run_order[-2], run_order[-1])["color"]
        ax.text(x[0] - 0.05, y[0], item, ha="right", va="center",
                fontsize=label_fontsize, color=left_color)
        ax.text(x[-1] + 0.05, y[-1], item, ha="left", va="center",
                fontsize=label_fontsize, color=right_color)

    ax.set_ylim(n + 1, 0)  # rank 1 at the top
    ax.set_yticks([])
    ax.set_xticks(x)
    ax.set_xticklabels(run_order, fontsize=11, fontweight="bold")
    ax.set_xlim(x[0] - 1.1, x[-1] + 1.1)  # room for the row labels on both sides
    ax.set_title("Laptop Preference Ranking Across Experiments", fontsize=14, fontweight="bold", pad=44)

    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_handles = [
        Line2D([0], [0], color=color_all, linewidth=2, linestyle="-",
               label="Adheres to all constraints"),
        Line2D([0], [0], color=color_prev_only, linewidth=2, linestyle="-",
               label="Adheres to previous only"),
        Line2D([0], [0], color=color_new_only, linewidth=2, linestyle="--",
               label="Adheres to new only"),
        Line2D([0], [0], color=color_none, linewidth=2, linestyle="-",
               label="Adheres to neither"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncol=4, frameon=False, title="Constraint adherence (per shift)", fontsize=9)

    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_feature_weight_trajectories(feature_results_by_run, run_order, p_threshold=0.05, figsize=None):
    """
    Small-multiples line plot: one subplot per feature dimension (brand,
    screen, ram, ...), with one line per feature level tracking its own
    Bradley-Terry weight across run_order. Markers are faded when that
    point's weight is not statistically significant (p > p_threshold) - so
    it's easy to see whether a feature's own preference is genuinely
    drifting across the constraint shifts, independent of the item-level
    rank churn that the constraint's main effect mechanically causes.

    feature_results_by_run: {run_key: fit_feature_based_bradley_terry(...) output},
        e.g. from load_scores_by_run(..., fit_fn=fit_feature_based_bradley_terry).
    """
    all_features = sorted({feat for res in feature_results_by_run.values() for feat in res["feature_scores"]})
    groups = {}
    for feat in all_features:
        group = feat.split("_", 1)[0]
        groups.setdefault(group, []).append(feat)

    group_names = sorted(groups)
    if figsize is None:
        figsize = (5 * len(group_names), 4)
    fig, axes = plt.subplots(1, len(group_names), figsize=figsize, sharey=True)
    if len(group_names) == 1:
        axes = [axes]

    x = np.arange(len(run_order))
    cmap = plt.get_cmap("tab10")

    for ax, group in zip(axes, group_names):
        for i, feat in enumerate(sorted(groups[group])):
            weights = [feature_results_by_run[rk]["feature_scores"].get(feat, 0.0) for rk in run_order]
            pvalues = [feature_results_by_run[rk]["feature_pvalues"].get(feat, 0.0) for rk in run_order]
            color = cmap(i % 10)
            ax.plot(x, weights, color=color, linewidth=1.6, label=feat.split("_", 1)[1], zorder=2)
            for xi, w, p in zip(x, weights, pvalues):
                alpha = 1.0 if p <= p_threshold else 0.3
                ax.scatter([xi], [w], color=color, alpha=alpha, s=40, zorder=3,
                           edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
        ax.set_title(group, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(run_order, rotation=30, ha="right")
        ax.legend(fontsize=8, loc="best", framealpha=0.9)

    axes[0].set_ylabel("Feature weight (log-odds)", fontsize=11)
    fig.suptitle("Per-feature weight trajectories across the constraint shift", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return fig, axes


def _diverging_attribution_barh(df_subset, base_features, title, colors, ax):
    """
    Draws one horizontal diverging bar per row of df_subset: positive
    per-feature deltas stack rightward from 0, negative ones stack leftward
    from 0 (SHAP-style), colored by feature dimension. A black tick marks
    the net delta_total, so it's visually obvious when a laptop's rank barely
    moved despite large, offsetting per-feature pulls.
    """
    n = len(df_subset)
    y_positions = np.arange(n)[::-1]

    for y, (_, row) in zip(y_positions, df_subset.iterrows()):
        pos_left, neg_right = 0.0, 0.0
        for feat in base_features:
            d = row[f"delta_{feat}"]
            color = colors[feat]
            if d >= 0:
                ax.barh(y, d, left=pos_left, height=0.6, color=color, edgecolor="white", linewidth=0.6, zorder=2)
                pos_left += d
            else:
                ax.barh(y, d, left=neg_right, height=0.6, color=color, edgecolor="white", linewidth=0.6, zorder=2)
                neg_right += d
        ax.plot(row["delta_total"], y, marker="|", color="black", markersize=14, markeredgewidth=2, zorder=5)

    ax.axvline(0, color="black", linewidth=1.0, zorder=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_subset["item"], fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Utility delta (log-odds)", fontsize=9)


def plot_shift_attribution(decomposition_df, base_features, run_a=None, run_b=None,
                            n_top=8, n_tug_of_war=6, n_feature_driven=6,
                            highlight_feature="brand", figsize=None):
    """
    Visualizes decompose_item_shift's output for a curated set of
    "interesting" laptops instead of the full item set (which, at N items,
    is unreadable and mostly just laptops passively following the
    constraint's main effect). Three panels, side by side:

      - "Biggest movers": the n_top items with the largest |delta_total| -
        the most dramatic rank changes in the bump chart.
      - "Tug-of-war": items NOT already in the biggest-movers panel where
        the per-feature deltas are large but largely CANCEL OUT (e.g. brand
        bias pushes one way, ram pushes the other), leaving a small net
        delta_total. A rank-only view hides these completely since the
        laptop barely moved - but its preference composition still shifted
        underneath.
      - "Highest <highlight_feature> weight": items NOT already shown above,
        ranked purely by |delta_<highlight_feature>| (default "brand") -
        screen/ram are ignored entirely for this ranking, so it directly
        answers "which laptops did brand contribute to the most" without
        that ranking being influenced by how much the laptop's other
        features happened to move.

    Each bar still shows the full picture though: positive feature
    contributions stack rightward from 0 and negative ones stack leftward
    from 0, colored by feature dimension, with a black tick marking the net
    delta_total - so you see how the highlighted feature's effect combines
    with (or is offset by) the laptop's other features.
    """
    delta_cols = [f"delta_{feat}" for feat in base_features]

    top_movers = decomposition_df.reindex(
        decomposition_df["delta_total"].abs().sort_values(ascending=False).index
    ).head(n_top).reset_index(drop=True)

    gross_movement = decomposition_df[delta_cols].abs().sum(axis=1)
    net_movement = decomposition_df["delta_total"].abs()
    disagreement = gross_movement - net_movement

    remaining = decomposition_df.loc[~decomposition_df["item"].isin(top_movers["item"])]
    tug_of_war = remaining.reindex(
        disagreement.loc[remaining.index].sort_values(ascending=False).index
    ).head(n_tug_of_war).reset_index(drop=True)

    already_selected = set(top_movers["item"]) | set(tug_of_war["item"])
    remaining_for_feature = decomposition_df.loc[~decomposition_df["item"].isin(already_selected)]
    highlight_col = f"delta_{highlight_feature}"
    feature_driven = remaining_for_feature.reindex(
        remaining_for_feature[highlight_col].abs().sort_values(ascending=False).index
    ).head(n_feature_driven).reset_index(drop=True)

    colors = dict(zip(base_features, plt.get_cmap("tab10").colors))

    n_rows = max(len(top_movers), len(tug_of_war), len(feature_driven), 1)
    if figsize is None:
        figsize = (19, 0.5 * n_rows + 2)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    _diverging_attribution_barh(top_movers, base_features, "Biggest movers (largest net shift)", colors, axes[0])
    _diverging_attribution_barh(tug_of_war, base_features,
                                 "Tug-of-war (large offsetting pulls, small net shift)", colors, axes[1])
    _diverging_attribution_barh(feature_driven, base_features,
                                 f"Highest {highlight_feature} weight (ranked by {highlight_feature} only)",
                                 colors, axes[2])

    x_lim = max(abs(v) for ax in axes for v in ax.get_xlim())
    for ax in axes:
        ax.set_xlim(-x_lim, x_lim)

    legend_handles = [Patch(facecolor=colors[feat], label=feat) for feat in base_features]
    legend_handles.append(Line2D([0], [0], marker="|", color="black", linestyle="",
                                  markersize=12, markeredgewidth=2, label="net delta"))
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    title = "What drove each laptop's utility shift, decomposed by feature"
    if run_a and run_b:
        title += f": {run_a} -> {run_b}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()
    return fig, axes


def plot_laptop_feature_weights(feature_results_by_run, run_order, item_feature_map, base_features,
                                 items, figsize=None):
    """
    For a curated set of specific laptops, draws one stacked-bar-chart
    subplot each: x = run_order, and each bar is that laptop's own utility
    in that run, stacked from its feature WEIGHTS (not deltas) - so unlike
    plot_shift_attribution (which shows the CHANGE between a single pair of
    runs), this shows the full trajectory of absolute feature contributions
    across every run in run_order at once.

    Positive segments stack upward from 0, negative ones stack downward
    (SHAP-style), colored by feature dimension (same colors as
    plot_shift_attribution); a black diamond+line traces the laptop's true
    total utility (the sum, not just the visual top of the stack, since some
    segments can be negative) across runs.

    items: list of item labels to plot (one subplot each), e.g. hand-picked
        or reused from decompose_item_shift's "biggest movers" /
        "tug-of-war" / highlight-feature selections.
    item_feature_map: {item_label: {feature: value}} as returned by
        get_item_feature_map.
    """
    colors = dict(zip(base_features, plt.get_cmap("tab10").colors))
    n = len(items)
    ncols = min(n, 4)
    nrows = -(-n // ncols)  # ceil division
    if figsize is None:
        figsize = (4.5 * ncols, 4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    x = np.arange(len(run_order))
    for ax, item in zip(axes_flat, items):
        feats = item_feature_map[item]
        totals = []
        for xi, rk in zip(x, run_order):
            scores = feature_results_by_run[rk]["feature_scores"]
            pos_bottom, neg_top = 0.0, 0.0
            total = 0.0
            for feat in base_features:
                w = scores.get(f"{feat}_{feats[feat]}", 0.0)
                total += w
                color = colors[feat]
                if w >= 0:
                    ax.bar(xi, w, bottom=pos_bottom, width=0.6, color=color,
                           edgecolor="white", linewidth=0.6, zorder=2)
                    pos_bottom += w
                else:
                    ax.bar(xi, w, bottom=neg_top, width=0.6, color=color,
                           edgecolor="white", linewidth=0.6, zorder=2)
                    neg_top += w
            totals.append(total)
        ax.plot(x, totals, color="black", marker="D", markersize=5, linewidth=1.2, zorder=5)
        ax.axhline(0, color="black", linewidth=1.0, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(run_order, rotation=30, ha="right", fontsize=8)
        ax.set_title(item, fontsize=10, fontweight="bold")
        ax.set_ylabel("Utility (log-odds)", fontsize=8)

    for ax in axes_flat[n:]:
        ax.axis("off")

    legend_handles = [Patch(facecolor=colors[feat], label=feat) for feat in base_features]
    legend_handles.append(Line2D([0], [0], color="black", marker="D", markersize=5, label="total utility"))
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Per-feature weight composition across experiments", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.show()
    return fig, axes
