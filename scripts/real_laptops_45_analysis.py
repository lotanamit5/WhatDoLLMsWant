"""Real laptops 45 (2026-11-18): base preferences, no contract, name shown (A) vs name hidden (B). Stage 3.

Standalone. Reads data/, writes figs/real_laptops_45/. Does not touch src/, data/ or any other script.
Run from the repo root:  ~/miniconda3/envs/whatdo-llms-want/bin/python scripts/real_laptops_45_analysis.py

Runs: data/real_laptops_45_named (A) and data/real_laptops_45_hidden (B), qwen-72 / gemma-27 / olmo-32, and for
step G the synthetic laptop grid with no contract (frame shopping, 5 templates): the run listed in SYNTH.

Y = score_a - score_b + C (C = the run's PMI constant, as in scripts/residual_noise.py; 0 for the new runs).
Cell = ordered pair (a in slot A, b in slot B), T = 5 rows. All fits OLS on Y with an intercept gamma:
  BT-feat   gamma + (z_a - z_b) . w     one-hot features (drop-first, then centred inside each feature)
                                        + storage size (log2 total GB, standardized over the 45 laptops)
  BT-full   gamma + u_a - u_b           = src/bt.py FeatureBT on the item id (used as is for scores / R2 / gamma)
  saturated one free mean per cell
F-tests: statsmodels compare_f_test (denominator = the larger model's residual). Bootstrap for BT-full vs
saturated: BT-full fitted cell value + the cell's own template deviations (Y_ct - Ybar_c) * sqrt(T/(T-1)),
resampled with replacement inside the cell; refit both; F. B = 1000, fixed seed.
Cell R2: 1 - sum_c (Ybar_c - Yhat_c)^2 / sum_c (Ybar_c - mean Ybar)^2 (balanced cells: the row fit = the cell fit).
var_e, var_m, rho: as in scripts/residual_noise.py. Cluster SEs: statsmodels cluster (CR1), cluster = unordered pair.

Two storage cases the rule in the request does not name (one line each in storage_type below):
  L26 "64GB Flash Storage + 1TB HDD" -> SSD + HDD     L31 "32GB SSD" -> SSD (the file says SSD, not Flash Storage)
Numbers only.
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from src.bt import FeatureBT  # noqa: E402

OUT = "figs/real_laptops_45"
os.makedirs(OUT, exist_ok=True)
MODELS = ["qwen-72", "gemma-27", "olmo-32"]
EXPS = {"A": "data/real_laptops_45_named", "B": "data/real_laptops_45_hidden"}
SYNTH = {"qwen-72": "data/laptops_robustness/1271581", "gemma-27": "data/laptops_robustness_gemma/68378741",
         "olmo-32": "data/laptops_olmo/1319097"}
N_ROWS, T, B_BOOT, SEED = 9900, 5, 1000, 20261118
BRANDS = ["ASUS", "Apple", "Dell", "HP", "Lenovo"]
BRAND_COLOR = {"ASUS": "#4C72B0", "Apple": "#7F7F7F", "Dell": "#DD8452", "HP": "#55A868", "Lenovo": "#C44E52"}

# ------------------------------------------------------------------ items and features
ITEMS = pd.read_csv("data/real_laptops_45/design/real_laptops_45.csv")


def screen_cat(x):
    return {11.6: "11.6-12", 12.0: "11.6-12", 13.3: "13.3", 14.0: "14", 15.4: "15.4-15.6", 15.6: "15.4-15.6",
            17.3: "17.3"}[x]


def gb(part):
    n, unit = re.match(r"([\d.]+)(GB|TB)", part.strip()).groups()
    return float(n) * (1024 if unit == "TB" else 1)


def storage_type(s):
    parts = [p.strip() for p in s.split("+")]
    kinds = []
    for p in parts:
        if "HDD" in p:
            kinds.append("HDD")
        elif "Flash Storage" in p and gb(p) < 128:
            kinds.append("small flash")
        else:                                   # SSD, or Flash Storage of 128GB or more
            kinds.append("SSD")
    if kinds == ["small flash", "HDD"]:         # L26 "64GB Flash Storage + 1TB HDD"
        return "SSD + HDD"
    return {("SSD",): "SSD", ("HDD",): "HDD", ("SSD", "HDD"): "SSD + HDD", ("small flash",): "small flash"}[tuple(kinds)]


def processor_cat(p):
    for key, cat in [("Core i3", "Core i3"), ("Core i5", "Core i5"), ("Core i7", "Core i7"), ("Core M", "Core M"),
                     ("Celeron", "Celeron/Pentium"), ("Pentium", "Celeron/Pentium"), ("AMD", "AMD"), ("Xeon", "Xeon")]:
        if key in p:
            return cat
    raise ValueError(p)


ITEMS["f_brand"] = ITEMS.brand
ITEMS["f_screen"] = ITEMS.screen_inches.map(screen_cat)
ITEMS["f_ram"] = ITEMS.ram
ITEMS["f_storage_type"] = ITEMS.storage.map(storage_type)
ITEMS["f_processor"] = ITEMS.processor.map(processor_cat)
_l2 = np.log2(ITEMS.storage.map(lambda s: sum(gb(p) for p in s.split("+"))))
ITEMS["f_storage_size"] = (_l2 - _l2.mean()) / _l2.std(ddof=0)
ITEMS["total_gb"] = ITEMS.storage.map(lambda s: sum(gb(p) for p in s.split("+")))
LEVELS = {"brand": BRANDS, "screen": ["11.6-12", "13.3", "14", "15.4-15.6", "17.3"], "ram": ["4GB", "8GB", "16GB"],
          "storage_type": ["SSD", "HDD", "SSD + HDD", "small flash"],
          "processor": ["Core i3", "Core i5", "Core i7", "Core M", "Celeron/Pentium", "AMD", "Xeon"]}
SYNTH_LEVELS = {"brand": BRANDS, "screen": ["13-inch", "14-inch", "16-inch"], "ram": ["4GB", "8GB", "16GB"]}
ITEMS[["item_id", "brand", "model_name", "storage", "f_screen", "f_ram", "f_storage_type", "total_gb",
       "f_storage_size", "f_processor"]].to_csv(f"{OUT}/item_features.csv", index=False)


# ------------------------------------------------------------------ loading
def pmi_c(d, tol=0.01, min_sat=0.01):            # copied from scripts/residual_noise.py
    if max(d.score_a.max(), d.score_b.max()) <= 1e-9:
        return 0.0
    mA, mB = d.score_a.max(), d.score_b.max()
    sat = min((d.score_a > mA - tol).mean(), (d.score_b > mB - tol).mean())
    return mB - mA if sat >= min_sat else np.nan


def load_exp(folder):
    """One complete run per model, or crash. A run = config.json AND scores.csv with N_ROWS rows."""
    runs = {}
    for r in sorted(os.listdir(folder)):
        cf, sc = f"{folder}/{r}/config.json", f"{folder}/{r}/scores.csv"
        if not (os.path.isfile(cf) and os.path.isfile(sc)):
            continue
        c = json.load(open(cf))
        d = pd.read_csv(sc)
        if len(d) != N_ROWS:
            print(f"   skipped {folder}/{r}: {len(d)} rows")
            continue
        assert c["constraints_id"] == "none" and c["frame"]["name"] == "shopping", cf
        m = f"{c['model_family']}-{c['model_size']}"
        assert m not in runs, f"two complete runs of {m} in {folder}"
        runs[m] = (f"{folder}/{r}", d)
    return runs


# ------------------------------------------------------------------ fits
def onehot_diff(fa, fb, levels):
    """Drop-first dummies of one feature, a minus b. Returns matrix and the kept level names."""
    kept = levels[1:]
    return np.column_stack([(fa == l).astype(float) - (fb == l).astype(float) for l in kept]), kept


def analyse(path, d, kind, rng):
    """kind = 'real' (45 real laptops, z as in the request) or 'synth' (the brand/screen/ram grid)."""
    C = pmi_c(d)
    Y = (d.score_a - d.score_b).values + (0.0 if np.isnan(C) else C)
    N = len(Y)
    if kind == "real":
        ida, idb = d.a_id.values, d.b_id.values
    else:
        ida = (d.a_brand + "|" + d.a_screen + "|" + d.a_ram).values
        idb = (d.b_brand + "|" + d.b_screen + "|" + d.b_ram).values
    items = sorted(set(ida) | set(idb))
    ix = {it: i for i, it in enumerate(items)}
    n = len(items)
    ia, ib = np.array([ix[x] for x in ida]), np.array([ix[x] for x in idb])
    cell = np.unique(ia * n + ib, return_inverse=True)[1]
    nc = cell.max() + 1
    assert (np.bincount(cell) == T).all() and nc == n * (n - 1)
    lo, hi = np.minimum(ia, ib), np.maximum(ia, ib)
    pair = np.unique(lo * n + hi, return_inverse=True)[1]

    # BT-full through src/bt.py (the existing outputs), and the same design for the tests
    bt = FeatureBT(pmi_c=0.0, r2_warn=None).fit(pd.DataFrame({"score_a": Y, "score_b": 0.0, "a_id": ida, "b_id": idb}))
    Xf = np.zeros((N, n))
    Xf[:, 0] = 1
    rows = np.arange(N)
    Xf[rows[ia > 0], ia[ia > 0]] += 1
    Xf[rows[ib > 0], ib[ib > 0]] -= 1
    res_full = sm.OLS(Y, Xf).fit()
    assert np.isclose(res_full.ssr, bt.result_.ssr) and np.isclose(res_full.params[0], bt.beta0_)

    # BT-feat
    blocks, names = [np.ones((N, 1))], ["gamma"]
    feat_blocks = {}
    lev = LEVELS if kind == "real" else SYNTH_LEVELS
    if kind == "real":
        F = ITEMS.set_index("item_id")
        za, zb = F.loc[ida], F.loc[idb]
        for f in lev:
            M, kept = onehot_diff(za[f"f_{f}"].values, zb[f"f_{f}"].values, lev[f])
            feat_blocks[f] = (len(names), kept)
            blocks.append(M); names += [f"{f}={l}" for l in kept]
        blocks.append((za.f_storage_size.values - zb.f_storage_size.values)[:, None]); names.append("storage_size")
    else:
        for f in lev:
            M, kept = onehot_diff(d[f"a_{f}"].values, d[f"b_{f}"].values, lev[f])
            feat_blocks[f] = (len(names), kept)
            blocks.append(M); names += [f"{f}={l}" for l in kept]
    Xz = np.column_stack(blocks)
    assert np.linalg.matrix_rank(Xz) == Xz.shape[1], "BT-feat design is not full rank"
    res_feat = sm.OLS(Y, Xz).fit()
    res_feat_cl = sm.OLS(Y, Xz).fit(cov_type="cluster", cov_kwds={"groups": pair})

    # centred weights + cluster SEs: w_centred = Mc @ params
    K = Xz.shape[1]
    wrows = []
    for f, (start, kept) in feat_blocks.items():
        L = len(lev[f])
        for j, l in enumerate(lev[f]):
            v = np.zeros(K)
            if j > 0:
                v[start + j - 1] += 1
            v[start:start + L - 1] -= 1.0 / L
            w, se = v @ res_feat.params, np.sqrt(v @ res_feat_cl.cov_params() @ v)
            p_ref = res_feat.pvalues[start + j - 1] if j > 0 else np.nan
            wrows.append(dict(feature=f, level=l, weight=w, se_cluster=se, p_vs_reference_plain=p_ref,
                              reference_level=lev[f][0]))
    if kind == "real":
        k = names.index("storage_size")
        wrows.append(dict(feature="storage_size", level="per 1 SD of log2 GB", weight=res_feat.params[k],
                          se_cluster=res_feat_cl.bse[k], p_vs_reference_plain=res_feat.pvalues[k], reference_level=""))

    # saturated
    Xs = np.zeros((N, nc))
    Xs[rows, cell] = 1
    res_sat = sm.OLS(Y, Xs).fit()

    f1, p1, df1 = res_full.compare_f_test(res_feat)          # (i)  BT-feat vs BT-full
    f2, p2, df2 = res_sat.compare_f_test(res_full)           # (ii) BT-full vs saturated

    # bootstrap of (ii)
    Ybar = np.bincount(cell, Y) / T
    order = np.lexsort((d.template_idx.values, cell))        # rows grouped by cell
    Ymat = Y[order].reshape(nc, T)
    dev = (Ymat - Ybar[:, None]) * np.sqrt(T / (T - 1))
    Yhat_full = np.bincount(cell, res_full.fittedvalues) / T
    Q, _ = np.linalg.qr(Xf)
    Qs = Q[order].reshape(nc, T, n)
    dfn, dfd = nc - n, N - nc
    Fb = np.empty(B_BOOT)
    for b in range(B_BOOT):
        pick = rng.integers(0, T, size=(nc, T))
        Ys = Yhat_full[:, None] + np.take_along_axis(dev, pick, axis=1)
        rss_sat = ((Ys - Ys.mean(1, keepdims=True)) ** 2).sum()
        proj = np.einsum("ctk,ct->k", Qs, Ys)
        rss_full = (Ys ** 2).sum() - proj @ proj
        Fb[b] = ((rss_full - rss_sat) / dfn) / (rss_sat / dfd)
    p_boot = (Fb >= f2).mean()

    # cell R2
    def cell_r2(res):
        yh = np.bincount(cell, res.fittedvalues) / T
        return 1 - ((Ybar - yh) ** 2).sum() / ((Ybar - Ybar.mean()) ** 2).sum()

    # variance split (scripts/residual_noise.py)
    e = Y - Ybar[cell]
    m = Ybar - Yhat_full
    var_e = (e ** 2).sum() / (nc * (T - 1))
    var_m = max(m.var(ddof=1) - var_e / T, 0.0)
    rho = var_m / (var_m + var_e)

    gamma_full = bt.beta0_
    summary = dict(run=path, n_items=n, n_cells=nc, T=T, pmi_C=C,
                   r2_full=res_full.rsquared, f_pvalue_full=res_full.f_pvalue,
                   r2_feat=res_feat.rsquared, f_pvalue_feat=res_feat.f_pvalue,
                   gamma_full=gamma_full, gamma_feat=res_feat.params[0],
                   gamma_full_se_cluster=sm.OLS(Y, Xf).fit(cov_type="cluster", cov_kwds={"groups": pair}).bse[0],
                   var_e=var_e, var_m=var_m, rho=rho,
                   F_i=f1, df_num_i=df1, df_den_i=res_full.df_resid, p_i=p1,
                   F_ii=f2, df_num_ii=df2, df_den_ii=res_sat.df_resid, p_ii=p2,
                   p_ii_bootstrap=p_boot, boot_F_median=np.median(Fb), boot_F_q95=np.quantile(Fb, 0.95), B=B_BOOT,
                   r2_cell_full=cell_r2(res_full), r2_cell_feat=cell_r2(res_feat))
    scores = pd.Series(bt.weights_["id"]).rename("score")
    pvals = pd.Series({k.replace("id_", "", 1): v for k, v in bt.pvalues_.items()}).rename("p_vs_reference_plain")
    return summary, pd.DataFrame(wrows), scores, pvals


# ------------------------------------------------------------------ run
rng = np.random.default_rng(SEED)
runs = {e: load_exp(f) for e, f in EXPS.items()}
for e in EXPS:
    missing = [m for m in MODELS if m not in runs[e]]
    assert not missing, f"experiment {e}: no complete run for {missing}"

summ, wts, sc_rows = [], [], []
for m in MODELS:
    for e in EXPS:
        path, d = runs[e][m]
        s, w, scores, pv = analyse(path, d, "real", rng)
        summ.append(dict(model=m, experiment=e, **s))
        wts.append(w.assign(model=m, experiment=e))
        ref = sorted(scores.index)[0]
        for iid, v in scores.items():
            sc_rows.append(dict(model=m, experiment=e, item_id=iid, score=v,
                                p_vs_reference_plain=pv.get(iid, np.nan), reference_item=ref))
    print(f"{m}: done")
summ = pd.DataFrame(summ)
wts = pd.concat(wts)[["model", "experiment", "feature", "level", "weight", "se_cluster", "p_vs_reference_plain",
                      "reference_level"]]
sc = pd.DataFrame(sc_rows).merge(ITEMS[["item_id", "brand", "model_name"]], on="item_id")
sc["rank"] = sc.groupby(["model", "experiment"]).score.rank(ascending=False, method="min").astype(int)
sc = sc.sort_values(["model", "experiment", "rank"])

# step G: synthetic grid, no contract
syn_summ, syn_w = [], []
for m in MODELS:
    d = pd.read_csv(f"{SYNTH[m]}/scores.csv")
    c = json.load(open(f"{SYNTH[m]}/config.json"))
    fr = c["frame"]["name"] if isinstance(c["frame"], dict) else c["frame"]
    assert c["constraints_id"] == "none" and fr == "shopping" and len(c["templates"]) == T and len(d) == N_ROWS
    s, w, _, _ = analyse(SYNTH[m], d, "synth", rng)
    syn_summ.append(dict(model=m, experiment="synthetic", **s))
    syn_w.append(w.assign(model=m, experiment="synthetic"))
syn_summ = pd.DataFrame(syn_summ)
syn_w = pd.concat(syn_w)

# ------------------------------------------------------------------ tables
summ.to_csv(f"{OUT}/fit_summary.csv", index=False)                       # A + B, one row per model x experiment
wts.to_csv(f"{OUT}/feature_weights.csv", index=False)                     # D
sc.to_csv(f"{OUT}/item_scores.csv", index=False)                          # C
syn_summ.to_csv(f"{OUT}/synthetic_fit_summary.csv", index=False)
syn_w.to_csv(f"{OUT}/synthetic_feature_weights.csv", index=False)

stepB = summ[["model", "experiment", "F_i", "df_num_i", "df_den_i", "p_i", "F_ii", "df_num_ii", "df_den_ii", "p_ii",
              "p_ii_bootstrap", "r2_cell_full", "r2_cell_feat"]]
stepB.to_csv(f"{OUT}/table_step_B.csv", index=False)

# E
wide = sc.pivot_table(index=["model", "item_id"], columns="experiment", values="score").reset_index()
wide = wide.merge(ITEMS[["item_id", "brand", "model_name"]], on="item_id")
wide["diff_A_minus_B"] = wide.A - wide.B
wide.to_csv(f"{OUT}/E_score_diff_per_laptop.csv", index=False)
E_sp = pd.DataFrame([dict(model=m, spearman_A_B=stats.spearmanr(g.A, g.B)[0], p=stats.spearmanr(g.A, g.B)[1])
                     for m, g in wide.groupby("model", sort=False)])
E_sp.to_csv(f"{OUT}/E_spearman_A_vs_B.csv", index=False)
E_br = wide.pivot_table(index="model", columns="brand", values="diff_A_minus_B", aggfunc="mean").loc[MODELS]
E_br.to_csv(f"{OUT}/E_mean_diff_per_brand.csv")

# F
F_rows = []
for e in EXPS:
    piv = sc[sc.experiment == e].pivot(index="item_id", columns="model", values="score")
    for i, m1 in enumerate(MODELS):
        for m2 in MODELS[i + 1:]:
            r, p = stats.spearmanr(piv[m1], piv[m2])
            F_rows.append(dict(experiment=e, model_1=m1, model_2=m2, spearman=r, p=p))
F_tab = pd.DataFrame(F_rows)
F_tab.to_csv(f"{OUT}/F_spearman_between_models.csv", index=False)

# G
G_w = []
for m in MODELS:
    for f in ["brand", "ram"]:
        for l in SYNTH_LEVELS[f]:
            row = dict(model=m, feature=f, level=l)
            for e, src in [("synthetic", syn_w), ("A", wts), ("B", wts)]:
                q = src[(src.model == m) & (src.experiment == e) & (src.feature == f) & (src.level == l)].iloc[0]
                row[f"w_{e}"], row[f"se_{e}"] = q.weight, q.se_cluster
            G_w.append(row)
G_w = pd.DataFrame(G_w)
G_w.to_csv(f"{OUT}/table_step_G_weights.csv", index=False)
G_cols = ["r2_full", "r2_feat", "gamma_full", "var_e", "var_m", "rho", "F_i", "df_num_i", "df_den_i", "p_i", "F_ii",
          "df_num_ii", "df_den_ii", "p_ii", "p_ii_bootstrap", "r2_cell_full", "r2_cell_feat"]
G_fit = pd.concat([summ, syn_summ])[["model", "experiment"] + G_cols]
G_fit["experiment"] = pd.Categorical(G_fit.experiment, ["synthetic", "A", "B"])
G_fit["model"] = pd.Categorical(G_fit.model, MODELS)
G_fit = G_fit.sort_values(["model", "experiment"])
G_fit.to_csv(f"{OUT}/table_step_G_fit.csv", index=False)

# ------------------------------------------------------------------ figures
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# 1. laptops ordered by A score, colored by brand; B as a second marker
fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=False)
for ax, m in zip(axes, MODELS):
    g = wide[wide.model == m].sort_values("A")
    y = np.arange(len(g))
    for yy, (_, r) in zip(y, g.iterrows()):
        ax.plot([r.A, r.B], [yy, yy], color="#cccccc", lw=0.8, zorder=1)
    ax.scatter(g.A, y, c=g.brand.map(BRAND_COLOR), s=36, zorder=3, label="A: name shown")
    ax.scatter(g.B, y, facecolors="none", edgecolors=g.brand.map(BRAND_COLOR), s=36, lw=1.2, zorder=3,
               label="B: name hidden")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.item_id} {r.brand} {r.model_name}" for _, r in g.iterrows()], fontsize=6.5)
    ax.axvline(0, color="#888888", lw=0.6)
    ax.set_title(m)
    ax.set_xlabel("BT-full score (centred)")
handles = [plt.Line2D([], [], marker="o", ls="", color=BRAND_COLOR[b], label=b) for b in BRANDS]
handles += [plt.Line2D([], [], marker="o", ls="", color="k", label="A: name shown"),
            plt.Line2D([], [], marker="o", ls="", mfc="none", color="k", label="B: name hidden")]
fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False)
fig.suptitle("45 real laptops, ordered by BT-full score in experiment A")
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
fig.savefig(f"{OUT}/fig1_item_scores.png", dpi=150)
plt.close(fig)

# 2. BT-feat weights, grouped by feature, A next to B, error bars = 1.96 cluster SE
order = [(f, l) for f in LEVELS for l in LEVELS[f]] + [("storage_size", "per 1 SD of log2 GB")]
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
x = np.arange(len(order))
for ax, m in zip(axes, MODELS):
    for k, (e, col, off) in enumerate([("A", "#1f5f99", -0.18), ("B", "#e08a2c", 0.18)]):
        g = wts[(wts.model == m) & (wts.experiment == e)].set_index(["feature", "level"]).loc[order]
        ax.errorbar(x + off, g.weight, yerr=1.96 * g.se_cluster, fmt="o", ms=4, color=col, capsize=2, lw=1,
                    label={"A": "A: name shown", "B": "B: name hidden"}[e])
    ax.axhline(0, color="#888888", lw=0.6)
    bounds = np.cumsum([len(LEVELS[f]) for f in LEVELS])
    for b in bounds:
        ax.axvline(b - 0.5, color="#dddddd", lw=0.8)
    ax.set_title(m, loc="left")
    ax.set_ylabel("weight (centred)")
axes[0].legend(frameon=False, loc="upper right")
axes[-1].set_xticks(x)
axes[-1].set_xticklabels([f"{f}: {l}" if l != "per 1 SD of log2 GB" else "storage size\n(per SD log2 GB)"
                          for f, l in order], rotation=60, ha="right", fontsize=8)
fig.suptitle("BT-feat weights, error bars = 1.96 x cluster-robust SE (cluster = unordered pair)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_feature_weights.png", dpi=150)
plt.close(fig)

# 3. scatter A vs B, colored by brand, identity line
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for ax, m in zip(axes, MODELS):
    g = wide[wide.model == m]
    ax.scatter(g.A, g.B, c=g.brand.map(BRAND_COLOR), s=30)
    lo_, hi_ = min(g.A.min(), g.B.min()), max(g.A.max(), g.B.max())
    pad = 0.05 * (hi_ - lo_)
    ax.plot([lo_ - pad, hi_ + pad], [lo_ - pad, hi_ + pad], color="#888888", lw=0.8, ls="--")
    ax.set_aspect("equal")
    ax.set_title(f"{m}   Spearman = {E_sp.set_index('model').loc[m, 'spearman_A_B']:.3f}")
    ax.set_xlabel("score, A: name shown")
    ax.set_ylabel("score, B: name hidden")
fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BRAND_COLOR[b], label=b) for b in BRANDS],
           loc="lower center", ncol=5, frameon=False)
fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.18, wspace=0.25)
fig.savefig(f"{OUT}/fig3_scatter_A_vs_B.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ print
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)
print("\n== A. fit quality\n", summ[["model", "experiment", "r2_full", "r2_feat", "gamma_full", "gamma_full_se_cluster",
                                     "var_e", "var_m", "rho", "f_pvalue_full", "f_pvalue_feat"]].round(4).to_string(index=False))
print("\n== B. goodness of fit\n", stepB.to_string(index=False))
print("\n== E. Spearman A vs B\n", E_sp.round(4).to_string(index=False))
print("\n== E. mean A - B per brand\n", E_br.round(3).to_string())
print("\n== F. Spearman between models\n", F_tab.round(4).to_string(index=False))
print("\n== G. weights\n", G_w.round(3).to_string(index=False))
print("\n== G. fit\n", G_fit.to_string(index=False))
