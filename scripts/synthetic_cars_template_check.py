"""Synthetic cars, stage 1: the template sanity check.

One run: no contract, 200 random pairs x 2 orders x 5 templates = 2,000 rows, one model.
Reports the variance split of scripts/residual_noise.py, with the same formulas:
    cell c = ordered pair (a in slot A, b in slot B), T = 5 templates per cell
    e_ct = Y_ct - Ybar_c (template component)        m_c = Ybar_c - Yhat_c (misfit component)
    var_e = sum e^2 / (n_cells (T-1))     var_m = max(var(m) - var_e / T, 0)     rho = var_m / (var_m + var_e)
Yhat comes from an OLS fit on the margins with an intercept (the position term). Two fits:
    bt_full   one parameter per car - what residual_noise.py uses
    bt_feat   one weight per feature level (6 features, 24 one-hot coordinates)
With 200 pairs the cars do not form one connected graph, so bt_full is solved by minimum-norm least
squares (the fitted values are the same for every solution).
Also: does template 0 alone (what the main runs use) pick the same car as the other four?

Standalone. Run from the repo root:
    ~/miniconda3/envs/whatdo-llms-want/bin/python scripts/synthetic_cars_template_check.py
Reads data/synthetic_cars_template_check/, writes figs/synthetic_cars/template_check*.csv.
"""
import json
import os

import numpy as np
import pandas as pd

BASE = "data/synthetic_cars_template_check"
OUT = "figs/synthetic_cars"
FEATS = ["company", "color", "fuel", "transmission", "ownership", "horsepower"]
os.makedirs(OUT, exist_ok=True)

runs = [r for r in sorted(os.listdir(BASE))
        if os.path.isfile(f"{BASE}/{r}/config.json") and os.path.isfile(f"{BASE}/{r}/scores.csv")]
rows, per_template = [], []
for r in runs:
    cfg = json.load(open(f"{BASE}/{r}/config.json"))
    d = pd.read_csv(f"{BASE}/{r}/scores.csv")
    assert len(d) == cfg["n_rows"], f"{r}: {len(d)} rows, expected {cfg['n_rows']}"
    model = f"{cfg['model_family']}-{cfg['model_size']}"
    Y = (d.score_a - d.score_b).values
    N = len(Y)
    ids = np.unique(np.r_[d.a_id.values, d.b_id.values])
    ix = {v: i for i, v in enumerate(ids)}
    ia, ib = d.a_id.map(ix).values, d.b_id.map(ix).values
    n = len(ids)
    ucell, cell = np.unique(ia * n + ib, return_inverse=True)
    nc = len(ucell)
    cnt = np.bincount(cell)
    T = cnt[0]
    assert (cnt == T).all() and d.template_idx.nunique() == T
    Ybar = np.bincount(cell, Y) / T
    e = Y - Ybar[cell]
    var_e = (e ** 2).sum() / (nc * (T - 1))

    # designs
    Xfull = np.zeros((N, 1 + n))
    Xfull[:, 0] = 1
    Xfull[np.arange(N), 1 + ia] += 1
    Xfull[np.arange(N), 1 + ib] -= 1
    za = pd.get_dummies(d[[f"a_{f}" for f in FEATS]].rename(columns=lambda c: c[2:]), dtype=float)
    zb = pd.get_dummies(d[[f"b_{f}" for f in FEATS]].rename(columns=lambda c: c[2:]), dtype=float)
    zb = zb.reindex(columns=za.columns, fill_value=0.0)
    Xfeat = np.column_stack([np.ones(N), za.values - zb.values])

    for arm, X in (("bt_full", Xfull), ("bt_feat", Xfeat)):
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        fitted = X @ beta
        rank = np.linalg.matrix_rank(X)
        Yhat = np.bincount(cell, fitted) / T
        m = Ybar - Yhat
        var_m_raw = m.var(ddof=1)
        var_m = max(var_m_raw - var_e / T, 0.0)
        rows.append(dict(model=model, run=f"{BASE}/{r}", fit=arm, n_items=n, n_cells=nc, T=T,
                         n_params=int(rank), gamma=beta[0], var_e=var_e, var_m_raw=var_m_raw, var_m=var_m,
                         rho=var_m / (var_m + var_e),
                         r2=1 - ((Y - fitted) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()))

    # template 0 against the others, on D = (Y[a,b] - Y[b,a]) / 2 per unordered pair
    lo, hi = np.minimum(ia, ib), np.maximum(ia, ib)
    sign = np.where(ia < ib, 1.0, -1.0)
    D = (pd.DataFrame(dict(pair=lo * n + hi, t=d.template_idx.values, v=sign * Y))
         .groupby(["pair", "t"]).v.mean().unstack("t"))          # mean of the two orders = D
    rest = D.drop(columns=0).mean(axis=1)
    for t in D.columns:
        Yt = Y[d.template_idx.values == t]
        per_template.append(dict(model=model, template=t,
                                 first_slot_wins_pct=100 * (Yt > 0).mean(),
                                 r_with_template_0=np.corrcoef(D[0], D[t])[0, 1],
                                 same_winner_as_template_0=int((np.sign(D[0]) == np.sign(D[t])).sum()),
                                 n_pairs=len(D)))
    print(f"\n{model}  ({BASE}/{r}, {N} rows, {n} cars, {len(D)} pairs)")
    print(f"template 0 vs the mean of the other {T - 1}: r = {np.corrcoef(D[0], rest)[0, 1]:.3f}, "
          f"same winner in {(np.sign(D[0]) == np.sign(rest)).sum()} of {len(D)} pairs; "
          f"all {T} templates agree in {(np.sign(D).nunique(axis=1) == 1).sum()} of {len(D)}")

R, P = pd.DataFrame(rows), pd.DataFrame(per_template)
R.to_csv(f"{OUT}/template_check_variance_split.csv", index=False)
P.to_csv(f"{OUT}/template_check_per_template.csv", index=False)
pd.set_option("display.width", 250)
print("\nvariance split:")
print(R.drop(columns="run").to_string(index=False, float_format="%.3f"))
print("\nper template:")
print(P.to_string(index=False, float_format="%.3f"))
