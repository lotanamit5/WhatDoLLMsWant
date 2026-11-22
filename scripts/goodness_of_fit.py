"""Goodness of fit of the BT-with-bias model (OLS on the logit margins), per run.

Standalone. Reads data/, writes figs/goodness_of_fit/. Does not touch src/ or data/ or any other script.
Run from the repo root:  ~/miniconda3/envs/whatdo-llms-want/bin/python scripts/goodness_of_fit.py

Runs (Lotan, 2026-11-18):
  check : the two template-check runs, T = 5 templates per cell -> every test below.
          data/real_laptops_template_check (gemma-27, no contract, 200 pairs x 2 orders)
          data/synthetic_cars_template_check (gemma-27, no contract, 200 pairs x 2 orders)
  main  : every complete run of data/real_laptops (81) and data/synthetic_cars (84). Template 0 only, so T = 1:
          M3 fits every cell exactly, and the tests that need template noise ((ii), (iii), bootstrap, var_e, rho)
          are left empty. Test (i) is then computed against the M2 residual instead of the pure template noise
          (column i_denominator says which).

Y = score_a - score_b (PMI is off in all of these runs: max score is 0). Cell c = ordered pair (a in slot A,
b in slot B) inside one run; it has T rows, one per template.
  M1 feature BT : gamma + (z_a - z_b) . w     real laptops: brand / screen / RAM, the 19 raw levels (no binning,
                                             2026-11-18; 18.4-inch and 32GB belong to the same single laptop, so
                                             one column is redundant and df uses the rank); cars: the 6 features.
  M2 full BT    : gamma + beta_a - beta_b, one beta per item (= fit_full of synthetic_cars_analysis.py).
  M3 saturated  : one free mean per cell.
A compound cars run is used with all its rows (500 shared pairs + every pair with a target car), as stored.

F (restricted R vs M3) = [(RSS_R - RSS_M3) / (df_R - df_M3)] / [RSS_M3 / df_M3]
F (i), M1 vs M2       = [(RSS_M1 - RSS_M2) / (df_M1 - df_M2)] / [RSS_M3 / df_M3]  (T >= 2)
                                                                / [RSS_M2 / df_M2]  (T = 1)
Bootstrap ((ii), (iii)): cell fitted value of the restricted model + that cell's own template deviations
(Y_ct - Ybar_c) * sqrt(T / (T - 1)), resampled with replacement inside the cell; refit; F. B = 1000, seed fixed.
var_e, var_m, rho, the 10 misfit bins and the CR1 cluster SEs follow scripts/residual_noise.py.
Holm at 0.05 inside each test family, across the runs of that family.

Output: gof_runs.csv (one row per run), gof_misfit_bins.csv (M2 misfit by |Yhat| bin), gof_pass_ii.csv.
Numbers only.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.getcwd())
from src.bt import FeatureBT      # used read-only, only to check M1

OUT = "figs/goodness_of_fit"
B, SEED, NB = 1000, 20261118, 10
os.makedirs(OUT, exist_ok=True)


DOMAINS = {                                    # bins: optional {feature: function} to merge levels; none used now
    "real_laptops": dict(feats=["brand", "screen", "ram"], bins={}),
    "synthetic_cars": dict(feats=["company", "color", "fuel", "transmission", "ownership", "horsepower"], bins={}),
}
SOURCES = [("check", "real_laptops", "data/real_laptops_template_check"),
           ("check", "synthetic_cars", "data/synthetic_cars_template_check"),
           ("main", "real_laptops", "data/real_laptops"),
           ("main", "synthetic_cars", "data/synthetic_cars")]


def load():
    runs, skipped = [], []
    for kind, dom, base in SOURCES:
        for r in sorted(os.listdir(base)):
            cf, sc = f"{base}/{r}/config.json", f"{base}/{r}/scores.csv"
            if not os.path.isfile(cf):
                continue                                      # log-only folder
            if not os.path.isfile(sc):
                skipped.append((f"{base}/{r}", "config.json but no scores.csv"))
                continue
            c = json.load(open(cf))
            d = pd.read_csv(sc)
            if "n_rows" in c and len(d) != c["n_rows"]:
                skipped.append((f"{base}/{r}", f"{len(d)} rows, expected {c['n_rows']}"))
                continue
            runs.append(dict(kind=kind, domain=dom, path=f"{base}/{r}", model=f"{c['model_family']}-{c['model_size']}",
                             frame=c["frame"]["name"], cid=c["constraints_id"], d=d))
    return runs, skipped


def ols(X, Y):
    """Least squares. Returns coefficients, fitted values, RSS, residual df (n - rank)."""
    co, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    fit = X @ co
    return co, fit, float(((Y - fit) ** 2).sum()), len(Y) - rank


def existing_fit_full(ia, ib, Y, NI):
    """Verbatim copy of fit_full in scripts/synthetic_cars_analysis.py (the existing BT-full fit)."""
    X = np.zeros((len(Y), 1 + NI))
    X[:, 0] = 1
    r = np.arange(len(Y))
    X[r, 1 + ia] += 1
    X[r, 1 + ib] -= 1
    co = np.linalg.lstsq(X, Y, rcond=None)[0]
    sst = ((Y - Y.mean()) ** 2).sum()
    return co[0], co[1:], 1 - ((Y - X @ co) ** 2).sum() / sst


def cluster_se(X, resid, groups):
    """CR1 sandwich (copy of cluster_cov in scripts/residual_noise.py), with a pseudo-inverse bread."""
    N, K = X.shape
    bread = np.linalg.pinv(X.T @ X)
    G = groups.max() + 1
    s = np.zeros((G, K))
    np.add.at(s, groups, X * resid[:, None])
    V = bread @ (s.T @ s) @ bread * G / (G - 1) * (N - 1) / (N - K)
    return np.sqrt(np.diag(V))


def f_test(rss_r, df_r, rss_f, df_f, rss_den, df_den):
    num = (rss_r - rss_f) / (df_r - df_f)
    F = num / (rss_den / df_den)
    return F, df_r - df_f, df_den, float(stats.f.sf(F, df_r - df_f, df_den))


def analyse(r):
    d, dom = r["d"], DOMAINS[r["domain"]]
    feats, bins = dom["feats"], dom["bins"]
    Y = (d.score_a - d.score_b).values.astype(float)
    N = len(Y)

    # items and cells; rows sorted by cell so a cell's T rows are contiguous
    items = sorted(set(d.a_id) | set(d.b_id))
    ix = {it: k for k, it in enumerate(items)}
    n = len(items)
    ia, ib = d.a_id.map(ix).values, d.b_id.map(ix).values
    ucell, cell = np.unique(ia * n + ib, return_inverse=True)
    C = len(ucell)
    cnt = np.bincount(cell)
    T = int(cnt[0])
    T_equal = bool((cnt == T).all())
    lo, hi = np.minimum(ia, ib), np.maximum(ia, ib)
    pair = np.unique(lo * n + hi, return_inverse=True)[1]

    # M1 design: one-hot levels (binned for the real laptops), first level of each feature dropped
    lv_a = {f: (d[f"a_{f}"].map(bins[f]) if f in bins else d[f"a_{f}"]).values for f in feats}
    lv_b = {f: (d[f"b_{f}"].map(bins[f]) if f in bins else d[f"b_{f}"]).values for f in feats}
    cols = [np.ones(N)]
    for f in feats:
        for lvl in sorted(set(lv_a[f]) | set(lv_b[f]))[1:]:
            cols.append((lv_a[f] == lvl).astype(float) - (lv_b[f] == lvl).astype(float))
    X1 = np.column_stack(cols)
    X1 = X1[:, np.r_[True, np.abs(X1[:, 1:]).sum(0) > 0]]           # drop levels that never differ in a pair
    # M2 design: one column per item, item 0 dropped
    X2 = np.zeros((N, n))
    X2[:, 0] = 1
    rows = np.arange(N)
    X2[rows[ia > 0], ia[ia > 0]] += 1
    X2[rows[ib > 0], ib[ib > 0]] -= 1

    co1, fit1, rss1, df1 = ols(X1, Y)
    co2, fit2, rss2, df2 = ols(X2, Y)
    Ybar = np.bincount(cell, Y) / cnt
    rss3, df3 = float(((Y - Ybar[cell]) ** 2).sum()), N - C

    # checks against the existing fitting code
    g_ex, b_ex, r2_ex = existing_fit_full(ia, ib, Y, n)
    r2_rows2 = 1 - rss2 / ((Y - Y.mean()) ** 2).sum()
    u_mine = np.r_[0.0, co2[1:]]
    u_mine, b_ex = u_mine - u_mine.mean(), b_ex - b_ex.mean()
    fr = pd.DataFrame({"score_a": d.score_a.values, "score_b": d.score_b.values})
    for f in feats:
        fr[f"a_{f}"], fr[f"b_{f}"] = lv_a[f], lv_b[f]
    r2_fbt = FeatureBT(r2_warn=None).fit(fr).r2_
    r2_rows1 = 1 - rss1 / ((Y - Y.mean()) ** 2).sum()

    row = dict(kind=r["kind"], domain=r["domain"], run=r["path"], model=r["model"], frame=r["frame"],
               constraints_id=r["cid"], n_items=n, n_rows=N, n_cells=C, T=T, T_equal=T_equal,
               check_M2_vs_fit_full_r2=abs(r2_rows2 - r2_ex), check_M2_vs_fit_full_u=float(np.abs(u_mine - b_ex).max()),
               check_M2_gamma_diff=abs(co2[0] - g_ex), check_M1_vs_FeatureBT_r2=abs(r2_rows1 - r2_fbt),
               p_M1=X1.shape[1], p_M2=X2.shape[1], df_M1=df1, df_M2=df2, df_M3=df3,
               RSS_M1=rss1, RSS_M2=rss2, RSS_M3=rss3, R2_rows_M1=r2_rows1, R2_rows_M2=r2_rows2,
               gamma_M2=co2[0])

    # 2. nested F-tests
    rep = T >= 2 and T_equal
    if rep:
        row.update(zip(["F_i", "dfn_i", "dfd_i", "p_i_table"], f_test(rss1, df1, rss2, df2, rss3, df3)))
        row["i_denominator"] = "M3 (template noise)"
        row.update(zip(["F_ii", "dfn_ii", "dfd_ii", "p_ii_table"], f_test(rss2, df2, rss3, df3, rss3, df3)))
        row.update(zip(["F_iii", "dfn_iii", "dfd_iii", "p_iii_table"], f_test(rss1, df1, rss3, df3, rss3, df3)))
    else:
        row.update(zip(["F_i", "dfn_i", "dfd_i", "p_i_table"], f_test(rss1, df1, rss2, df2, rss2, df2)))
        row["i_denominator"] = "M2 residual (T = 1)"
        for t in ("ii", "iii"):
            row.update({f"F_{t}": np.nan, f"dfn_{t}": np.nan, f"dfd_{t}": np.nan, f"p_{t}_table": np.nan})

    # 3. bootstrap p-values for (ii) and (iii)
    row["p_ii_boot"] = row["p_iii_boot"] = np.nan
    if rep:
        rng = np.random.default_rng(SEED)
        order = np.argsort(cell, kind="stable")                      # rows grouped by cell, T each
        dev = ((Y - Ybar[cell])[order] * np.sqrt(T / (T - 1))).reshape(C, T)
        idx = rng.integers(0, T, size=(C, T, B))
        Estar = np.take_along_axis(dev[:, :, None].repeat(B, axis=2), idx, axis=1).reshape(C * T, B)
        cell_sorted = cell[order]
        for t, X, fit in (("ii", X2, fit2), ("iii", X1, fit1)):
            X = X[order]
            Yhat_c = (np.bincount(cell, fit) / cnt)[cell_sorted]
            Ys = Yhat_c[:, None] + Estar
            co = np.linalg.lstsq(X, Ys, rcond=None)[0]
            rss_r = ((Ys - X @ co) ** 2).sum(0)
            ybar_s = Ys.reshape(C, T, B).mean(1)
            rss_3 = ((Ys.reshape(C, T, B) - ybar_s[:, None, :]) ** 2).sum((0, 1))
            dfr = row["df_M2"] if t == "ii" else row["df_M1"]
            Fs = ((rss_r - rss_3) / (dfr - df3)) / (rss_3 / df3)
            row[f"p_{t}_boot"] = float((Fs >= row[f"F_{t}"]).mean())

    # 4. effect sizes on the cell means
    sst_c = ((Ybar - Ybar.mean()) ** 2).sum()
    for k, fit in (("M1", fit1), ("M2", fit2)):
        Yhat = np.bincount(cell, fit) / cnt
        m = Ybar - Yhat
        row[f"R2_cell_{k}"] = 1 - (m ** 2).sum() / sst_c
        row[f"misfit_rmse_{k}"] = float(np.sqrt((m ** 2).mean()))
        row[f"var_m_raw_{k}"] = float(m.var(ddof=1))
    if rep:
        e = Y - Ybar[cell]
        var_e = (e ** 2).sum() / (C * (T - 1))
        row["var_e"], row["noise_sd"] = var_e, float(np.sqrt(var_e))
        for k in ("M1", "M2"):
            vm = max(row[f"var_m_raw_{k}"] - var_e / T, 0.0)
            row[f"var_m_{k}"], row[f"rho_{k}"] = vm, vm / (vm + var_e)
    else:
        for k in ("var_e", "noise_sd", "var_m_M1", "rho_M1", "var_m_M2", "rho_M2"):
            row[k] = np.nan

    # 5. where the M2 misfit sits
    Yhat2 = np.bincount(cell, fit2) / cnt
    m2 = Ybar - Yhat2
    ca, cb = ucell // n, ucell % n
    M = np.full((n, n), np.nan)
    M[ca, cb] = Ybar
    Dbar = (M - M.T) / 2                                              # order-averaged half difference
    S = Dbar[:, :, None] + Dbar[None, :, :] + Dbar.T[:, None, :]       # S[a, b, c] = Dbar[a,b] + Dbar[b,c] + Dbar[c,a]
    a_, b_, c_ = np.ogrid[:n, :n, :n]
    tri = (a_ < b_) & (b_ < c_) & np.isfinite(S)
    row["n_triples"] = int(tri.sum())
    row["intransitivity_mean_abs"] = float(np.abs(S[tri]).mean()) if tri.any() else np.nan
    up = np.triu(np.isfinite(Dbar), 1)
    row["mean_abs_Dbar"] = float(np.abs(Dbar[up]).mean()) if up.any() else np.nan
    am = np.abs(m2)
    per_item = (np.bincount(ca, am, n) + np.bincount(cb, am, n)) / (np.bincount(ca, minlength=n) + np.bincount(cb, minlength=n))
    top = np.argsort(-per_item)[:5]
    row["top_misfit_items"] = "; ".join(f"{items[k]}:{per_item[k]:.2f}" for k in top)
    bins_out = []
    order = np.argsort(np.abs(Yhat2), kind="stable")
    for k, sel in enumerate(np.array_split(order, NB)):
        bins_out.append(dict(run=r["path"], kind=r["kind"], domain=r["domain"], model=r["model"],
                             constraints_id=r["cid"], bin=k, n_cells=len(sel),
                             mean_abs_yhat=np.abs(Yhat2[sel]).mean(),
                             mean_misfit=m2[sel].mean(), se_mean_misfit=m2[sel].std(ddof=1) / np.sqrt(len(sel)),
                             mean_abs_misfit=am[sel].mean(), se_mean_abs_misfit=am[sel].std(ddof=1) / np.sqrt(len(sel))))

    # 6. SEs of the M1 weights: plain OLS vs cluster-robust
    resid1 = Y - fit1
    K = X1.shape[1]
    se_plain = np.sqrt(np.diag(np.linalg.pinv(X1.T @ X1)) * rss1 / (N - K))
    for name, g in (("cell", cell), ("pair", pair)):
        ratio = (cluster_se(X1, resid1, g) / se_plain)[1:]               # the weights, not gamma
        row[f"se_ratio_{name}_median"], row[f"se_ratio_{name}_max"] = float(np.median(ratio)), float(ratio.max())
    return row, bins_out


def holm(df, pcol, mask, out):
    df[out] = np.nan
    ok = mask & df[pcol].notna()
    if ok.sum():
        df.loc[ok, out] = multipletests(df.loc[ok, pcol].values, alpha=0.05, method="holm")[1]


if __name__ == "__main__":
    runs, skipped = load()
    print(len(runs), "runs;", len(skipped), "skipped:", skipped)
    rows, bins_all = [], []
    for k, r in enumerate(runs):
        row, b = analyse(r)
        rows.append(row)
        bins_all += b
        if k % 20 == 0:
            print(k, r["path"], flush=True)
    R = pd.DataFrame(rows)

    # Holm inside each family, across its runs
    check, main = R.kind == "check", R.kind == "main"
    holm(R, "p_i_table", check, "p_i_table_holm")                       # (i), template-noise denominator
    R.loc[main, "p_i_table_holm"] = np.nan
    tmp = R.copy()
    holm(tmp, "p_i_table", main, "p_i_table_holm")                       # (i), M2-residual denominator (T = 1)
    R.loc[main, "p_i_table_holm"] = tmp.loc[main, "p_i_table_holm"]
    for t in ("ii", "iii"):
        for src in ("table", "boot"):
            holm(R, f"p_{t}_{src}", check, f"p_{t}_{src}_holm")

    R.to_csv(f"{OUT}/gof_runs.csv", index=False)
    pd.DataFrame(bins_all).to_csv(f"{OUT}/gof_misfit_bins.csv", index=False)
    P = R[check][["domain", "run", "model", "constraints_id", "F_ii", "dfn_ii", "dfd_ii", "p_ii_table",
                  "p_ii_table_holm", "p_ii_boot", "p_ii_boot_holm"]].copy()
    P["pass_ii_boot_holm"] = P.p_ii_boot_holm >= 0.05
    P["pass_ii_table_holm"] = P.p_ii_table_holm >= 0.05
    P.to_csv(f"{OUT}/gof_pass_ii.csv", index=False)

    print("\nchecks: max |R2 M2 - fit_full| =", R.check_M2_vs_fit_full_r2.max(),
          "| max |u M2 - fit_full| =", R.check_M2_vs_fit_full_u.max(),
          "| max |R2 M1 - FeatureBT| =", R.check_M1_vs_FeatureBT_r2.max())
    print("cells with unequal T:", int((~R.T_equal).sum()), "runs")
    pd.set_option("display.width", 250)
    print(P.to_string(index=False))
    print(f"\nwrote {OUT}/gof_runs.csv ({len(R)} rows), gof_misfit_bins.csv, gof_pass_ii.csv")
