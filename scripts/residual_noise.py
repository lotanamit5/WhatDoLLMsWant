"""How does the residual noise of the OLS BT fit behave? Zero mean, equal variance, independence, normality.

Standalone. Reads data/, writes figs/residual_noise/. Does not touch src/ or data/.
Run from the repo root:  /home/user1/.venvs/whatdo/bin/python scripts/residual_noise.py

Fit per run: Y = gamma + u_a - u_b + eta, one u per item (full parameterization), u centred to mean 0.
Y = score_a - score_b + C (the run's own PMI constant, as in scripts/gamma_per_contract.py).
Cell c = ordered pair (a in slot A, b in slot B). Residual of a row = m_c + e_ct:
    e_ct = Y_ct - Ybar_c   (template component)      m_c = Ybar_c - Yhat_c   (misfit component)
Group 1 = the 189 complete 5-template runs. Group 2 = the 43-template runs. Never pooled.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

G1 = ["data/laptops_robustness", "data/laptops_robustness_gemma", "data/laptops_olmo",
      "data/laptops_robustness_pt", "data/laptops_olmo_pt", "data/laptops_persona"]
G2 = ["data/laptops_all_templates", "data/laptops_robustness_50prompts", "data/laptops_num_vs_txt"]
OUT = "figs/residual_noise"
NB = 10
os.makedirs(OUT, exist_ok=True)


def pmi_c(d, tol=0.01, min_sat=0.01):
    if max(d.score_a.max(), d.score_b.max()) <= 1e-9:
        return 0.0
    mA, mB = d.score_a.max(), d.score_b.max()
    sat = min((d.score_a > mA - tol).mean(), (d.score_b > mB - tol).mean())
    return mB - mA if sat >= min_sat else np.nan


def load(folders, n_tpl, n_rows=None):
    runs, skipped = [], []
    for base in folders:
        for r in sorted(os.listdir(base)):
            cf, sc = f"{base}/{r}/config.json", f"{base}/{r}/scores.csv"
            if not os.path.isfile(cf):
                continue
            if not os.path.isfile(sc):
                skipped.append((f"{base}/{r}", "config.json but no scores.csv"))
                continue
            c = json.load(open(cf))
            d = pd.read_csv(sc)
            if len(c["templates"]) != n_tpl or (n_rows and len(d) != n_rows):
                skipped.append((f"{base}/{r}", f"{len(d)} rows, {len(c['templates'])} templates"))
                continue
            runs.append(dict(path=f"{base}/{r}", model=f"{c['model_family']}-{c['model_size']}",
                             frame=c["frame"]["name"], cid=c["constraints_id"], alt=c["alternatives"], d=d))
    return runs, skipped


def moments(x):
    z = (x - x.mean()) / x.std()
    return stats.skew(x), stats.kurtosis(x), (np.abs(z) > 3).mean()


def bins(x, y):
    """10 equal-size bins by x. Returns per bin: mean x, mean y, SE of mean y, n."""
    order = np.argsort(x, kind="stable")
    out = []
    for k, idx in enumerate(np.array_split(order, NB)):
        out.append((k, x[idx].mean(), y[idx].mean(), y[idx].std(ddof=1) / np.sqrt(len(idx)), len(idx)))
    return out


def cluster_cov(X, resid, groups, bread):
    """CR1 sandwich (the statsmodels default correction)."""
    N, K = X.shape
    G = groups.max() + 1
    s = np.zeros((G, K))
    np.add.at(s, groups, X * resid[:, None])
    return bread @ (s.T @ s) @ bread * G / (G - 1) * (N - 1) / (N - K)


def analyse(r, group):
    d = r["d"]
    fa = [c for c in d.columns if c.startswith("a_")]
    fb = ["b_" + c[2:] for c in fa]
    A, B = list(zip(*[d[c] for c in fa])), list(zip(*[d[c] for c in fb]))
    items = sorted(set(A) | set(B))
    ix = {it: i for i, it in enumerate(items)}
    n = len(items)
    ia, ib = np.array([ix[x] for x in A]), np.array([ix[x] for x in B])
    C = pmi_c(d)
    Y = (d.score_a - d.score_b).values + (0.0 if np.isnan(C) else C)
    N = len(Y)

    # cells
    cell_code = ia * n + ib
    ucell, cell = np.unique(cell_code, return_inverse=True)
    nc = len(ucell)
    cnt = np.bincount(cell)
    T = cnt[0]
    assert (cnt == T).all() and d.template_idx.nunique() == T, f"cells without all templates: {r['path']}"
    ca, cb = ucell // n, ucell % n
    lo, hi = np.minimum(ia, ib), np.maximum(ia, ib)
    pair = np.unique(lo * n + hi, return_inverse=True)[1]

    # fit: intercept + one column per item (item 0 dropped), then centre
    X = np.zeros((N, n))
    X[:, 0] = 1
    rows = np.arange(N)
    m_a, m_b = ia > 0, ib > 0
    X[rows[m_a], ia[m_a]] += 1
    X[rows[m_b], ib[m_b]] -= 1
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ (X.T @ Y)
    fitted = X @ beta
    resid = Y - fitted
    K = n
    r2 = 1 - (resid ** 2).sum() / ((Y - Y.mean()) ** 2).sum()

    Ybar = np.bincount(cell, Y) / T
    Yhat = np.bincount(cell, fitted) / T
    e = Y - Ybar[cell]
    m = Ybar - Yhat
    sd_c = np.sqrt(np.bincount(cell, e ** 2) / (T - 1))

    # 1. variance split
    var_e = (e ** 2).sum() / (nc * (T - 1))
    var_m_raw = m.var(ddof=1)
    var_m = max(var_m_raw - var_e / T, 0.0)
    rho = var_m / (var_m + var_e)
    pos = {c: i for i, c in enumerate(ucell)}
    rev = np.array([pos.get(b * n + a, -1) for a, b in zip(ca, cb)])
    ok = rev >= 0
    corr_rev = np.corrcoef(m[ok], m[rev[ok]])[0, 1]

    row = dict(group=group, model=r["model"], contract=r["cid"], frame=r["frame"], alternatives=r["alt"],
               run=r["path"], n_items=n, n_cells=nc, T=T, C=C, gamma=beta[0] if False else np.nan,
               var_e=var_e, var_m_raw=var_m_raw, var_m=var_m, rho=rho, r2=r2, corr_m_reverse=corr_rev)

    # 2. normality
    row["e_skew"], row["e_exkurt"], row["e_beyond3sd"] = moments(e)
    row["m_skew"], row["m_exkurt"], row["m_beyond3sd"] = moments(m)
    Ymat = Y[np.lexsort((d.template_idx.values, cell))].reshape(nc, T)
    if group == 2:
        live = Ymat.std(1) > 0
        p = np.array([stats.shapiro(y).pvalue for y in Ymat[live]])
        row["shapiro_share_rejected"] = (p < 0.05).mean()
        row["n_cells_zero_var"] = int((~live).sum())
        row["cell_skew_median"] = np.median(stats.skew(Ymat[live], axis=1))
        row["cell_exkurt_median"] = np.median(stats.kurtosis(Ymat[live], axis=1))

    # 3. equal variance
    binned = []
    aY = np.abs(Ybar)
    row["spearman_sd_absYbar"] = stats.spearmanr(sd_c, aY)[0]
    binned += [("3_sd_c_by_absYbar",) + b for b in bins(aY, sd_c)]
    P = 1 / (1 + np.exp(-Ymat))
    sdP, dP = P.std(1, ddof=1), np.abs(P.mean(1) - 0.5)
    row["spearman_sdP_absPbar"] = stats.spearmanr(sdP, dP)[0]
    binned += [("3_sdP_by_absPbar",) + b for b in bins(dP, sdP)]
    row["spearman_absm_absYhat"] = stats.spearmanr(np.abs(m), np.abs(Yhat))[0]
    binned += [("3_absm_by_absYhat",) + b for b in bins(np.abs(Yhat), np.abs(m))]

    # 4. zero mean
    binned += [("4_m_by_Yhat",) + b for b in bins(Yhat, m)]

    # 5. standard errors of the centred item weights, three ways
    M = np.zeros((n, n))                       # centred weights = M @ beta (intercept column is 0)
    M[1:, 1:] = np.eye(n - 1)
    M[:, 1:] -= 1.0 / n
    covs = dict(plain=bread * (resid ** 2).sum() / (N - K),
                cell=cluster_cov(X, resid, cell, bread), pair=cluster_cov(X, resid, pair, bread))
    se = {k: np.sqrt(np.diag(M @ V @ M.T)) for k, V in covs.items()}
    seg = {k: np.sqrt(V[0, 0]) for k, V in covs.items()}
    row["gamma"] = beta[0]
    for k in ("cell", "pair"):
        row[f"se_ratio_{k}_median"] = np.median(se[k] / se["plain"])
        row[f"se_ratio_{k}_max"] = (se[k] / se["plain"]).max()
        row[f"se_ratio_{k}_gamma"] = seg[k] / seg["plain"]
    row["sqrt_1_T1_rho"] = np.sqrt(1 + (T - 1) * rho)

    bdf = pd.DataFrame(binned, columns=["step", "bin", "x_mean", "value", "se", "n"])
    for k in ("group", "model", "contract", "frame", "alternatives", "run"):
        bdf.insert(0, k, row[k])
    return row, bdf, dict(e=e, m=m, Ybar=Ybar, Yhat=Yhat, sd_c=sd_c, Ymat=Ymat, bdf=bdf)


# ---------------------------------------------------------------- run
runs1, skip1 = load(G1, 5, 9900)
runs2, skip2 = load(G2, 43)
print(f"group 1: {len(runs1)} runs (skipped {len(skip1)}); group 2: {len(runs2)} runs (skipped {len(skip2)})")
for p, why in skip1 + skip2:
    print(f"   skipped {p}: {why}")

rows, bdfs, keep = [], [], {}
FIG_MODELS = ["qwen-72", "gemma-27", "olmo-32"]
for group, runs in ((1, runs1), (2, runs2)):
    for r in runs:
        row, bdf, extra = analyse(r, group)
        rows.append(row)
        bdfs.append(bdf)
        if r["cid"] == "none" and r["frame"] == "shopping" and (group == 2 or r["model"] in FIG_MODELS):
            keep[(group, r["model"])] = extra
R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/residual_noise_per_run.csv", index=False)
pd.concat(bdfs).to_csv(f"{OUT}/residual_noise_binned.csv", index=False)

pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 40)
COLS = ["rho", "r2", "e_skew", "e_exkurt", "spearman_sd_absYbar", "se_ratio_cell_median", "se_ratio_pair_median"]
order = {m: i for i, m in enumerate(dict.fromkeys(R.model))}
for g in (1, 2):
    by = ["model"] if g == 1 else ["model", "alternatives"]
    S = R[R.group == g].groupby(by, sort=False)[COLS].median()
    S.insert(0, "n_runs", R[R.group == g].groupby(by, sort=False).size())
    S.to_csv(f"{OUT}/summary_group{g}.csv")
    print(f"\n=== GROUP {g}: median over the runs of each model")
    print(S.to_string(float_format="%.3f"))
print("\n=== GROUP 2 per run")
print(R[R.group == 2][["model", "contract", "frame", "alternatives", "n_cells", "var_e", "var_m", "rho", "r2",
                       "corr_m_reverse", "e_skew", "e_exkurt", "e_beyond3sd", "m_skew", "m_exkurt", "m_beyond3sd",
                       "shapiro_share_rejected", "cell_skew_median", "cell_exkurt_median",
                       "se_ratio_cell_median", "se_ratio_cell_max", "se_ratio_pair_median", "se_ratio_pair_max",
                       "sqrt_1_T1_rho"]].T.to_string(float_format=lambda v: f"{v:.3f}"))

# ---------------------------------------------------------------- figures
INK, BLUE = "#333333", "#3b6fb6"


def qq(ax, x, title):
    (osm, osr), (slope, inter, _) = stats.probplot(x, dist="norm")
    step = max(len(osm) // 3000, 1)
    ax.plot(osm[::step], osr[::step], ".", ms=3, color=BLUE)
    ax.plot(osm[[0, -1]], slope * osm[[0, -1]] + inter, "-", lw=1, color=INK)
    ax.set(title=title, xlabel="normal quantile", ylabel="sample quantile")


for (group, model), x in keep.items():
    if group != 1:
        continue
    b = x["bdf"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    qq(ax[0], x["e"], "QQ of e (template component)")
    qq(ax[1], x["m"], "QQ of m (misfit component)")
    s = b[b.step == "3_sd_c_by_absYbar"]
    ax[2].plot(s.x_mean, s.value, "o-", color=BLUE)
    ax[2].set(title="sd within cell vs |mean Y of cell|", xlabel="|Ybar_c| (bin mean)", ylabel="mean sd_c", ylim=(0, None))
    s = b[b.step == "4_m_by_Yhat"]
    ax[3].errorbar(s.x_mean, s.value, yerr=1.96 * s.se, fmt="o-", color=BLUE, capsize=3)
    ax[3].axhline(0, color=INK, lw=0.8)
    ax[3].set(title="mean misfit vs fitted value (±1.96 SE)", xlabel="Yhat_c (bin mean)", ylabel="mean m_c")
    fig.suptitle(f"{model}, no contract, shopping frame, 5 templates")
    fig.tight_layout()
    fig.savefig(f"{OUT}/noise_{model}.png", dpi=150)
    plt.close(fig)

for (group, model), x in keep.items():
    if group != 2:
        continue
    o = np.argsort(np.abs(x["Ybar"]))
    mid = len(o) // 2
    pick = [(o[0], "smallest"), (o[1], "smallest"), (o[mid - 1], "median"), (o[mid], "median"),
            (o[-2], "largest"), (o[-1], "largest")]
    fig, ax = plt.subplots(1, 6, figsize=(21, 3.4))
    for a, (c, lab) in zip(ax, pick):
        a.hist(x["Ymat"][c], bins=12, color=BLUE)
        a.axvline(x["Ybar"][c], color=INK, lw=1)
        a.set(title=f"{lab} |Ybar|: Ybar = {x['Ybar'][c]:.2f}, sd = {x['sd_c'][c]:.2f}", xlabel="Y (43 templates)")
        a.title.set_fontsize(9)
    ax[0].set_ylabel("templates")
    fig.suptitle(f"{model}, no contract, shopping frame, 43 templates: Y inside 6 cells")
    fig.tight_layout()
    fig.savefig(f"{OUT}/cells_43templates_{model}.png", dpi=150)
    plt.close(fig)
print("\nfigures:", sorted(f for f in os.listdir(OUT) if f.endswith(".png")))
