"""Is one positional bias gamma per model enough, or does each contract need its own gamma_i?

Standalone. Reads data/, writes figs/gamma_per_contract/. Does not touch src/ or data/.
Run from the repo root:  /home/user1/.venvs/whatdo/bin/python scripts/gamma_per_contract.py

Per run i (= constraints_id + frame) and unordered pair {a,b}, with m the margin corrected by the
run's PMI constant C_i (gamma = beta0 + C, as in src/bt.py):
    S = (m(a,b) + m(b,a)) / 2      position only
    D = (m(a,b) - m(b,a)) / 2      preference only
Steps per model (family x size; "-pt" families are the base models):
  1. balance check: every unordered pair in both orders, same templates, same count
  2. gamma_i = mean S over pairs, 95% bootstrap CI over pairs; C per run
  3. one-way ANOVA on S across runs (H0: one shared gamma)
  4. joint OLS, per-contract feature weights, shared gamma vs gamma per run
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

FOLDERS = ["data/laptops_robustness", "data/laptops_robustness_gemma", "data/laptops_olmo",
           "data/laptops_robustness_pt", "data/laptops_olmo_pt", "data/laptops_persona"]
EXPECT_ROWS, EXPECT_TPL = 9900, 5
FEATS = ["brand", "screen", "ram"]
N_BOOT = 2000
OUT = "figs/gamma_per_contract"
rng = np.random.default_rng(0)


def pmi_c(d, tol=0.01, min_sat=0.01):
    """The PMI constant C of ONE run (same rule as model_summary / contract_tables).
    PMI off => every stored score <= 0 => C = 0. PMI on => from saturation; NaN if never saturates."""
    if max(d.score_a.max(), d.score_b.max()) <= 1e-9:
        return 0.0
    mA, mB = d.score_a.max(), d.score_b.max()
    sat = min((d.score_a > mA - tol).mean(), (d.score_b > mB - tol).mean())
    return mB - mA if sat >= min_sat else np.nan


# ---------------------------------------------------------------- load
runs, skipped = [], []
for base in FOLDERS:
    for r in sorted(os.listdir(base)):
        cf, sc = f"{base}/{r}/config.json", f"{base}/{r}/scores.csv"
        if not os.path.isfile(cf):
            continue
        if not os.path.isfile(sc):
            skipped.append((f"{base}/{r}", "config.json but no scores.csv"))
            continue
        c = json.load(open(cf))
        d = pd.read_csv(sc)
        if len(d) != EXPECT_ROWS or len(c["templates"]) != EXPECT_TPL:
            skipped.append((f"{base}/{r}", f"{len(d)} rows, {len(c['templates'])} templates"))
            continue
        runs.append(dict(path=f"{base}/{r}", family=c["model_family"], size=str(c["model_size"]),
                         frame=c["frame"]["name"], cid=c["constraints_id"], d=d))
keys = [(r["family"], r["size"], r["frame"], r["cid"]) for r in runs]
assert len(keys) == len(set(keys)), "two complete runs for one key"
print(f"{len(runs)} runs loaded; skipped {len(skipped)}:")
for p, why in skipped:
    print(f"   {p}: {why}")

LEVELS = {f: sorted(set().union(*[set(r["d"][f"a_{f}"]) for r in runs])) for f in FEATS}


def item_key(d, side):
    return list(zip(*[d[f"{side}_{f}"] for f in FEATS]))


# ---------------------------------------------------------------- 1 + 2 per run
balance_fail = []
for r in runs:
    d = r["d"]
    r["C"] = pmi_c(d)
    C = 0.0 if np.isnan(r["C"]) else r["C"]
    m = (d.score_a - d.score_b).values + C
    a, b = item_key(d, "a"), item_key(d, "b")
    t = pd.DataFrame({"a": a, "b": b, "tpl": d.template_idx.values, "m": m})
    t["lo"] = [min(x, y) for x, y in zip(a, b)]
    t["hi"] = [max(x, y) for x, y in zip(a, b)]
    t["fwd"] = [x == lo for x, lo in zip(a, t["lo"])]          # True: lo in slot A

    # 1. balance: per (pair, template), forward count == backward count, both > 0
    cnt = t.groupby(["lo", "hi", "tpl", "fwd"]).size().unstack("fwd", fill_value=0)
    for col in (True, False):
        if col not in cnt:
            cnt[col] = 0
    bad = cnt[(cnt[True] != cnt[False]) | (cnt[True] == 0)]
    # a pair missing a template entirely (present under other templates) also fails
    tpl_per_pair = t.groupby(["lo", "hi"]).tpl.nunique()
    n_tpl = t.tpl.nunique()
    bad_pairs = set(bad.index.droplevel("tpl")) | set(tpl_per_pair[tpl_per_pair != n_tpl].index)
    r["balanced"] = len(bad_pairs) == 0
    r["n_pairs"] = len(tpl_per_pair)
    for p in sorted(bad_pairs):
        balance_fail.append(dict(run=r["path"], pair=f"{p[0]} | {p[1]}"))

    # 2. S per pair: mean over templates of the two orders, then (fwd + bwd) / 2
    pm = t.groupby(["lo", "hi", "fwd"]).m.mean().unstack("fwd")
    S = ((pm[True] + pm[False]) / 2).dropna().values
    r["S"] = S
    r["gamma"] = S.mean()
    boot = S[rng.integers(0, len(S), (N_BOOT, len(S)))].mean(1)
    r["ci"] = np.percentile(boot, [2.5, 97.5])

    # design matrix for step 4: dummy(a) - dummy(b), first level of each feature dropped
    cols = []
    for f in FEATS:
        for lvl in LEVELS[f][1:]:
            cols.append((d[f"a_{f}"] == lvl).values.astype(float) - (d[f"b_{f}"] == lvl).values)
    r["X"], r["Y"] = np.column_stack(cols), m

print(f"\nBalance check: {sum(not r['balanced'] for r in runs)} of {len(runs)} runs fail")
for row in balance_fail:
    print("   FAIL", row["run"], row["pair"])

COLNAMES = [(f, lvl) for f in FEATS for lvl in LEVELS[f][1:]]


def centred(beta):
    """Raw coefficients (reference level = 0) -> weights centred to mean 0 per feature, like FeatureBT."""
    w, k = {}, 0
    for f in FEATS:
        v = np.r_[0.0, beta[k:k + len(LEVELS[f]) - 1]]
        k += len(LEVELS[f]) - 1
        for lvl, x in zip(LEVELS[f], v - v.mean()):
            w[(f, lvl)] = x
    return w


def joint_fits(rs):
    """Joint OLS over all runs of one model, per-run feature weights.
    Shared: theta = [gamma, beta_1, ..., beta_R].  Per-run: theta = [gamma_1, beta_1, ..., gamma_R, beta_R].
    Built from the normal equations block by block (the per-run blocks are independent)."""
    p = rs[0]["X"].shape[1]
    R, N = len(rs), sum(len(r["Y"]) for r in rs)
    Ys = np.concatenate([r["Y"] for r in rs])
    tss = ((Ys - Ys.mean()) ** 2).sum()
    out = {}

    # shared gamma
    A = np.zeros((1 + R * p, 1 + R * p))
    bvec = np.zeros(1 + R * p)
    A[0, 0] = N
    bvec[0] = Ys.sum()
    for i, r in enumerate(rs):
        sl = slice(1 + i * p, 1 + (i + 1) * p)
        s = r["X"].sum(0)
        A[0, sl] = A[sl, 0] = s
        A[sl, sl] = r["X"].T @ r["X"]
        bvec[sl] = r["X"].T @ r["Y"]
    th = np.linalg.solve(A, bvec)
    rss = sum(((r["Y"] - th[0] - r["X"] @ th[1 + i * p:1 + (i + 1) * p]) ** 2).sum() for i, r in enumerate(rs))
    df_res = N - len(th)
    Ainv = np.linalg.inv(A)
    se = np.sqrt(np.diag(Ainv) * rss / df_res)
    out["shared"] = dict(gamma=[th[0]] * R, beta=[th[1 + i * p:1 + (i + 1) * p] for i in range(R)],
                         se=[se[1 + i * p:1 + (i + 1) * p] for i in range(R)],
                         rss=rss, df=df_res, sigma2=rss / df_res, r2=1 - rss / tss)

    # gamma per run: blocks fully decouple, but sigma^2 is pooled over the joint fit
    th_r, inv_r, rss = [], [], 0.0
    for r in rs:
        Z = np.column_stack([np.ones(len(r["Y"])), r["X"]])
        Ai = Z.T @ Z
        t_ = np.linalg.solve(Ai, Z.T @ r["Y"])
        rss += ((r["Y"] - Z @ t_) ** 2).sum()
        th_r.append(t_)
        inv_r.append(np.linalg.inv(Ai))
    df_res = N - R * (1 + p)
    s2 = rss / df_res
    out["per_run"] = dict(gamma=[t_[0] for t_ in th_r], beta=[t_[1:] for t_ in th_r],
                          se=[np.sqrt(np.diag(I)[1:] * s2) for I in inv_r],
                          rss=rss, df=df_res, sigma2=s2, r2=1 - rss / tss)
    return out


# ---------------------------------------------------------------- 3 + 4 per model
os.makedirs(OUT, exist_ok=True)
table, wrows = [], []
models = sorted({(r["family"], r["size"]) for r in runs}, key=lambda m: (m[0], float(m[1])))
for fam, size in models:
    rs = sorted([r for r in runs if (r["family"], r["size"]) == (fam, size)],
                key=lambda r: (r["frame"] != "shopping", r["frame"], r["cid"] != "none", r["cid"]))
    name = f"{fam}-{size}"
    ref = [r for r in rs if r["frame"] == "shopping" and r["cid"] == "none"]
    g_ref = ref[0]["gamma"] if ref else np.nan

    for r in rs:
        table.append(dict(model=name, contract=r["cid"], frame=r["frame"], C=r["C"],
                          C_known=not np.isnan(r["C"]), gamma_i=r["gamma"], ci_lo=r["ci"][0], ci_hi=r["ci"][1],
                          n_pairs=r["n_pairs"], balanced=r["balanced"],
                          sign_flip_vs_none=bool(np.sign(r["gamma"]) != np.sign(g_ref)) if ref else None,
                          run=r["path"]))

    # 3. ANOVA on S across runs: all runs, and shopping-frame runs only (frame held fixed)
    F, p = stats.f_oneway(*[r["S"] for r in rs]) if len(rs) > 1 else (np.nan, np.nan)
    shop = [r for r in rs if r["frame"] == "shopping"]
    Fs, ps = stats.f_oneway(*[r["S"] for r in shop]) if len(shop) > 1 else (np.nan, np.nan)

    # 4. joint fits
    J = joint_fits(rs)
    diffs, se_ratio = [], []
    for i, r in enumerate(rs):
        ws, wp = centred(J["shared"]["beta"][i]), centred(J["per_run"]["beta"][i])
        diffs += [abs(ws[k] - wp[k]) for k in ws]
        for j, k in enumerate(COLNAMES):
            se_s, se_p = J["shared"]["se"][i][j], J["per_run"]["se"][i][j]
            se_ratio.append(se_s / se_p)
            wrows.append(dict(model=name, contract=r["cid"], frame=r["frame"], feature=k[0], level=k[1],
                              w_shared=ws[k], w_per_run=wp[k], coef_se_shared=se_s, coef_se_per_run=se_p))
    gam = [r["gamma"] for r in rs]
    table.append(dict(
        model=name, contract="SUMMARY", frame="", n_pairs=sum(r["n_pairs"] for r in rs),
        balanced=all(r["balanced"] for r in rs), C_known=all(not np.isnan(r["C"]) for r in rs),
        C=np.nan, gamma_i=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_runs=len(rs),
        C_min=min(r["C"] for r in rs), C_max=max(r["C"] for r in rs),
        gamma_min=min(gam), gamma_max=max(gam), gamma_none=g_ref,
        n_sign_flip=sum(np.sign(g) != np.sign(g_ref) for g in gam) if ref else None,
        F=F, p_F=p, n_runs_shopping=len(shop), F_shopping=Fs, p_F_shopping=ps,
        gamma_shared=J["shared"]["gamma"][0],
        max_w_diff=max(diffs),
        resid_var_shared=J["shared"]["sigma2"], resid_var_per_run=J["per_run"]["sigma2"],
        resid_var_ratio=J["shared"]["sigma2"] / J["per_run"]["sigma2"],
        r2_shared=J["shared"]["r2"], r2_per_run=J["per_run"]["r2"],
        se_median_shared=np.median(np.concatenate(J["shared"]["se"])),
        se_median_per_run=np.median(np.concatenate(J["per_run"]["se"])),
        se_ratio_min=min(se_ratio), se_ratio_max=max(se_ratio)))

T = pd.DataFrame(table)
T.to_csv(f"{OUT}/gamma_per_contract.csv", index=False)
pd.DataFrame(wrows).to_csv(f"{OUT}/weights_shared_vs_per_run.csv", index=False)
pd.DataFrame(balance_fail, columns=["run", "pair"]).to_csv(f"{OUT}/balance_failures.csv", index=False)

pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 40)
per_run = T[T.contract != "SUMMARY"]
for name, g in per_run.groupby("model", sort=False):
    print(f"\n=== {name}")
    print(g[["contract", "frame", "C", "gamma_i", "ci_lo", "ci_hi", "n_pairs", "balanced", "sign_flip_vs_none"]]
          .to_string(index=False, float_format="%.3f"))
print("\n=== SUMMARY per model")
S_ = T[T.contract == "SUMMARY"]
print(S_[["model", "n_runs", "balanced", "C_known", "C_min", "C_max", "gamma_none", "gamma_min", "gamma_max",
          "n_sign_flip", "F", "p_F", "n_runs_shopping", "F_shopping", "p_F_shopping"]]
      .to_string(index=False, float_format="%.4g"))
print(S_[["model", "gamma_shared", "max_w_diff", "resid_var_shared", "resid_var_per_run", "resid_var_ratio",
          "r2_shared", "r2_per_run", "se_median_shared", "se_median_per_run", "se_ratio_min", "se_ratio_max"]]
      .to_string(index=False, float_format="%.4g"))
