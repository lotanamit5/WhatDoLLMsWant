"""First simple test: does a GRUM-style model have any potential over one BT per contract?

Standalone. Reads data/, writes figs/grum_first_test/. Does not touch src/ or data/.
Run from the repo root:  /home/user1/.venvs/whatdo/bin/python scripts/grum_first_test.py

A. Efficiency: same accuracy with fewer samples?   B. Generalization: predict an unseen compound contract?
Arms (all OLS, minimum-norm least squares):
  bt_full  per contract   Y = gamma + beta_a - beta_b
  bt_feat  per contract   Y = gamma + (z_a - z_b).w
  grum     joint          Y = gamma_i + (z_a - z_b).(w + x_i B)
Y = score_a - score_b + C (the run's own PMI constant), shopping frame, complete 5-template runs only.
Budgets "full" / "all" are deterministic, so they are run once (seed 0).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

FOLDERS = ["data/laptops_robustness", "data/laptops_robustness_gemma", "data/laptops_olmo",
           "data/laptops_robustness_pt", "data/laptops_olmo_pt", "data/laptops_persona"]
MODELS = [("qwen", "72"), ("gemma", "27"), ("olmo", "32")]
FEATS = ["brand", "screen", "ram"]
BUDGETS_A = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
BUDGETS_B = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
N_SEEDS = 20
OUT = "figs/grum_first_test"
os.makedirs(OUT, exist_ok=True)


def pmi_c(d, tol=0.01, min_sat=0.01):
    if max(d.score_a.max(), d.score_b.max()) <= 1e-9:
        return 0.0
    mA, mB = d.score_a.max(), d.score_b.max()
    sat = min((d.score_a > mA - tol).mean(), (d.score_b > mB - tol).mean())
    return mB - mA if sat >= min_sat else np.nan


def lstsq(X, Y):
    return np.linalg.lstsq(X, Y, rcond=None)[0]


# ---------------------------------------------------------------- arms
def bt_full(c, rows=None):
    ia, ib, Y = (c["ia"], c["ib"], c["Y"]) if rows is None else (c["ia"][rows], c["ib"][rows], c["Y"][rows])
    if len(Y) == 0:
        return np.zeros(NI)
    X = np.zeros((len(Y), 1 + NI))
    X[:, 0] = 1
    r = np.arange(len(Y))
    X[r, 1 + ia] += 1
    X[r, 1 + ib] -= 1
    return lstsq(X, Y)[1:]


def bt_feat(c, rows=None):
    dz, Y = (c["dz"], c["Y"]) if rows is None else (c["dz"][rows], c["Y"][rows])
    if len(Y) == 0:
        return np.zeros(NI)
    return Z @ lstsq(np.column_stack([np.ones(len(Y)), dz]), Y)[1:]


def grum_X(parts, n_con):
    """parts: list of (column index of the contract's gamma, x, dz). Columns: [gammas | w | B flattened]."""
    blocks = []
    for k, x, dz in parts:
        G = np.zeros((len(dz), n_con))
        G[:, k] = 1
        blocks.append(np.column_stack([G, dz, np.kron(x[None, :], dz)]))
    return np.vstack(blocks)


def grum_scores(theta, n_con, x):
    w, B = theta[n_con:n_con + D], theta[n_con + D:].reshape(D, D)
    return Z @ (w + x @ B)


def gram_solve(G, b):
    """Minimum-norm least squares from the normal equations (same solution as lstsq on the rows)."""
    return np.linalg.pinv(G, rcond=1e-10, hermitian=True) @ b


def metrics(s, c):
    pred = s[IU] - s[JU]
    sp, st = np.sign(pred), np.sign(c["Dbar"])
    acc = np.where(sp == 0, 0.5, sp == st).mean()
    r2 = 1 - ((c["Dbar"] - pred) ** 2).sum() / ((c["Dbar"] - c["Dbar"].mean()) ** 2).sum()
    rho = stats.spearmanr(s, c["ref"])[0] if s.std() > 0 else 0.0     # constant scores: no order, 0
    return acc, r2, rho


records, inventory, summA, summB = [], [], [], []


def rec(model, part, arm, budget, seed, c, s, test=""):
    acc, r2, rho = metrics(s, c)
    records.append(dict(model=model, part=part, arm=arm, budget=budget, seed=seed, contract=c["cid"],
                        accuracy=acc, r2=r2, spearman=rho))


for fam, size in MODELS:
    model = f"{fam}-{size}"
    # ------------------------------------------------------------ load
    cons = []
    for base in FOLDERS:
        for r in sorted(os.listdir(base)):
            cf, sc = f"{base}/{r}/config.json", f"{base}/{r}/scores.csv"
            if not (os.path.isfile(cf) and os.path.isfile(sc)):
                continue
            cfg = json.load(open(cf))
            if (cfg["model_family"], str(cfg["model_size"])) != (fam, size) or cfg["frame"]["name"] != "shopping":
                continue
            d = pd.read_csv(sc)
            if len(d) != 9900 or len(cfg["templates"]) != 5:
                continue
            cons.append(dict(cid=cfg["constraints_id"], constraints=cfg["constraints"], d=d, path=f"{base}/{r}"))
    assert len({c["cid"] for c in cons}) == len(cons), "two complete runs for one key"

    LEVELS = [(f, lvl) for f in FEATS for lvl in sorted(set().union(*[set(c["d"][f"a_{f}"]) for c in cons]))]
    D = len(LEVELS)
    items = sorted(set(zip(*[cons[0]["d"][f"a_{f}"] for f in FEATS])))
    NI = len(items)
    ix = {it: i for i, it in enumerate(items)}
    Z = np.array([[float(it[FEATS.index(f)] == lvl) for f, lvl in LEVELS] for it in items])
    IU, JU = np.triu_indices(NI, 1)

    for c in cons:
        d = c["d"]
        C = pmi_c(d)
        assert not np.isnan(C)
        c["Y"] = (d.score_a - d.score_b).values + C
        c["ia"] = np.array([ix[t] for t in zip(*[d[f"a_{f}"] for f in FEATS])])
        c["ib"] = np.array([ix[t] for t in zip(*[d[f"b_{f}"] for f in FEATS])])
        c["dz"] = Z[c["ia"]] - Z[c["ib"]]
        c["x"] = np.array([float(c["constraints"].get(f) == lvl) for f, lvl in LEVELS])
        c["kind"] = "none" if not c["constraints"] else ("single" if len(c["constraints"]) == 1 else "compound")
        Mm = np.zeros((NI, NI))
        np.add.at(Mm, (c["ia"], c["ib"]), c["Y"])
        Mm /= d.template_idx.nunique()
        c["Dbar"] = ((Mm - Mm.T) / 2)[IU, JU]
        c["ref"] = bt_full(c)                                   # BT-full on the full data of the contract
    order = {"none": 0, "single": 1, "compound": 2}
    cons.sort(key=lambda c: (order[c["kind"]], c["cid"]))
    singles = {c["cid"]: c for c in cons if c["kind"] == "single"}
    none = [c for c in cons if c["kind"] == "none"][0]

    # ------------------------------------------------------------ step 0: inventory
    for c in cons:
        comp = [f"{f}={c['constraints'][f]}" for f in FEATS if f in c["constraints"]]
        c["components"] = comp
        c["testable"] = c["kind"] == "compound" and all(k in singles for k in comp)
        inventory.append(dict(model=model, contract=c["cid"], kind=c["kind"],
                              x="".join(str(int(v)) for v in c["x"]),
                              x_levels=" + ".join(lvl for (f, lvl), v in zip(LEVELS, c["x"]) if v),
                              all_components_exist=c["testable"] if c["kind"] == "compound" else "",
                              missing_components=", ".join(k for k in comp if k not in singles)
                              if c["kind"] == "compound" else "", run=c["path"]))

    # ------------------------------------------------------------ step 1: sanity
    base = [none] + list(singles.values())
    nb = len(base)
    Xb = grum_X([(k, c["x"], c["dz"]) for k, c in enumerate(base)], nb)
    Yb = np.concatenate([c["Y"] for c in base])
    th_base = lstsq(Xb, Yb)
    G0, b0 = Xb.T @ Xb, Xb.T @ Yb
    th_gram = gram_solve(G0, b0)
    dmax = dgram = 0.0
    for c in base:
        s_g, s_f = grum_scores(th_base, nb, c["x"]), bt_feat(c)
        dmax = max(dmax, np.abs((s_g[IU] - s_g[JU]) - (s_f[IU] - s_f[JU])).max())
        s_q = grum_scores(th_gram, nb, c["x"])
        dgram = max(dgram, np.abs((s_g[IU] - s_g[JU]) - (s_q[IU] - s_q[JU])).max())
    print(f"\n##### {model}: {len(cons)} shopping contracts ({len(singles)} single, "
          f"{sum(c['kind'] == 'compound' for c in cons)} compound, {sum(c['testable'] for c in cons)} testable)")
    print(f"step 1 sanity: max |GRUM - BT-feat| predicted margin = {dmax:.2e}   "
          f"(normal-equation solver vs lstsq: {dgram:.2e})")
    assert dmax < 1e-6, "sanity check failed: GRUM != BT-feat on {none + singles}"

    # ------------------------------------------------------------ part A
    nC = len(cons)
    off = np.cumsum([0] + [len(c["Y"]) for c in cons])
    N = off[-1]
    for M in BUDGETS_A + [N]:
        for seed in range(N_SEEDS if M < N else 1):
            rng = np.random.default_rng([MODELS.index((fam, size)), 1, M, seed])
            pick = np.sort(rng.choice(N, M, replace=False))
            rows = [pick[(pick >= off[k]) & (pick < off[k + 1])] - off[k] for k in range(nC)]
            X = grum_X([(k, c["x"], c["dz"][rows[k]]) for k, c in enumerate(cons)], nC)
            th = lstsq(X, np.concatenate([c["Y"][rows[k]] for k, c in enumerate(cons)]))
            for k, c in enumerate(cons):
                rec(model, "A", "bt_full", M, seed, c, bt_full(c, rows[k]))
                rec(model, "A", "bt_feat", M, seed, c, bt_feat(c, rows[k]))
                rec(model, "A", "grum", M, seed, c, grum_scores(th, nC, c["x"]))
    full_M = N

    # ------------------------------------------------------------ part B
    tests = [c for c in cons if c["testable"]]
    for c in tests:
        rec(model, "B", "grum_zero_shot", 0, 0, c, grum_scores(th_base, nb, c["x"]))
        rec(model, "B", "ref_none", 0, 0, c, bt_feat(none))
        for k in c["components"]:
            rec(model, "B", f"ref_{k}", 0, 0, c, bt_feat(singles[k]))
        n = len(c["Y"])
        for kk in BUDGETS_B + [n]:
            for seed in range(N_SEEDS if kk < n else 1):
                rng = np.random.default_rng([MODELS.index((fam, size)), 2, cons.index(c), kk, seed])
                rows = np.sort(rng.choice(n, kk, replace=False))
                rec(model, "B", "bt_full", kk, seed, c, bt_full(c, rows))
                rec(model, "B", "bt_feat", kk, seed, c, bt_feat(c, rows))
                Xk = grum_X([(nb, c["x"], c["dz"][rows])], nb + 1)        # gamma of the test contract: column nb
                P = np.r_[np.arange(nb), np.arange(nb + 1, nb + 1 + D + D * D), nb]   # base columns -> new layout
                G = Xk.T @ Xk
                b = Xk.T @ c["Y"][rows]
                # put the base normal equations into the new column layout [base gammas | test gamma | w | B]
                idx = np.r_[np.arange(nb), np.arange(nb + 1, nb + 1 + D + D * D)]
                G[np.ix_(idx, idx)] += G0
                b[idx] += b0
                rec(model, "B", "grum_few_shot", kk, seed, c, grum_scores(gram_solve(G, b), nb + 1, c["x"]))

    # ------------------------------------------------------------ summaries
    R = pd.DataFrame([r for r in records if r["model"] == model])
    A = R[R.part == "A"].groupby(["arm", "budget"])[["accuracy", "r2", "spearman"]].mean().reset_index()
    target = 0.95 * A[(A.arm == "bt_full") & (A.budget == full_M)].accuracy.iloc[0]
    for arm in ("bt_full", "bt_feat", "grum"):
        a = A[A.arm == arm].sort_values("budget")
        hit = a[a.accuracy >= target]
        f = a[a.budget == full_M].iloc[0]
        summA.append(dict(model=model, arm=arm, n_contracts=nC, full_pool_rows=full_M, acc_full=f.accuracy,
                          r2_full=f.r2, spearman_full=f.spearman, target_acc=target,
                          min_M_reaching_target=int(hit.budget.iloc[0]) if len(hit) else np.nan))
    Bm = R[R.part == "B"].groupby(["contract", "arm", "budget"])[["accuracy", "r2", "spearman"]].mean().reset_index()
    for c in tests:
        b = Bm[Bm.contract == c["cid"]]
        z = b[b.arm == "grum_zero_shot"].iloc[0]
        row = dict(model=model, contract=c["cid"], zero_acc=z.accuracy, zero_r2=z.r2, zero_spearman=z.spearman)
        for arm in ["ref_none"] + [f"ref_{k}" for k in c["components"]]:
            q = b[b.arm == arm].iloc[0]
            row[f"{arm}"] = f"{q.accuracy:.3f} / {q.r2:.3f} / {q.spearman:.3f}"
        for arm in ("bt_full", "bt_feat"):
            a = b[(b.arm == arm)].sort_values("budget")
            hit = a[a.accuracy >= z.accuracy]
            row[f"min_k_{arm}"] = int(hit.budget.iloc[0]) if len(hit) else np.nan
            row[f"{arm}_all_rows_acc"] = a.accuracy.iloc[-1]
        summB.append(row)

    # ------------------------------------------------------------ figures
    COL = dict(bt_full="#3b6fb6", bt_feat="#d9822b", grum="#2e9e6b", grum_few_shot="#2e9e6b")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sd = R[R.part == "A"].groupby(["arm", "budget", "seed"])[["accuracy", "r2"]].mean().groupby(["arm", "budget"])
    mu, sg = sd.mean(), sd.std().fillna(0)
    for j, met in enumerate(("accuracy", "r2")):
        for arm in ("bt_full", "bt_feat", "grum"):
            x = mu.loc[arm].index.values
            ax[j].plot(x, mu.loc[arm][met], "o-", ms=3, color=COL[arm], label=arm)
            ax[j].fill_between(x, mu.loc[arm][met] - sg.loc[arm][met], mu.loc[arm][met] + sg.loc[arm][met],
                               color=COL[arm], alpha=0.2, lw=0)
        ax[j].set(xscale="log", xlabel="M = training rows over all contracts", ylabel=met)
    ax[1].set(ylim=(-0.5, 1.02), ylabel="R² (axis cut at −0.5)")
    ax[0].legend(frameon=False)
    fig.suptitle(f"{model}, part A: mean over {nC} contracts, ± sd over seeds")
    fig.tight_layout()
    fig.savefig(f"{OUT}/partA_{model}.png", dpi=150)
    plt.close(fig)

    if tests:
        fig, axs = plt.subplots(len(tests), 2, figsize=(12, 3.6 * len(tests)), squeeze=False)
        RB = R[R.part == "B"]
        REFC = ["#888888", "#b05cc6", "#c44e52", "#8c6d31"]
        for i, c in enumerate(tests):
            rb = RB[RB.contract == c["cid"]]
            for j, met in enumerate(("accuracy", "r2")):
                a = axs[i, j]
                for arm in ("bt_full", "bt_feat", "grum_few_shot"):
                    g = rb[rb.arm == arm].groupby("budget")[met]
                    m_, s_ = g.mean(), g.std().fillna(0)
                    a.plot(m_.index, m_, marker="o", ms=3, color=COL[arm], label=arm, ls="--" if arm == "grum_few_shot" else "-")
                    a.fill_between(m_.index, m_ - s_, m_ + s_, color=COL[arm], alpha=0.2, lw=0)
                a.axhline(rb[rb.arm == "grum_zero_shot"][met].iloc[0], color="#2e9e6b", lw=2, label="grum_zero_shot")
                for q, arm in enumerate(["ref_none"] + [f"ref_{k}" for k in c["components"]]):
                    a.axhline(rb[rb.arm == arm][met].iloc[0], color=REFC[q], lw=1, ls=":", label=arm)
                a.set(xscale="log", xlabel="k = rows of the test contract", ylabel=met, title=c["cid"])
                if met == "r2":
                    a.set(ylim=(-0.5, 1.02), ylabel="R² (axis cut at −0.5)")
            axs[i, 0].legend(frameon=False, fontsize=7)
        fig.suptitle(f"{model}, part B: unseen compound contracts (mean ± sd over seeds)")
        fig.tight_layout()
        fig.savefig(f"{OUT}/partB_{model}.png", dpi=150)
        plt.close(fig)

pd.DataFrame(records).to_csv(f"{OUT}/grum_first_test.csv", index=False)
inv, SA, SB = pd.DataFrame(inventory), pd.DataFrame(summA), pd.DataFrame(summB)
inv.to_csv(f"{OUT}/inventory.csv", index=False)
SA.to_csv(f"{OUT}/summary_partA.csv", index=False)
SB.to_csv(f"{OUT}/summary_partB.csv", index=False)

pd.set_option("display.width", 280, "display.max_rows", 500, "display.max_columns", 40, "display.max_colwidth", 60)
print("\nx coordinates:", [lvl for f, lvl in LEVELS])
print("\n=== STEP 0: inventory")
print(inv.drop(columns="run").to_string(index=False))
print("\n=== PART A summary")
print(SA.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
A_all = pd.DataFrame(records)
A_all = A_all[A_all.part == "A"].groupby(["model", "arm", "budget"], sort=False)[["accuracy", "r2", "spearman"]].mean()
print(A_all.unstack("arm").to_string(float_format=lambda v: f"{v:.3f}"))
print("\n=== PART B summary (reference columns: accuracy / R² / Spearman)")
for m, g in SB.groupby("model", sort=False):
    print(g.dropna(axis=1, how="all").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
