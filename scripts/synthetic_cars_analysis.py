"""Synthetic cars, stage 3: the analysis (parts A-G). Numbers only, no interpretation.

Run from the repo root:
    ~/miniconda3/envs/whatdo-llms-want/bin/python scripts/synthetic_cars_analysis.py
Reads data/synthetic_cars/ and, for part G, existing laptop outputs under figs/. Writes figs/synthetic_cars/.
Does not touch src/, data/ or any existing output.

Conventions (the same as scripts/grum_first_test.py and src/bt.py):
    Y            = score_a - score_b of one prompt (one template, PMI off, so no constant is added)
    BT-full      Y = gamma + beta_a - beta_b              OLS, minimum-norm least squares, one fit per contract
    BT-feat      Y = gamma + (z_a - z_b).w                OLS; z = 24 one-hot coordinates (6 features, horsepower
                                                          categorical); w centred to mean 0 inside each feature
    gamma        the position term, one per contract
    truth        Dbar(a, b) = (Y[a first, b second] - Y[b first, a second]) / 2
    accuracy     share of pairs where sign(predicted margin) = sign(Dbar); a predicted 0 counts 0.5
    R2           1 - sum (Dbar - pred)^2 / sum (Dbar - mean Dbar)^2
    Spearman     between the predicted item scores and BT-full item scores of the same contract on its own data

A  fit quality        R2 of the two OLS fits on the rows of each train contract, and gamma
B  compliance and consistency, the code of Notebooks/contract_tables.ipynb; all 27 contracts
       (a compound is scored on a sample of pairs: every number uses the pairs that exist; "before" is the
        no-contract run on the SAME pairs)
C  shift table        w_p - w_none (BT-feat), 15 single contracts x 24 coordinates
D  contract additivity, zero rows from the compound, evaluated on the 500 shared pairs
E  efficiency         M rows drawn from the pool of the 16 train contracts, 20 seeds
F  target cars along the four tracks
G  cars next to laptops (laptop numbers are read from existing outputs, never recomputed)

PARTIAL=1 in the environment: analyse only the models whose 28 runs are all complete (for a look while
jobs still run). Without it the script stops if a run is missing.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))     # for src.bt
BASE = os.environ.get("CARS_DATA", "data/synthetic_cars")
OUT = os.environ.get("CARS_OUT", "figs/synthetic_cars")
DESIGN = "data/synthetic_cars/design"
PARTIAL = os.environ.get("PARTIAL", "") == "1"
os.makedirs(OUT, exist_ok=True)

MODELS = [("qwen", "72"), ("gemma", "27"), ("olmo", "32")]
FEATS = ["company", "color", "fuel", "transmission", "ownership", "horsepower"]
LEVELS_OF = {
    "company": ["Honda", "Toyota", "Ford", "BMW", "Audi", "Mercedes"],
    "color": ["white", "black", "silver", "red", "blue", "grey"],
    "fuel": ["petrol", "diesel"],
    "transmission": ["manual", "automatic"],
    "ownership": ["private first owner", "private second owner", "private third owner", "ex-lease", "ex-rental"],
    "horsepower": ["100 hp", "150 hp", "200 hp"],
}
LEVELS = [(f, lvl) for f in FEATS for lvl in LEVELS_OF[f]]
D = len(LEVELS)                                              # 24
FIDX = {f: [k for k, (g, _) in enumerate(LEVELS) if g == f] for f in FEATS}
TRACKS = {
    "luxury": [("transmission", "automatic"), ("company", "BMW"), ("horsepower", "200 hp"), ("color", "black")],
    "economy": [("fuel", "petrol"), ("transmission", "manual"), ("horsepower", "100 hp"), ("company", "Honda")],
    "family": [("transmission", "automatic"), ("ownership", "private first owner"), ("company", "Toyota"),
               ("color", "white")],
    "work": [("fuel", "diesel"), ("horsepower", "150 hp"), ("company", "Ford"), ("ownership", "ex-lease")],
}
BUDGETS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
N_SEEDS = 20

ITEMS = pd.read_csv(f"{DESIGN}/items.csv")
assert list(ITEMS.id) == list(range(1, 121))
NI = len(ITEMS)
Z = np.array([[float(row[f] == lvl) for f, lvl in LEVELS] for _, row in ITEMS.iterrows()])
IU, JU = np.triu_indices(NI, 1)                              # all 7,140 unordered pairs
_sh = pd.read_csv(f"{DESIGN}/compound_pairs.csv").values - 1
SH_I, SH_J = _sh.min(1), _sh.max(1)                          # the 500 shared pairs
assert len(SH_I) == 500


def cid_of(con):
    """The constraints_id slug, as scripts/data_collection.py builds it."""
    return "+".join(f"{f}={v}" for f, v in sorted(con.items())) or "none"


TRAIN = [{}] + [{f: v} for f, v in dict.fromkeys(fv for t in TRACKS.values() for fv in t)]     # none + 15 singles
SINGLES = TRAIN[1:]
COMPOUNDS = [(t, k, dict(TRACKS[t][:k])) for t in TRACKS for k in (2, 3, 4)]                   # 12
assert len(TRAIN) == 16 and len(COMPOUNDS) == 12


def track_cid(t, k):
    return cid_of(dict(TRACKS[t][:k]))


def lstsq(X, Y):
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def r2_rows(Y, fitted):
    sst = ((Y - Y.mean()) ** 2).sum()
    return 1 - ((Y - fitted) ** 2).sum() / sst if sst > 0 else np.nan


def fit_full(ia, ib, Y):
    """BT-full. Returns gamma, the 120 item scores, R2 on the rows."""
    if len(Y) == 0:
        return 0.0, np.zeros(NI), np.nan
    X = np.zeros((len(Y), 1 + NI))
    X[:, 0] = 1
    r = np.arange(len(Y))
    X[r, 1 + ia] += 1
    X[r, 1 + ib] -= 1
    co = lstsq(X, Y)
    return co[0], co[1:], r2_rows(Y, X @ co)


def fit_feat(dz, Y):
    """BT-feat. Returns gamma, the 24 weights (centred inside each feature), R2 on the rows."""
    if len(Y) == 0:
        return 0.0, np.zeros(D), np.nan
    X = np.column_stack([np.ones(len(Y)), dz])
    co = lstsq(X, Y)
    w = co[1:].copy()
    for f in FEATS:
        w[FIDX[f]] -= w[FIDX[f]].mean()
    return co[0], w, r2_rows(Y, X @ co)


def metrics(s, dbar, iu, ju, ref):
    pred = s[iu] - s[ju]
    sp, st = np.sign(pred), np.sign(dbar)
    acc = np.where(sp == 0, 0.5, sp == st).mean()
    r2 = 1 - ((dbar - pred) ** 2).sum() / ((dbar - dbar.mean()) ** 2).sum()
    rho = stats.spearmanr(s, ref)[0] if s.std() > 0 else 0.0
    return acc, r2, rho


def sig(x):
    return 1 / (1 + np.exp(-x))


def meets(con):
    return np.all([ITEMS[f].values == v for f, v in con.items()], axis=0) if con else np.ones(NI, bool)


# ------------------------------------------------------------------ load
RUNS, skipped = {}, []
for r in sorted(os.listdir(BASE)):
    cf, sc = f"{BASE}/{r}/config.json", f"{BASE}/{r}/scores.csv"
    if not os.path.isfile(cf):
        continue
    cfg = json.load(open(cf))
    if not os.path.isfile(sc):
        skipped.append((r, "config.json, no scores.csv"))
        continue
    d = pd.read_csv(sc)
    if len(d) != cfg["n_rows"]:
        skipped.append((r, f"{len(d)} rows of {cfg['n_rows']}"))
        continue
    assert cfg["frame"]["name"] == "car_shopping" and d.template_idx.nunique() == 1
    assert max(d.score_a.max(), d.score_b.max()) <= 1e-9, "scores are not log-probs: PMI is on?"
    key = (cfg["model_family"], str(cfg["model_size"]), cfg["constraints_id"])
    assert key not in RUNS, f"two complete runs for {key}"
    ia, ib = d.a_id.values - 1, d.b_id.values - 1
    Y = (d.score_a - d.score_b).values
    M = np.full((NI, NI), np.nan)
    M[ia, ib] = Y
    Dm = (M - M.T) / 2                                       # Dm[i, j] > 0: car i is preferred to car j
    RUNS[key] = dict(con=cfg["constraints"], ia=ia, ib=ib, Y=Y, dz=Z[ia] - Z[ib], M=M, Dm=Dm,
                     P=0.5 * (sig(M) + 1 - sig(M.T)), folder=r)
    with np.errstate(all="ignore"):
        RUNS[key]["borda"] = np.nanmean(Dm, axis=1)
print(f"{BASE}: {len(RUNS)} complete runs, {len(skipped)} incomplete folders skipped")
for r, why in skipped:
    print(f"   skipped {r}: {why}")

ALL_CIDS = [cid_of(c) for c in TRAIN] + [cid_of(c) for _, _, c in COMPOUNDS]
models = []
for fam, size in MODELS:
    missing = [c for c in ALL_CIDS if (fam, size, c) not in RUNS]
    if missing:
        print(f"{fam}-{size}: {len(missing)} of 28 runs missing: {missing}")
        assert PARTIAL, "runs are missing - set PARTIAL=1 to analyse only the complete models"
        continue
    models.append((fam, size))
assert models, "no model has its 28 runs"

rowsA, rowsB, rowsBs, rowsC, rowsW, rowsD, rowsE, rowsEs, rowsF, rowsF4 = [], [], [], [], [], [], [], [], [], []

for fam, size in models:
    model = f"{fam}-{size}"
    R = {c: RUNS[(fam, size, c)] for c in ALL_CIDS}
    none = R["none"]

    # ---------------------------------------------------------------- A: fit quality, full data
    for con in TRAIN:
        c = R[cid_of(con)]
        assert len(c["Y"]) == NI * (NI - 1)
        c["g_full"], c["beta"], c["r2_full"] = fit_full(c["ia"], c["ib"], c["Y"])
        c["g_feat"], c["w"], c["r2_feat"] = fit_feat(c["dz"], c["Y"])
        c["dbar"] = c["Dm"][IU, JU]
        rowsA.append(dict(model=model, contract=cid_of(con), n_rows=len(c["Y"]), r2_bt_full=c["r2_full"],
                          r2_bt_feat=c["r2_feat"], gamma_bt_full=c["g_full"], gamma_bt_feat=c["g_feat"]))
        rowsW.append(dict(model=model, contract=cid_of(con), gamma=c["g_feat"],
                          **{f"{f}={lvl}": c["w"][k] for k, (f, lvl) in enumerate(LEVELS)}))
    if (fam, size) == models[0]:                 # the same numbers as src/bt.py, checked once
        from src.bt import FeatureBT
        cols = [f"{p}_{f}" for p in "ab" for f in FEATS]
        df = pd.DataFrame({**{f"a_{f}": ITEMS[f].values[none["ia"]] for f in FEATS},
                           **{f"b_{f}": ITEMS[f].values[none["ib"]] for f in FEATS}})[cols]
        df["score_a"], df["score_b"] = none["Y"], 0.0
        fb = FeatureBT(r2_warn=None).fit(df)
        dw = max(abs(fb.weights_[f][lvl] - none["w"][k]) for k, (f, lvl) in enumerate(LEVELS))
        print(f"check against src/bt.py FeatureBT ({model}, none): max |weight diff| {dw:.1e}, "
              f"|gamma diff| {abs(fb.gamma_ - none['g_feat']):.1e}, |R2 diff| {abs(fb.r2_ - none['r2_feat']):.1e}")
        assert dw < 1e-6 and abs(fb.gamma_ - none["g_feat"]) < 1e-6

    # ---------------------------------------------------------------- B: compliance and consistency
    for cid in ALL_CIDS[1:]:
        c = R[cid]
        ok = meets(c["con"])
        use = ok[:, None] & ~ok[None, :] & np.isfinite(c["P"])           # x meets, y does not, pair was scored
        after, before = 100 * c["P"][use].mean(), 100 * none["P"][use].mean()
        tr = [t for t, k, con in COMPOUNDS if cid_of(con) == cid]
        rowsB.append(dict(model=model, contract=cid, n_requirements=len(c["con"]), track=tr[0] if tr else "",
                          pairs=int(use.sum()), compliance_raw_before=before, compliance_raw=after,
                          compliance_gain=(after - before) / (100 - before) if 100 - before >= 1 else np.nan))
    track_steps = {(track_cid(t, k - 1), track_cid(t, k)) for t in TRACKS for k in (1, 2, 3, 4)}
    for cb in ALL_CIDS:
        for ca in ALL_CIDS:
            A_, B_ = R[ca]["con"], R[cb]["con"]
            extra = {f: v for f, v in B_.items() if A_.get(f) != v}
            if len(extra) != 1 or any(B_.get(f) != v for f, v in A_.items()):
                continue                                                  # B must be A + exactly one requirement
            (nf, nv), = extra.items()
            inner = meets(extra)
            row = dict(model=model, contract_before=ca, contract_after=cb, n_requirements_after=len(B_),
                       new_feature=nf, new_level=nv, track_step=(ca, cb) in track_steps,
                       # the ONE step shown for a contract in figure 1: a single comes from no contract,
                       # a compound from the step before it on its track
                       main_step=(ca == "none") or (ca, cb) in track_steps)
            for g, sel in (("inner", inner), ("outer", ~inner)):
                row[f"consistency_{g}"] = stats.spearmanr(R[ca]["borda"][sel], R[cb]["borda"][sel])[0]
                row[f"n_{g}"] = int(sel.sum())
            rowsBs.append(row)

    # ---------------------------------------------------------------- C: shift table
    for con in SINGLES:
        c = R[cid_of(con)]
        rowsC.append(dict(model=model, contract=cid_of(con),
                          **{f"{f}={lvl}": c["w"][k] - none["w"][k] for k, (f, lvl) in enumerate(LEVELS)}))

    # ---------------------------------------------------------------- D: additivity, zero rows from the compound
    for t, k, con in COMPOUNDS:
        c = R[cid_of(con)]
        comp = [R[cid_of({f: v})] for f, v in TRACKS[t][:k]]
        _, ceil, _ = fit_full(c["ia"], c["ib"], c["Y"])                   # BT-full on the compound's own rows
        dbar = c["Dm"][SH_I, SH_J]
        assert np.isfinite(dbar).all()
        arms = {"item_additive": none["beta"] + sum(p["beta"] - none["beta"] for p in comp),
                "feature_additive": Z @ (none["w"] + sum(p["w"] - none["w"] for p in comp)),
                "ref_none": none["beta"], "ceiling": ceil}
        for (f, v), p in zip(TRACKS[t][:k], comp):
            arms[f"ref_{f}={v}"] = p["beta"]
        for arm, s in arms.items():
            acc, r2, rho = metrics(s, dbar, SH_I, SH_J, ceil)
            rowsD.append(dict(model=model, track=t, depth=k, contract=cid_of(con), arm=arm, n_pairs=len(dbar),
                              accuracy=acc, r2=r2, spearman=rho))

    # ---------------------------------------------------------------- E: efficiency
    train = [R[cid_of(con)] for con in TRAIN]
    off = np.cumsum([0] + [len(c["Y"]) for c in train])
    N = off[-1]
    with np.errstate(all="ignore"):
        for Mb in BUDGETS + [N]:
            for seed in range(N_SEEDS if Mb < N else 1):
                rng = np.random.default_rng([MODELS.index((fam, size)), 1, Mb, seed])
                pick = np.sort(rng.choice(N, Mb, replace=False))
                for q, c in enumerate(train):
                    rows = pick[(pick >= off[q]) & (pick < off[q + 1])] - off[q]
                    s_full = fit_full(c["ia"][rows], c["ib"][rows], c["Y"][rows])[1]
                    s_feat = Z @ fit_feat(c["dz"][rows], c["Y"][rows])[1]
                    for arm, s in (("bt_full", s_full), ("bt_feat", s_feat)):
                        acc, r2, rho = metrics(s, c["dbar"], IU, JU, c["beta"])
                        rowsE.append(dict(model=model, arm=arm, budget=Mb, seed=seed, contract=cid_of(TRAIN[q]),
                                          accuracy=acc, r2=r2, spearman=rho))
    E = pd.DataFrame([r for r in rowsE if r["model"] == model])
    Em = E.groupby(["arm", "budget"])[["accuracy", "r2", "spearman"]].mean().reset_index()
    target = 0.95 * Em[(Em.arm == "bt_full") & (Em.budget == N)].accuracy.iloc[0]
    for arm in ("bt_full", "bt_feat"):
        a = Em[Em.arm == arm].sort_values("budget")
        hit, full = a[a.accuracy >= target], a[a.budget == N].iloc[0]
        rowsEs.append(dict(model=model, arm=arm, n_contracts=len(train), full_pool_rows=N, acc_full=full.accuracy,
                           r2_full=full.r2, spearman_full=full.spearman, target_acc=target,
                           min_M_reaching_target=int(hit.budget.iloc[0]) if len(hit) else np.nan))

    # ---------------------------------------------------------------- F: target cars along the track
    for t in TRACKS:
        tg = ITEMS.index[(ITEMS.part == "target") & (ITEMS.track == t)].values
        assert len(tg) == 3
        for k in range(5):
            Dm = R[track_cid(t, k)]["Dm"]
            for i in tg:
                v = np.delete(Dm[i], i)
                assert np.isfinite(v).all(), "a target car has an unscored pair"
                rowsF.append(dict(model=model, track=t, step=k, contract=track_cid(t, k), car_id=int(ITEMS.id[i]),
                                  car=ITEMS.text[i], win_rate=(v > 0).mean(), ties=int((v == 0).sum()),
                                  mean_dbar=v.mean()))
        Dm = R[track_cid(t, 4)]["Dm"]
        free = [f for f in FEATS if f not in dict(TRACKS[t])]            # the two features never mentioned
        sub = Dm[np.ix_(tg, tg)]
        wins, mean = np.nansum(sub > 0, axis=1), np.nanmean(sub, axis=1)
        for rank, q in enumerate(sorted(range(3), key=lambda q: (-wins[q], -mean[q])), 1):
            i = tg[q]
            others = [p for p in range(3) if p != q]
            rowsF4.append(dict(model=model, track=t, rank=rank, car_id=int(ITEMS.id[i]),
                               **{f"unmentioned_{n + 1}": f"{f}: {ITEMS[f][i]}" for n, f in enumerate(free)},
                               wins_vs_other_two=int(wins[q]), mean_dbar_vs_other_two=mean[q],
                               other_a=int(ITEMS.id[tg[others[0]]]), dbar_vs_a=sub[q, others[0]],
                               other_b=int(ITEMS.id[tg[others[1]]]), dbar_vs_b=sub[q, others[1]]))
    print(f"{model}: done")

A, B, Bs = pd.DataFrame(rowsA), pd.DataFrame(rowsB), pd.DataFrame(rowsBs)
C, W, Dd = pd.DataFrame(rowsC), pd.DataFrame(rowsW), pd.DataFrame(rowsD)
E, Es, F, F4 = pd.DataFrame(rowsE), pd.DataFrame(rowsEs), pd.DataFrame(rowsF), pd.DataFrame(rowsF4)
Em = E.groupby(["model", "arm", "budget", "seed"], sort=False)[["accuracy", "r2", "spearman"]].mean() \
    .groupby(["model", "arm", "budget"], sort=False).agg(["mean", "std"])
Em.columns = [f"{m}_{s}" for m, s in Em.columns]
Em = Em.reset_index()
A.to_csv(f"{OUT}/A_fit_quality.csv", index=False)
B.to_csv(f"{OUT}/B_compliance.csv", index=False)
Bs.to_csv(f"{OUT}/B_consistency_steps.csv", index=False)
C.to_csv(f"{OUT}/C_shift_table.csv", index=False)
W.to_csv(f"{OUT}/C_weights.csv", index=False)
Dd.to_csv(f"{OUT}/D_additivity.csv", index=False)
E.to_csv(f"{OUT}/E_efficiency_all.csv", index=False)
Em.to_csv(f"{OUT}/E_efficiency_by_budget.csv", index=False)
Es.to_csv(f"{OUT}/E_efficiency_summary.csv", index=False)
F.to_csv(f"{OUT}/F_targets.csv", index=False)
F4.to_csv(f"{OUT}/F_targets_step4_order.csv", index=False)

# -------------------------------------------------------------------- G: cars next to laptops
# Laptop numbers come from existing outputs only:
#   figs/grum_first_test/grum_first_test.csv   part A, full pool: BT-feat R2 against Dbar, per contract
#   figs/grum_first_test/summary_partA.csv     smallest M (BT-feat reaching 95% of BT-full's full-pool accuracy)
#   figs/grum_first_test/summary_partB.csv     zero-shot on compounds (GRUM zero-shot = the feature-level sum)
#   figs/metrics/table_compliance.csv, table_consistency.csv
LG = pd.read_csv("figs/grum_first_test/grum_first_test.csv")
LA = pd.read_csv("figs/grum_first_test/summary_partA.csv")
LB = pd.read_csv("figs/grum_first_test/summary_partB.csv")
LC = pd.read_csv("figs/metrics/table_compliance.csv")
LS = pd.read_csv("figs/metrics/table_consistency.csv")
rowsG = []


def g(model, metric, cars, laptops):
    rowsG.append(dict(model=model, metric=metric, cars=cars, laptops=laptops))


for fam, size in models:
    model = f"{fam}-{size}"
    lg = LG[(LG.model == model) & (LG.part == "A") & (LG.arm == "bt_feat") & (LG.contract == "none")]
    lg = lg[lg.budget == lg.budget.max()]
    e = E[(E.model == model) & (E.arm == "bt_feat") & (E.contract == "none")]
    g(model, "BT-feat R2 vs Dbar, no contract, full data", e[e.budget == e.budget.max()].r2.iloc[0], lg.r2.iloc[0])
    g(model, "BT-feat R2 of the OLS fit on the rows, no contract",
      A[(A.model == model) & (A.contract == "none")].r2_bt_feat.iloc[0], np.nan)
    lc = LC[(LC.family == fam) & (LC.size_B == float(size)) & (LC.variant == "aligned") & (LC.n_requirements == 1)]
    ls = LS[(LS.family == fam) & (LS.size_B == float(size)) & (LS.variant == "aligned")
            & (LS.contract_before == "default")]
    b = B[(B.model == model) & (B.n_requirements == 1)]
    s = Bs[(Bs.model == model) & (Bs.contract_before == "none")]
    g(model, "single contracts, n", len(b), len(lc))
    g(model, "compliance raw, mean over singles", b.compliance_raw.mean(), lc.compliance_raw.mean())
    g(model, "compliance gain, mean over singles", b.compliance_gain.mean(), lc.compliance_gain.mean())
    g(model, "consistency inner, mean over singles", s.consistency_inner.mean(), ls.consistency_inner.mean())
    g(model, "consistency outer, mean over singles", s.consistency_outer.mean(), ls.consistency_outer.mean())
    lb = LB[LB.model == model].assign(depth=lambda x: x.contract.str.count("=") )
    d = Dd[Dd.model == model]
    for k in (2, 3, 4):
        lk = lb[lb.depth == k]
        g(model, f"compounds with {k} requirements, n", int((d[d.arm == "ceiling"].depth == k).sum()), len(lk))
        for arm, name in (("feature_additive", "feature-level"), ("item_additive", "item-level")):
            dk = d[(d.depth == k) & (d.arm == arm)]
            lap = lk if arm == "feature_additive" else lk.iloc[:0]        # laptops: the feature-level sum only
            g(model, f"zero-shot accuracy, {k} requirements, {name}", dk.accuracy.mean(), lap.zero_acc.mean())
            g(model, f"zero-shot R2, {k} requirements, {name}", dk.r2.mean(), lap.zero_r2.mean())
    es = Es[(Es.model == model) & (Es.arm == "bt_feat")].iloc[0]
    la = LA[(LA.model == model) & (LA.arm == "bt_feat")].iloc[0]
    g(model, "train contracts in the pool, n", es.n_contracts, la.n_contracts)
    g(model, "rows in the full pool", es.full_pool_rows, la.full_pool_rows)
    g(model, "BT-full accuracy on the full pool", Es[(Es.model == model) & (Es.arm == "bt_full")].acc_full.iloc[0],
      LA[(LA.model == model) & (LA.arm == "bt_full")].acc_full.iloc[0])
    g(model, "smallest M: BT-feat reaches 95% of BT-full full-pool accuracy", es.min_M_reaching_target,
      la.min_M_reaching_target)
G = pd.DataFrame(rowsG)
G.to_csv(f"{OUT}/G_cars_vs_laptops.csv", index=False)

# -------------------------------------------------------------------- print
pd.set_option("display.width", 250, "display.max_rows", 600, "display.max_columns", 60, "display.max_colwidth", 70)
fmt = lambda v: f"{v:.3f}"
print("\n=== A. fit quality on the full data of each train contract (R2 of the OLS fit on the rows; gamma)")
print(A.to_string(index=False, float_format=fmt))
print("\n=== B. compliance (raw in %, gain as a share), 27 contracts")
print(B.to_string(index=False, float_format=fmt))
print("\n=== B. consistency (Spearman of Borda scores), every step A -> A + one requirement")
print(Bs.to_string(index=False, float_format=fmt))
print("\n=== C. shift table: w_p - w_none (BT-feat), rows = single contract, columns = 24 coordinates")
for m, x in C.groupby("model", sort=False):
    print(m)
    print(x.drop(columns="model").set_index("contract").T.to_string(float_format=lambda v: f"{v:.2f}"))
print("\n=== D. additivity on the 500 shared pairs (accuracy / R2 / Spearman)")
print(Dd.drop(columns="n_pairs").to_string(index=False, float_format=fmt))
print("\n=== E. efficiency: mean over the 16 train contracts, mean and sd over seeds")
print(Em.to_string(index=False, float_format=fmt))
print(Es.to_string(index=False, float_format=fmt))
print("\n=== F. target cars: win rate against the other 119 cars and mean Dbar, by step")
print(F.drop(columns=["contract", "car"]).to_string(index=False, float_format=fmt))
print("\n=== F. step 4: order of the three targets among themselves")
print(F4.to_string(index=False, float_format=fmt, na_rep=""))
print("\n=== G. cars next to laptops")
for m, x in G.groupby("model", sort=False):
    print(m)
    print(x.drop(columns="model").to_string(index=False, float_format=fmt, na_rep="-"))

# -------------------------------------------------------------------- figures
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2})
MN = [f"{f}-{s}" for f, s in models]


def panels(nrows=1, w=4.6, h=3.6, **kw):
    fig, ax = plt.subplots(nrows, len(MN), figsize=(w * len(MN), h * nrows), squeeze=False, **kw)
    return fig, ax


# 1. compliance and consistency against the number of requirements
fig, ax = panels(sharey=True)
SER = [("compliance_raw", "compliance, raw (share)", BLUE), ("compliance_gain", "compliance, gain", ORANGE),
       ("consistency_inner", "consistency, inner", AQUA), ("consistency_outer", "consistency, outer", YELLOW)]
for j, m in enumerate(MN):
    b = B[B.model == m].copy()
    b["compliance_raw"] /= 100
    s = Bs[(Bs.model == m) & Bs.main_step].rename(columns={"contract_after": "contract"})
    t = b.merge(s[["contract", "consistency_inner", "consistency_outer"]], on="contract")
    assert len(t) == 27
    for q, (col, lab, colr) in enumerate(SER):
        x = t.n_requirements + (q - 1.5) * 0.14
        ax[0, j].plot(x, t[col], "o", ms=4, color=colr, alpha=0.45, mec="none")
        mu = t.groupby("n_requirements")[col].mean()
        ax[0, j].plot(mu.index + (q - 1.5) * 0.14, mu, "o-", ms=7, color=colr, mec=SURF, mew=1.5, label=lab)
    ax[0, j].set(title=m, xlabel="number of requirements in the contract", xticks=[1, 2, 3, 4])
ax[0, 0].set_ylabel("value")
fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8)
fig.suptitle("Compliance and consistency by number of requirements: one small dot per contract, "
             "large dot = mean of that depth", fontsize=10)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(f"{OUT}/fig1_compliance_consistency.png", dpi=150)
plt.close(fig)

# 2. heatmap of the shift table
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#e66767", "#a82424"])
fig, ax = plt.subplots(1, len(MN), figsize=(7.4 * len(MN), 6.4), squeeze=False)
for j, m in enumerate(MN):
    c = C[C.model == m].set_index("contract").drop(columns="model")
    v = np.abs(c.values).max()
    im = ax[0, j].imshow(c.values, cmap=DIV, vmin=-v, vmax=v, aspect="auto")
    ax[0, j].set_xticks(range(D), [lvl for _, lvl in LEVELS], rotation=90, fontsize=7)
    ax[0, j].set_yticks(range(len(c)), c.index if j == 0 else [], fontsize=7)
    ax[0, j].grid(False)
    edge = np.cumsum([len(LEVELS_OF[f]) for f in FEATS])[:-1] - 0.5
    for e_ in edge:
        ax[0, j].axvline(e_, color=SURF, lw=2)
    for i, cid in enumerate(c.index):                        # a frame on the coordinate the contract asks for
        k = list(c.columns).index(cid)
        ax[0, j].add_patch(plt.Rectangle((k - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
    for n, f in enumerate(FEATS):
        ax[0, j].text(np.mean(FIDX[f]), -0.7 - 0.75 * (n % 2), f, fontsize=7, color=INK2, ha="center", va="bottom")
    ax[0, j].set_title(m, pad=38)
    fig.colorbar(im, ax=ax[0, j], fraction=0.035, pad=0.02, label="w under the contract - w with no contract")
fig.suptitle("Shift of the BT-feat weights under each single contract (framed cell = the level asked for)",
             fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_shift_heatmap.png", dpi=150)
plt.close(fig)

# 3. additivity by depth
fig, ax = panels(nrows=2, h=3.3, sharex=True)
for j, m in enumerate(MN):
    d = Dd[Dd.model == m]
    comp = d[d.arm.str.startswith("ref_") & (d.arm != "ref_none")].groupby(["track", "depth"])[["accuracy", "r2"]].mean()
    for i, met in enumerate(("accuracy", "r2")):
        a = ax[i, j]
        for arm, lab, colr, off in (("item_additive", "item-level sum", BLUE, -0.07),
                                    ("feature_additive", "feature-level sum", ORANGE, 0.07)):
            x = d[d.arm == arm]
            a.plot(x.depth + off, x[met], "o", ms=4, color=colr, alpha=0.45, mec="none")
            mu = x.groupby("depth")[met].mean()
            a.plot(mu.index + off, mu, "o-", ms=7, color=colr, mec=SURF, mew=1.5, label=lab)
        for arm, lab, ls, colr in (("ceiling", "ceiling: BT-full on the compound's own rows", "-", INK),
                                   ("ref_none", "reference: no-contract scores", "--", MUTED)):
            mu = d[d.arm == arm].groupby("depth")[met].mean()
            a.plot(mu.index, mu, ls, color=colr, lw=1.3, label=lab)
        mu = comp.groupby("depth")[met].mean()
        a.plot(mu.index, mu, ":", color=MUTED, lw=1.6, label="reference: one single contract (mean over its components)")
        a.set(xticks=[2, 3, 4], ylabel="accuracy" if met == "accuracy" else "R² (axis cut at −1)")
        if met == "r2":
            a.set_ylim(-1.02, 1.02)
            a.set_xlabel("number of requirements in the compound")
        else:
            a.set_title(m)
fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8)
fig.suptitle("Zero-shot prediction of a compound contract from its single contracts, on the 500 shared pairs "
             "(small dot = one compound, line = mean over the 4 tracks)", fontsize=10)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(f"{OUT}/fig3_additivity.png", dpi=150)
plt.close(fig)

# 4. efficiency
fig, ax = panels(sharey=True)
for j, m in enumerate(MN):
    e = Em[Em.model == m]
    for arm, lab, colr in (("bt_full", "BT-full", BLUE), ("bt_feat", "BT-feat", ORANGE)):
        x = e[e.arm == arm].sort_values("budget")
        sd = x.accuracy_std.fillna(0)
        ax[0, j].fill_between(x.budget, x.accuracy_mean - sd, x.accuracy_mean + sd, color=colr, alpha=0.2, lw=0)
        ax[0, j].plot(x.budget, x.accuracy_mean, "o-", ms=5, color=colr, mec=SURF, mew=1, label=lab)
    ax[0, j].set(xscale="log", title=m, xlabel="M = rows drawn from the pool of the 16 train contracts")
ax[0, 0].set_ylabel("accuracy (mean over the 16 contracts)")
ax[0, 0].legend(loc="lower right")
fig.suptitle(f"Efficiency: accuracy against the number of rows, mean ± sd over {N_SEEDS} seeds", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_efficiency.png", dpi=150)
plt.close(fig)

# 5. target cars along the tracks
fig, ax = panels(sharey=True)
for j, m in enumerate(MN):
    f = F[F.model == m]
    for q, (t, colr) in enumerate(zip(TRACKS, (BLUE, ORANGE, AQUA, YELLOW))):
        x = f[f.track == t]
        off = (q - 1.5) * 0.06
        ax[0, j].plot(x.step + off, x.win_rate, "o", ms=4.5, color=colr, alpha=0.55, mec="none")
        mu = x.groupby("step").win_rate.mean()
        ax[0, j].plot(mu.index + off, mu, "-", color=colr, label=t)
    ax[0, j].set(title=m, xlabel="step of the track (0 = no contract, 4 = all four requirements)",
                 xticks=range(5))
ax[0, 0].set_ylabel("win rate of a target car against the other 119 cars")
ax[0, 0].legend(title="track", loc="lower right")
fig.suptitle("Target cars along their track: one dot per target car (3 per track), line = their mean", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_tracks.png", dpi=150)
plt.close(fig)
print(f"\nwrote 11 CSV files and 5 figures to {OUT}/")
