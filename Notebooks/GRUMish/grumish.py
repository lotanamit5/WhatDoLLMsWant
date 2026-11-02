"""Shared code for the GRUMish notebooks (laptops only).

Notebooks here run from the repo root (first cell walks up), then `sys.path.insert(0, "Notebooks/GRUMish")`
and `from grumish import *`. Kept in this folder, not in src/, because src/ is read-only.

Everything is built on the position-corrected margin
    D(x, y) = (m(x, y) - m(y, x)) / 2,      m = score_a - score_b,
one value per unordered pair and template. D does not contain the slot bias, nor the PMI constant
of the old qwen/gemma runs, so no gamma and no C appear anywhere below.
"""
import os, json, itertools
import numpy as np
import pandas as pd

BRANDS = ["ASUS", "Apple", "Dell", "HP", "Lenovo"]
SCREENS = ["13-inch", "14-inch", "16-inch"]
RAMS = ["4GB", "8GB", "16GB"]
FEATS = ["brand", "screen", "ram"]
LEVELS = {"brand": BRANDS, "screen": SCREENS, "ram": RAMS}
LAPTOPS = [(b, s, r) for b in BRANDS for s in SCREENS for r in RAMS]
IDX = {l: i for i, l in enumerate(LAPTOPS)}
N = len(LAPTOPS)
LV = [(f, l) for f in FEATS for l in LEVELS[f]]                  # the 11 (feature, level) slots

BASES = ["data/laptops_robustness", "data/laptops_robustness_gemma", "data/laptops_olmo"]
MODELS = ([f"qwen-{s}B" for s in ["0.5", "7", "32", "72"]] + [f"gemma-{s}B" for s in ["1", "4", "12", "27"]]
          + [f"olmo-{s}B" for s in ["1", "7", "13", "32"]])
REAL = [m for m in MODELS if m != "qwen-0.5B"]                  # qwen-0.5B: noise floor (status.md, point 12)
FAMILY = {m: m.split("-")[0] for m in MODELS}


def parse(cid):
    """'ram=8GB+screen=14-inch' -> {'ram': '8GB', 'screen': '14-inch'}; 'none' -> {}"""
    return {} if cid == "none" else dict(p.split("=") for p in cid.split("+"))


def short(cid):
    """'ram=8GB+screen=14-inch' -> '14in + 8GB' (always brand, screen, ram order)"""
    c = parse(cid)
    return " + ".join(c[f].replace("-inch", "in") for f in FEATS if f in c) or "none"


def lap_name(l):
    return f"{l[0]} {l[1].replace('-inch', 'in')} {l[2]}"


# ---------------------------------------------------------------- loading
def load_runs(models=MODELS):
    """(model, constraints_id) -> scores DataFrame. Frame `shopping`, complete runs only (9900 rows).
    A key with two complete runs is an error, not something to pick from."""
    R = {}
    for base in BASES:
        for run in sorted(os.listdir(base)):
            cf, sc = f"{base}/{run}/config.json", f"{base}/{run}/scores.csv"
            if not (os.path.isfile(cf) and os.path.isfile(sc)):        # incomplete run: skip
                continue
            c = json.load(open(cf))
            m = f"{c['model_family']}-{c['model_size']}B"
            if c["frame"]["name"] != "shopping" or m not in models:
                continue
            d = pd.read_csv(sc)
            if len(d) != 9900:
                continue
            k = (m, c["constraints_id"])
            assert k not in R, f"two complete runs for {k}"
            R[k] = d
    return R


def pair_D(d):
    """D[t, i, j] for every template t and ordered pair (i, j) of laptop indices; D[t, j, i] = -D[t, i, j]."""
    ia = np.array([IDX[x] for x in zip(d.a_brand, d.a_screen, d.a_ram)])
    ib = np.array([IDX[x] for x in zip(d.b_brand, d.b_screen, d.b_ram)])
    T = sorted(d.template_idx.unique())
    M = np.full((len(T), N, N), np.nan)
    M[[T.index(t) for t in d.template_idx], ia, ib] = (d.score_a - d.score_b).values
    D = (M - M.transpose(0, 2, 1)) / 2
    for t in range(len(T)):
        np.fill_diagonal(D[t], 0.0)
    return D


# ---------------------------------------------------------------- utilities and weights
def utility(D):
    """Per-laptop utility (45 numbers, mean 0) = the full parametrization. On this complete, balanced
    design the OLS per-item BT utility is exactly the Borda score x (N-1)/N (progress 2026-10-07)."""
    return D.sum(-1) / N            # works for D[t] (one template) and for D (all templates) alike


# pairs that differ in ONE feature only, per feature: the "ceteris paribus" (CP) pairs
def _cp_pairs(f):
    others = [g for g in FEATS if g != f]
    k = FEATS.index(f)
    out = []
    for i, j in itertools.combinations(range(N), 2):
        a, b = LAPTOPS[i], LAPTOPS[j]
        if a[k] != b[k] and all(a[FEATS.index(g)] == b[FEATS.index(g)] for g in others):
            out.append((i, j))
    return np.array(out)


CP_PAIRS = {f: _cp_pairs(f) for f in FEATS}           # brand 90, screen 45, ram 45


def cp_weights(D2):
    """Ceteris-paribus weights from a template-averaged D (45 x 45): for each feature, the level
    weights (mean 0) that best explain D on the pairs that differ in that feature ONLY.
    This is the lexicographic-safe measure: brand is measured where brand is the only difference
    (grum_formalization.md section 10). Returns a Series indexed by (feature, level)."""
    out = {}
    for f in FEATS:
        L, k = LEVELS[f], FEATS.index(f)
        P = CP_PAIRS[f]
        X = np.zeros((len(P), len(L)))
        for r, (i, j) in enumerate(P):
            X[r, L.index(LAPTOPS[i][k])] += 1
            X[r, L.index(LAPTOPS[j][k])] -= 1
        w = np.linalg.lstsq(X, D2[P[:, 0], P[:, 1]], rcond=None)[0]
        w -= w.mean()
        out.update({(f, l): v for l, v in zip(L, w)})
    return pd.Series(out)


def additive_weights(u):
    """Project a 45-utility on the additive (sum of features) space: w(f, l) = mean utility of the laptops
    with level l, minus the overall mean. Exactly the FeatureBT / OLS fit on all pairs for this design."""
    u = pd.Series(np.asarray(u), index=pd.MultiIndex.from_tuples(LAPTOPS, names=FEATS))
    return pd.Series({(f, l): u.xs(l, level=f).mean() - u.mean() for f, l in LV})


def additive_utility(w):
    """45-utility of a weight vector w (indexed by (feature, level))."""
    return np.array([sum(w[(f, l[k])] for k, f in enumerate(FEATS)) for l in LAPTOPS])


def meets(cid):
    """Boolean mask over the 45 laptops: meets every requirement of the contract."""
    c = parse(cid)
    return np.array([all(l[FEATS.index(f)] == v for f, v in c.items()) for l in LAPTOPS])


class Run:
    """Everything we need from one (model, contract) run."""
    def __init__(self, d):
        self.Dt = pair_D(d)                          # templates x 45 x 45
        self.D = self.Dt.mean(0)                     # template-averaged
        self.u = utility(self.D)                     # full parametrization
        self.u_t = np.array([utility(x) for x in self.Dt])
        # MAIN weights (2026-11-03): the all-pairs additive fit = FeatureBT shape, least squares on D.
        # It predicts choices better than CP in every task (oracle 0.947 vs 0.922; see 01 §2).
        self.w = additive_weights(self.u)
        self.w_t = pd.DataFrame({t: additive_weights(x) for t, x in enumerate(self.u_t)})
        # ceteris-paribus weights: descriptive only ("how much does it like Apple when all else is equal")
        self.wcp = cp_weights(self.D)
        self.wcp_t = pd.DataFrame({t: cp_weights(x) for t, x in enumerate(self.Dt)})


def load_all(models=MODELS):
    return {k: Run(d) for k, d in load_runs(models).items()}


# ---------------------------------------------------------------- comparing a prediction with a run
IU = np.triu_indices(N, 1)


def pair_accuracy(u_pred, D_obs):
    """Share of the 990 laptop pairs where the predicted utility picks the same winner as the observed D."""
    dp = u_pred[:, None] - u_pred[None, :]
    a, b = dp[IU], D_obs[IU]
    keep = b != 0
    return float((np.sign(a[keep]) == np.sign(b[keep])).mean())


def pair_r2(u_pred, D_obs):
    """Continuous version of pair_accuracy: R^2 of the predicted margin u(x) - u(y) against the observed D."""
    dp = u_pred[:, None] - u_pred[None, :]
    return r2(dp[IU], D_obs[IU])


def r2(pred, obs):
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    return float(1 - ((obs - pred) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())


# ---------------------------------------------------------------- plotting style (dataviz palette)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
GRAY, LGRAY, INK, INK2, GRID = "#a3a29d", "#d6d5d1", "#0b0b0b", "#52514e", "#e5e4df"
FAMC = {"qwen": BLUE, "gemma": ORANGE, "olmo": AQUA}             # one hue per family
SIZE_MK = {0: "o", 1: "s", 2: "D", 3: "^"}                        # marker by size rank within family
FEATC = {"brand": "#6aa83f", "screen": "#c75f8b", "ram": "#4c8fd4"}  # same muted colours as figs4nir2


def model_style(m):
    fam = FAMILY[m]
    sizes = [x for x in MODELS if FAMILY[x] == fam]
    return dict(color=FAMC[fam], marker=SIZE_MK[sizes.index(m)])


def setup_plots():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
                         "axes.titlesize": 10.5, "figure.facecolor": "white", "axes.axisbelow": True})


# ---------------------------------------------------------------- the structured GRUM (the inductive bias)
def e_vec(f, l):
    """Centred indicator of level l of feature f: +1 - 1/L on l, -1/L elsewhere (sums to 0, like the weights)."""
    e = pd.Series(-1 / len(LEVELS[f]), index=LEVELS[f]); e[l] += 1
    return e


ORDERED = {"screen", "ram"}            # brand has no order


def d_vec(f, l):
    """Ideal-point shape: minus the distance (in level steps) from the requested level l, centred.
    For the middle level it equals e_vec; for an end level it also ranks the far end last (13 > 14 > 16)."""
    L = LEVELS[f]
    d = pd.Series([-abs(L.index(x) - L.index(l)) for x in L], index=L, dtype=float)
    return d - d.mean()


def push_vec(f, l, shape):
    """'peak' = push the requested level only; 'ideal' = ideal point (distance) for ordered features."""
    return d_vec(f, l) if (shape == "ideal" and f in ORDERED) else e_vec(f, l)


def fit_struct(a, singles, self_shrink=True, shape="ideal"):
    """Fit the structured GRUM of ONE model from its single-requirement runs.
    a       : CP weights with no contract
    singles : list of (feature, level, CP weights under `feature=level`)
    Named feature f:   w_f = (1 - sigma_f) a_f + rho_f e_l      (sigma: old taste flattened, rho: push to l)
    Unnamed feature g: w_g = (1 - kappa_fg) a_g                 (kappa: leakage, as shrinkage)
    One sigma and one rho per feature and one kappa per ordered pair, shared by all levels.
    shape: 'ideal' (default) pushes screen/RAM as an ideal point, 'peak' pushes the requested level only."""
    P = {"sigma": {}, "rho": {}, "kappa": {}}
    for f in FEATS:
        tr = [(l, w) for (g, l, w) in singles if g == f]
        if not tr:
            continue
        y = np.concatenate([(w[f] - a[f]).values for l, w in tr])
        Xr = np.concatenate([push_vec(f, l, shape).values for l, w in tr])
        if self_shrink:
            Xs = np.concatenate([-a[f].values for l, w in tr])
            P["sigma"][f], P["rho"][f] = np.linalg.lstsq(np.c_[Xs, Xr], y, rcond=None)[0]
        else:
            P["sigma"][f], P["rho"][f] = 0.0, float(Xr @ y / (Xr @ Xr))
        for g in FEATS:
            if g != f:
                yg = np.concatenate([w[g].values for l, w in tr]); xg = np.concatenate([a[g].values for l, w in tr])
                P["kappa"][(f, g)] = 1 - float(xg @ yg / (xg @ xg))
    return P


def predict_struct(a, P, con, lam=1.0, shape="ideal"):
    """CP weights under contract `con` ({feature: level}) from the structured GRUM. Shrinkage from several named
    features multiplies; the push of each named feature is diluted by lam**(q-1), q = number of requirements."""
    q, w = len(con), a.copy()
    for g in FEATS:
        keep = np.prod([1 - P["kappa"][(f, g)] for f in con if f != g])
        if g in con:
            w[g] = ((1 - P["sigma"][g]) * keep * a[g] + lam ** (q - 1) * P["rho"][g] * push_vec(g, con[g], shape)).values
        else:
            w[g] = (keep * a[g]).values
    return w


def predict_sum(a, single_w, con, lam=1.0):
    """GRUM with free B rows ('sum of singles'): w = a + lam**(q-1) * sum of the observed single shifts."""
    return a + lam ** (len(con) - 1) * sum(single_w[(f, l)] - a for f, l in con.items())
