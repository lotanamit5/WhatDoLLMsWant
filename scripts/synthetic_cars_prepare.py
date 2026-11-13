"""Synthetic cars, stage 1: build the 120 cars, the 28 contracts and the pair files.

Standalone, no GPU. Run from the repo root:
    ~/miniconda3/envs/whatdo-llms-want/bin/python scripts/synthetic_cars_prepare.py
Writes data/synthetic_cars/design/:
    items.csv              the 120 cars (id, part, track, the 6 features, text)
    contracts.csv          the 28 contracts with how many cars satisfy each
    balance.csv            min / max cell count for every pair of features
    compound_pairs.csv     the 500 random pairs shared by all 12 compound contracts
    pairs_<track>.csv      what a compound job of that track scores: the 500 + every pair with a target car
    sanity_pairs.csv       200 random pairs for the template check
Everything is drawn with one fixed seed, so re-running gives the same files.

The design. 12 target cars are fixed first (3 per track, all four requirements met, different on the
two features the track does not mention). The other 108 are found by a swap search: every feature
column always holds exactly its balanced counts (a move swaps one feature between two non-target cars),
and the search minimises  sum over the 15 feature pairs of sum over cells (count - 120 / n_cells)^2,
with duplicate cars forbidden. LOWER_BOUND below is the smallest value any 120-car design can reach.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import data_collection as dc           # only for the contract sentence; no torch is imported

DESIGN = "data/synthetic_cars/design"
SEED = 20261113
N = 120
LEVELS = {
    "company": ["Honda", "Toyota", "Ford", "BMW", "Audi", "Mercedes"],
    "color": ["white", "black", "silver", "red", "blue", "grey"],
    "fuel": ["petrol", "diesel"],
    "transmission": ["manual", "automatic"],
    "ownership": ["private first owner", "private second owner", "private third owner", "ex-lease", "ex-rental"],
    "horsepower": ["100 hp", "150 hp", "200 hp"],
}
FEATURES = list(LEVELS)
TEXT_ORDER = ["company", "color", "fuel", "transmission", "horsepower", "ownership"]   # the line shown
TRACKS = {          # one buyer per track; one more requirement per step
    "luxury": [("transmission", "automatic"), ("company", "BMW"), ("horsepower", "200 hp"), ("color", "black")],
    "economy": [("fuel", "petrol"), ("transmission", "manual"), ("horsepower", "100 hp"), ("company", "Honda")],
    "family": [("transmission", "automatic"), ("ownership", "private first owner"), ("company", "Toyota"),
               ("color", "white")],
    "work": [("fuel", "diesel"), ("horsepower", "150 hp"), ("company", "Ford"), ("ownership", "ex-lease")],
}
N_TARGETS, N_COMPOUND_PAIRS, N_SANITY_PAIRS = 3, 500, 200
PAIRS = list(itertools.combinations(range(len(FEATURES)), 2))
os.makedirs(DESIGN, exist_ok=True)
rng = np.random.default_rng(SEED)

# ---- 1. the 12 target cars
targets, target_track = [], []
for track, reqs in TRACKS.items():
    fixed = dict(reqs)
    free = {}
    for f in FEATURES:
        if f in fixed:
            continue
        lv = LEVELS[f]
        if len(lv) >= N_TARGETS:                      # three different levels
            free[f] = list(rng.choice(lv, N_TARGETS, replace=False))
        else:                                         # two levels: one of them twice
            twice = rng.integers(2)
            free[f] = list(rng.permutation([lv[twice], lv[twice], lv[1 - twice]]))
    for k in range(N_TARGETS):
        targets.append([LEVELS[f].index(fixed[f] if f in fixed else free[f][k]) for f in FEATURES])
        target_track.append(track)
targets = np.array(targets)
NT = len(targets)

# ---- 2. the other 108: swap search under exact one-feature balance
ideal = {p: N / (len(LEVELS[FEATURES[p[0]]]) * len(LEVELS[FEATURES[p[1]]])) for p in PAIRS}


def lower_bound(p):
    """Smallest possible sum of squared deviations of a table with equal row and column sums:
    every cell is the ideal rounded down or up."""
    cells = len(LEVELS[FEATURES[p[0]]]) * len(LEVELS[FEATURES[p[1]]])
    lo = int(np.floor(ideal[p]))
    n_up = N - lo * cells
    return n_up * (lo + 1 - ideal[p]) ** 2 + (cells - n_up) * (lo - ideal[p]) ** 2


LOWER_BOUND = sum(lower_bound(p) for p in PAIRS)


def tables(X):
    out = {}
    for i, j in PAIRS:
        t = np.zeros((len(LEVELS[FEATURES[i]]), len(LEVELS[FEATURES[j]])))
        np.add.at(t, (X[:, i], X[:, j]), 1)
        out[(i, j)] = t
    return out


def cost(X):
    return sum(((t - ideal[p]) ** 2).sum() for p, t in tables(X).items())


def n_dup(X):
    return len(X) - len({tuple(r) for r in X})


def search(rng, n_iter=400_000, t0=1.0, t1=0.01):
    X = np.zeros((N, len(FEATURES)), dtype=int)
    X[:NT] = targets
    for k, f in enumerate(FEATURES):                  # fill each column with what is left of its quota
        left = np.full(len(LEVELS[f]), N // len(LEVELS[f])) - np.bincount(targets[:, k], minlength=len(LEVELS[f]))
        X[NT:, k] = rng.permutation(np.repeat(np.arange(len(LEVELS[f])), left))
    tab = tables(X)
    seen = {}
    for r in X:
        seen[tuple(r)] = seen.get(tuple(r), 0) + 1
    cur = cost(X) + 1000 * n_dup(X)
    best, best_X = cur, X.copy()
    for it in range(n_iter):
        temp = t0 * (t1 / t0) ** (it / n_iter)
        k = rng.integers(len(FEATURES))
        r1, r2 = rng.integers(NT, N, 2)
        a, b = X[r1, k], X[r2, k]
        if a == b:
            continue
        delta = 0.0
        touched = []
        for p in PAIRS:
            if k not in p:
                continue
            o = p[1] if p[0] == k else p[0]
            t = tab[p]
            for row, old, new in ((r1, a, b), (r2, b, a)):
                c_old = (old, X[row, o]) if p[0] == k else (X[row, o], old)
                c_new = (new, X[row, o]) if p[0] == k else (X[row, o], new)
                delta += 1 - 2 * (t[c_old] - ideal[p])          # (x-1-q)^2 - (x-q)^2
                t[c_old] -= 1
                delta += 1 + 2 * (t[c_new] - ideal[p])          # (x+1-q)^2 - (x-q)^2
                t[c_new] += 1
                touched.append((t, c_old, c_new))
        old1, old2 = tuple(X[r1]), tuple(X[r2])
        X[r1, k], X[r2, k] = b, a
        new1, new2 = tuple(X[r1]), tuple(X[r2])
        d0 = sum(v - 1 for v in seen.values())
        for key in (old1, old2):
            seen[key] -= 1
            if not seen[key]:
                del seen[key]
        for key in (new1, new2):
            seen[key] = seen.get(key, 0) + 1
        delta += 1000 * (sum(v - 1 for v in seen.values()) - d0)
        if delta <= 0 or rng.random() < np.exp(-delta / temp):
            cur += delta
            if cur < best - 1e-9:
                best, best_X = cur, X.copy()
                if best <= LOWER_BOUND + 1e-9:
                    break
        else:                                          # undo
            for key in (new1, new2):
                seen[key] -= 1
                if not seen[key]:
                    del seen[key]
            for key in (old1, old2):
                seen[key] = seen.get(key, 0) + 1
            X[r1, k], X[r2, k] = a, b
            for t, c_old, c_new in touched:
                t[c_old] += 1
                t[c_new] -= 1
    return best, best_X


best, X = None, None
for restart in range(20):
    c, cand = search(rng)
    print(f"restart {restart}: cost {c:.3f} (lower bound {LOWER_BOUND:.3f})")
    if best is None or c < best:
        best, X = c, cand
    if best <= LOWER_BOUND + 1e-9:
        break
assert n_dup(X) == 0 and abs(cost(X) - best) < 1e-6
assert (X[:NT] == targets).all()

# rows: the 12 targets first, then the other 108 in a fixed shuffled order; id = 1..120
order = list(range(NT)) + list(NT + rng.permutation(N - NT))
items = pd.DataFrame({f: [LEVELS[f][v] for v in X[order, k]] for k, f in enumerate(FEATURES)})
items.insert(0, "track", target_track + [""] * (N - NT))
items.insert(0, "part", ["target"] * NT + ["rest"] * (N - NT))
items.insert(0, "id", np.arange(1, N + 1))
items["text"] = [", ".join(r[f] for f in TEXT_ORDER) for _, r in items.iterrows()]
assert items.text.is_unique
for f in FEATURES:
    assert (items[f].value_counts() == N // len(LEVELS[f])).all(), f
items.to_csv(f"{DESIGN}/items.csv", index=False)


def satisfies(con):
    return np.all([items[f] == v for f, v in con.items()], axis=0) if con else np.ones(N, bool)


# ---- 3. the 28 contracts
contracts = [dict(spec="", depth=0, set="train", track="", step=0, con={})]
for track, reqs in TRACKS.items():                    # singles, in the order they first appear
    for f, v in reqs:
        if not any(c["con"] == {f: v} for c in contracts):
            contracts.append(dict(spec=f"{f}={v}", depth=1, set="train", track="", step=1, con={f: v}))
for track, reqs in TRACKS.items():                    # compounds: the first 2, 3, 4 requirements
    for k in (2, 3, 4):
        contracts.append(dict(spec=",".join(f"{f}={v}" for f, v in reqs[:k]), depth=k, set="compound",
                              track=track, step=k, con=dict(reqs[:k])))
for c in contracts:
    con = c.pop("con")
    dc.validate_constraints(con, items.to_dict("records"))
    c["constraints_id"] = dc.constraints_id(con)
    c["n_items_satisfying"] = int(satisfies(con).sum())
    c["text"] = dc.constraints_text(con)
contracts = pd.DataFrame(contracts)[["spec", "constraints_id", "depth", "set", "track", "step",
                                     "n_items_satisfying", "text"]]
assert len(contracts) == 28 and contracts.constraints_id.is_unique
for track, reqs in TRACKS.items():                    # the 3 targets satisfy their own full contract
    assert satisfies(dict(reqs))[(items.track == track).values].all()
contracts.to_csv(f"{DESIGN}/contracts.csv", index=False)

# ---- 4. pair files (unordered, by item id; the collector asks both orders)
pairs = np.array(list(itertools.combinations(items.id.values, 2)))
assert len(pairs) == 7140
shared = pairs[np.sort(rng.choice(len(pairs), N_COMPOUND_PAIRS, replace=False))]
sanity = pairs[np.sort(rng.choice(len(pairs), N_SANITY_PAIRS, replace=False))]
pd.DataFrame(shared, columns=["id_x", "id_y"]).to_csv(f"{DESIGN}/compound_pairs.csv", index=False)
pd.DataFrame(sanity, columns=["id_x", "id_y"]).to_csv(f"{DESIGN}/sanity_pairs.csv", index=False)
n_track_pairs = {}
for track in TRACKS:
    t_ids = set(items.id[items.track == track])
    with_target = np.array([p for p in pairs if p[0] in t_ids or p[1] in t_ids])
    assert len(with_target) == 354
    union = np.unique(np.vstack([shared, with_target]), axis=0)
    n_track_pairs[track] = len(union)
    pd.DataFrame(union, columns=["id_x", "id_y"]).to_csv(f"{DESIGN}/pairs_{track}.csv", index=False)

# ---- 5. report
pd.set_option("display.width", 250, "display.max_rows", 300, "display.max_colwidth", 200)
print(f"\nitems: {N} ({NT} target + {N - NT} rest), seed {SEED}, no duplicates; "
      f"pair-balance cost {best:.3f}, lower bound {LOWER_BOUND:.3f}")
print("\ncount per level:")
for f in FEATURES:
    print(f"  {f:13}", dict(items[f].value_counts().reindex(LEVELS[f])))
bal = []
for i, j in PAIRS:
    t = pd.crosstab(items[FEATURES[i]], items[FEATURES[j]]).values
    bal.append(dict(feature_1=FEATURES[i], feature_2=FEATURES[j], n_cells=t.size, ideal=round(N / t.size, 2),
                    min_count=int(t.min()), max_count=int(t.max())))
bal = pd.DataFrame(bal)
bal.to_csv(f"{DESIGN}/balance.csv", index=False)
print("\ncell counts for every pair of features:")
print(bal.to_string(index=False))
onehot = pd.get_dummies(items[FEATURES]).astype(float)
corr = onehot.corr().values
owner = np.array([c.split("_")[0] for c in onehot.columns])
other = owner[:, None] != owner[None, :]
worst = np.unravel_index(np.argmax(np.abs(corr) * other), corr.shape)
print(f"\nlargest |correlation| between one-hot coordinates of different features: "
      f"{np.abs(corr[other]).max():.4f}  ({onehot.columns[worst[0]]} vs {onehot.columns[worst[1]]})")
print("\ncontracts:")
print(contracts.drop(columns="spec").to_string(index=False))
n_train = (contracts.set == "train").sum()
total = n_train * len(pairs) * 2 + sum(3 * 2 * n for n in n_track_pairs.values())
print(f"\npairs per compound job (500 shared + 354 with a target, overlap removed): {n_track_pairs}")
print(f"prompts per model: {n_train} x {len(pairs) * 2} + " +
      " + ".join(f"3 x {2 * n}" for n in n_track_pairs.values()) + f" = {total}")
print("\nthe 120 cars:")
for r in items.itertuples():
    print(f"{r.id:3d} {'[TARGET ' + r.track + ']' if r.part == 'target' else '':17} {r.text}")
