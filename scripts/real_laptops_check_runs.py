"""Real laptops: is the main batch (jobs 1498397-1498402, 2026-11-13) complete and sane?

Run from the repo root:
    ~/miniconda3/envs/whatdo-llms-want/bin/python scripts/real_laptops_check_runs.py
Reads data/real_laptops/ and its design files. Writes nothing.

A. structure   27 runs per model, one per contract of contracts.csv; rows = n_rows; the ordered pairs are
               exactly the design's (singles: all 3,540; compounds: the 500 pairs, both orders); the item
               features equal items.csv; no missing score.
B. scores      the two answers hold all the probability; size of the margin; ties; slot-1 share.
C. order       no contract: per pair, D = (m[x first] - m[y first]) / 2 is the choice with the order
               averaged out, P = (m[x first] + m[y first]) / 2 is the push to the first slot.
               "order decides" = |P| > |D| (the same slot wins both orders).
D. contract    pairs where exactly ONE laptop meets the contract: share where it wins (D), next to the
               same pairs in the no-contract run.
"""
import json, glob, os, itertools
import numpy as np, pandas as pd
pd.set_option("display.width", 200)
JOBS = {"1498397", "1498398", "1498399", "1498400", "1498401", "1498402"}
items = pd.read_csv("data/real_laptops/design/items.csv")
contracts = pd.read_csv("data/real_laptops/design/contracts.csv")
cp = pd.read_csv("data/real_laptops/design/compound_pairs.csv")
ids = set(items.id)
feat = items.set_index("id")[["brand", "screen", "ram"]]
all_ordered = set(itertools.permutations(sorted(ids), 2))
comp_ordered = set(zip(cp.id_x, cp.id_y)) | set(zip(cp.id_y, cp.id_x))
want_cids = dict(zip(contracts.constraints_id, contracts.set))

runs, problems = {}, []
for cfg in sorted(glob.glob("data/real_laptops/*/config.json")):
    d = os.path.dirname(cfg); job = os.path.basename(d).split("_")[0]
    if job not in JOBS: continue
    c = json.load(open(cfg)); sc = os.path.join(d, "scores.csv")
    key = (f"{c['model_family']}-{c['model_size']}", c["frame"]["name"], c["constraints_id"])
    if not os.path.exists(sc): problems.append(f"{d}: no scores.csv"); continue
    assert key not in runs, f"duplicate key {key}"
    df = pd.read_csv(sc)
    runs[key] = (c, df, d)
    name = os.path.basename(d)
    def bad(msg): problems.append(f"{name}: {msg}")
    # --- config
    if c["constraints_id"] not in want_cids: bad("constraints_id not in contracts.csv")
    if "+".join(f"{k}={v}" for k, v in sorted(c["constraints"].items())) != c["constraints_id"] and c["constraints"]: bad("constraints dict != id")
    if (c["constraints"] == {}) != (c["constraints_id"] == "none"): bad("none mismatch")
    if list(c["templates"]) != ["0"]: bad("templates != [0]")
    if c["alternatives"] != "real_laptops" or c["frame"]["name"] != "shopping": bad("alias/frame")
    is_comp = want_cids.get(c["constraints_id"]) == "compound"
    if (c["pairs"] != "all") != is_comp: bad(f"pairs field {c['pairs']}")
    # --- scores
    if len(df) != c["n_rows"]: bad(f"rows {len(df)} != n_rows {c['n_rows']}")
    if list(df.columns) != "template_idx,score_a,score_b,a_id,a_brand,a_screen,a_ram,b_id,b_brand,b_screen,b_ram".split(","): bad("columns")
    if not (df.template_idx == 0).all(): bad("template_idx")
    if df[["score_a", "score_b"]].isna().any().any() or not np.isfinite(df[["score_a", "score_b"]].values).all(): bad("NaN/inf score")
    got = list(zip(df.a_id, df.b_id))
    if len(set(got)) != len(got): bad("duplicate ordered pairs")
    if set(got) != (comp_ordered if is_comp else all_ordered): bad("pair set differs from the design")
    for s in "ab":
        if not (df[[f"{s}_brand", f"{s}_screen", f"{s}_ram"]].values == feat.loc[df[f"{s}_id"]].values).all(): bad(f"{s}_ features differ from items.csv")
    for f, lvl in c["constraints"].items():
        if lvl not in set(items[f]): bad(f"level {lvl} not in items")

models = sorted({k[0] for k in runs})
print("=== A. structure")
print("runs found:", len(runs), "| per model:", {m: sum(k[0] == m for k in runs) for m in models})
for m in models:
    have = {k[2] for k in runs if k[0] == m}
    print(f"  {m}: missing {sorted(set(want_cids) - have) or 'nothing'} | extra {sorted(have - set(want_cids)) or 'nothing'}"
          f" | rows {sum(len(v[1]) for k, v in runs.items() if k[0] == m):,}"
          f" | commits {sorted({v[0]['git_commit'] for k, v in runs.items() if k[0] == m})}")
print("problems:", problems or "none")

print("\n=== B. scores (log-probabilities of 'Option 1' / 'Option 2')")
rows = []
for m in models:
    df = pd.concat([v[1] for k, v in runs.items() if k[0] == m])
    mass = np.exp(df.score_a.astype(float)) + np.exp(df.score_b.astype(float))
    mg = df.score_a - df.score_b
    rows.append(dict(model=m, rows=len(df), score_max=max(df.score_a.max(), df.score_b.max()), score_min=min(df.score_a.min(), df.score_b.min()),
                     mass_on_2_labels_mean=mass.mean(), mass_min=mass.min(), share_mass_below_0_9=(mass < 0.9).mean(),
                     margin_abs_mean=mg.abs().mean(), exact_ties=(mg == 0).sum(), option1_wins=(mg > 0).mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))

def DP(df):                      # per unordered pair x<y: D = position-corrected margin for x, P = push to the first slot
    m = (df.score_a - df.score_b).astype(float)
    t = pd.DataFrame({"a": df.a_id, "b": df.b_id, "m": m.values})
    f = t[t.a < t.b].set_index(["a", "b"]).m                      # x shown first
    s = t[t.a > t.b].rename(columns={"a": "b", "b": "a"}).set_index(["a", "b"]).m   # x shown second: m is for y
    s = s.reindex(f.index)
    return pd.DataFrame({"D": (f - s) / 2, "P": (f + s) / 2})

print("\n=== C. order effect, no contract (1,770 pairs)")
rows = []
for m in models:
    x = DP(runs[(m, "shopping", "none")][1])
    rows.append(dict(model=m, pairs=len(x), order_decides=int((x.P.abs() > x.D.abs()).sum()), share=(x.P.abs() > x.D.abs()).mean(),
                     first_slot_push_mean=x.P.mean(), abs_D_median=x.D.abs().median(), abs_P_median=x.P.abs().median()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("agreement between models on D (no contract):")
Ds = pd.DataFrame({m: DP(runs[(m, "shopping", "none")][1]).D for m in models})
print(Ds.corr().round(3).to_string())
print("same winner:", {f"{a} vs {b}": f"{((Ds[a] > 0) == (Ds[b] > 0)).mean():.3f}" for a, b in itertools.combinations(models, 2)})

print("\n=== D. does the contract move the choice?  pairs where exactly ONE laptop satisfies the contract")
print("    'follows' = share of those pairs where the satisfying laptop wins (position-corrected); 'base' = same pairs, no contract")
out = []
for (m, fr, cid), (c, df, d) in sorted(runs.items()):
    if cid == "none": continue
    x = DP(df)
    sat = feat.apply(lambda r: all(r[f] == l for f, l in c["constraints"].items()), axis=1)
    sx, sy = sat.loc[x.index.get_level_values(0)].values, sat.loc[x.index.get_level_values(1)].values
    one = sx != sy
    sign = np.where(sx, 1, -1)                       # +1 if x is the satisfying one
    base = DP(runs[(m, "shopping", "none")][1]).D.reindex(x.index)
    out.append(dict(model=m, cid=cid, depth=len(c["constraints"]), n=int(one.sum()),
                    follows=((x.D * sign)[one] > 0).mean(), base=((base * sign)[one] > 0).mean(),
                    order_decides=(x.P.abs() > x.D.abs()).mean()))
o = pd.DataFrame(out)
piv = o.pivot(index=["depth", "cid"], columns="model", values=["follows", "base"]).round(2)
piv.insert(0, "n", o.drop_duplicates("cid").set_index(["depth", "cid"]).n)
print(piv.to_string())
print("\nmean by depth:")
print(o.groupby(["model", "depth"])[["follows", "base", "order_decides"]].mean().round(3).unstack(0).to_string())
