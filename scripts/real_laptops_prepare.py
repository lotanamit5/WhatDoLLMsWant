"""Real laptops, stage 1: build the item list, the contracts and the pair samples.

Standalone, no GPU. Run from the repo root:
    /home/user1/.venvs/whatdo/bin/python scripts/real_laptops_prepare.py
Reads  data/real_laptops/design/laptop_price.csv   (Kaggle "Laptop Price", latin-1)
Writes data/real_laptops/design/{items.csv, contracts.csv, compound_pairs.csv, sanity_pairs.csv}
Everything is drawn with one fixed seed, so re-running gives the same files.
"""
import itertools

import numpy as np
import pandas as pd

DESIGN = "data/real_laptops/design"
SEED = 20261112
N_ITEMS, PER_CELL = 60, 3
CELLS = list(itertools.product(["Dell", "HP"], [14.0, 15.6], ["4GB", "8GB"]))
LEVELS = {"brand": ["Dell", "HP"], "screen": ["14-inch", "15.6-inch"], "ram": ["4GB", "8GB"]}
MIN_ITEMS = {1: 12, 2: 6, 3: 3}
N_COMPOUND_PAIRS, N_SANITY_PAIRS = 500, 200


def item_text(r):
    """One fixed line: name + specs + price."""
    name = f"{r.Company} {r.Product.split(' (')[0]}"
    line = (f"{name}: {r.TypeName}, {r.Inches:g}-inch {r.ScreenResolution}, {r.Cpu}, {r.Ram} RAM, "
            f"{r.Memory}, {r.Gpu}, {r.OpSys}, {r.Weight}, €{round(r.Price_euros):d}")
    line = " ".join(line.split())               # the table has double and trailing spaces ("SSD +  1TB")
    return line.replace(" ,", ",")


raw = pd.read_csv(f"{DESIGN}/laptop_price.csv", encoding="latin-1")
assert len(raw) == 1303, len(raw)
tab = raw.drop_duplicates(subset=[c for c in raw.columns if c != "laptop_ID"]).reset_index(drop=True)
assert len(tab) == 1275, f"expected 1275 rows after dropping duplicates, got {len(tab)}"

rng = np.random.default_rng(SEED)
core = []
for company, inches, ram in CELLS:
    cell = tab[(tab.Company == company) & (tab.Inches == inches) & (tab.Ram == ram)]
    core += list(rng.choice(cell.index.values, PER_CELL, replace=False))
rest = list(rng.choice(tab.index.difference(core).values, N_ITEMS - len(core), replace=False))

items = tab.loc[core + rest].copy()
items.insert(0, "part", ["core"] * len(core) + ["rest"] * len(rest))
items["brand"] = items.Company
items["screen"] = [f"{x:g}-inch" for x in items.Inches]
items["ram"] = items.Ram
items["text"] = [item_text(r) for r in items.itertuples()]
items = items.rename(columns={"laptop_ID": "id"}).reset_index(drop=True)
assert items.id.is_unique and items.text.is_unique, "two items share an id or a display line"
items.to_csv(f"{DESIGN}/items.csv", index=False)

# 27 contracts: each feature is not mentioned, or set to one of its two levels
contracts = []
for combo in itertools.product(*[[None] + LEVELS[f] for f in LEVELS]):
    con = {f: v for f, v in zip(LEVELS, combo) if v is not None}
    n_sat = int(np.all([items[f] == v for f, v in con.items()], axis=0).sum()) if con else len(items)
    contracts.append(dict(spec=",".join(f"{f}={v}" for f, v in con.items()),
                          constraints_id="+".join(f"{f}={v}" for f, v in sorted(con.items())) or "none",
                          depth=len(con), set="train" if len(con) <= 1 else "compound",
                          n_items_satisfying=n_sat, min_required=MIN_ITEMS.get(len(con), 0)))
contracts = pd.DataFrame(contracts).sort_values(["depth", "constraints_id"]).reset_index(drop=True)
assert (contracts.n_items_satisfying >= contracts.min_required).all(), "a contract has too few items"
contracts.to_csv(f"{DESIGN}/contracts.csv", index=False)

# pair samples (unordered, by item id; the collector asks both orders)
pairs = np.array(list(itertools.combinations(items.id.values, 2)))
assert len(pairs) == 1770
pd.DataFrame(pairs[np.sort(rng.choice(len(pairs), N_COMPOUND_PAIRS, replace=False))],
             columns=["id_x", "id_y"]).to_csv(f"{DESIGN}/compound_pairs.csv", index=False)
pd.DataFrame(pairs[np.sort(rng.choice(len(pairs), N_SANITY_PAIRS, replace=False))],
             columns=["id_x", "id_y"]).to_csv(f"{DESIGN}/sanity_pairs.csv", index=False)

n_train, n_comp = (contracts.set == "train").sum(), (contracts.set == "compound").sum()
total = n_train * len(pairs) * 2 + n_comp * N_COMPOUND_PAIRS * 2
pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_colwidth", 200)
print(f"rows: {len(raw)} -> {len(tab)} after dropping duplicates")
print(f"items: {len(items)} ({len(core)} core + {len(rest)} rest), seed {SEED}")
print(f"contracts: {len(contracts)} ({n_train} train, {n_comp} compound); prompts per model: {total}")
print(contracts[["constraints_id", "depth", "set", "n_items_satisfying", "min_required"]].to_string(index=False))
print()
for r in items.itertuples():
    print(f"{r.Index:2d} [{r.part}] id={r.id}: {r.text}")
