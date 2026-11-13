"""Synthetic cars: are the runs in a folder complete and sane?

Run from the repo root:
    ~/miniconda3/envs/whatdo-llms-want/bin/python scripts/synthetic_cars_check_runs.py data/synthetic_cars_pipeline_test
Reads every run folder under the given folder (config.json + scores.csv) and the job logs. Writes nothing.

Per run:
    rows / n_rows      scores.csv rows against config.json's n_rows (a COMPLETE run has them equal)
    bad                rows with a missing or non-finite score
    both orders        every pair was scored in both orders
    saturated %        rows where the winning answer has log-prob > -0.01
    slot 1 %           rows where the first slot wins
    complies %         pairs where exactly one car meets the contract: share where it wins (D > 0),
                       D = (Y[a first] - Y[b first]) / 2;  "before" = the same pairs in the no-contract run
                       of the same job
Per job (from output.log): server, GPUs, prompts per second, exit status.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

BASE = sys.argv[1].rstrip("/")
FEATS = ["company", "color", "fuel", "transmission", "ownership", "horsepower"]
pd.set_option("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 500)


def pair_D(d):
    """One row per unordered pair: D for the lower id, and whether both orders exist."""
    lo, hi = np.minimum(d.a_id, d.b_id).values, np.maximum(d.a_id, d.b_id).values
    y = (d.score_a - d.score_b).values * np.where(d.a_id.values < d.b_id.values, 1.0, -1.0)
    g = pd.DataFrame(dict(lo=lo, hi=hi, y=y, t=d.template_idx.values)).groupby(["lo", "hi"])
    return pd.DataFrame(dict(D=g.y.mean(), n=g.y.size(), nt=g.t.nunique()))


runs, incomplete = [], []
for r in sorted(os.listdir(BASE)):
    cf, sc = f"{BASE}/{r}/config.json", f"{BASE}/{r}/scores.csv"
    if not os.path.isfile(cf):
        continue
    cfg = json.load(open(cf))
    if not os.path.isfile(sc):
        incomplete.append((r, "config.json but no scores.csv"))
        continue
    runs.append(dict(folder=r, job=str(cfg["slurm_job_id"]), cfg=cfg, d=pd.read_csv(sc)))

items = pd.read_csv("data/synthetic_cars/design/items.csv").set_index("id")
rows = []
for R in runs:
    cfg, d = R["cfg"], R["d"]
    con = cfg["constraints"]
    P = pair_D(d)
    row = dict(job=R["job"], model=f"{cfg['model_family']}-{cfg['model_size']}", contract=cfg["constraints_id"],
               rows=len(d), n_rows=cfg["n_rows"], complete=len(d) == cfg["n_rows"],
               bad=int((~np.isfinite(d[["score_a", "score_b"]])).any(axis=1).sum()),
               both_orders=bool((P.n == 2 * P.nt).all()), pairs=len(P),
               saturated_pct=100 * (d[["score_a", "score_b"]].max(axis=1) > -0.01).mean(),
               slot1_pct=100 * (d.score_a > d.score_b).mean())
    if con:
        ok = pd.Series(np.all([items[f] == v for f, v in con.items()], axis=0), index=items.index)
        lo_ok, hi_ok = ok.loc[P.index.get_level_values("lo")].values, ok.loc[P.index.get_level_values("hi")].values
        mixed = lo_ok != hi_ok
        win = np.where(lo_ok, P.D.values > 0, P.D.values < 0)       # the car that meets the contract wins
        row["mixed_pairs"] = int(mixed.sum())
        row["complies_pct"] = 100 * win[mixed].mean() if mixed.any() else np.nan
        none = [Q for Q in runs if Q["job"] == R["job"] and not Q["cfg"]["constraints"]]
        if none:
            P0 = pair_D(none[0]["d"]).reindex(P.index)
            win0 = np.where(lo_ok, P0.D.values > 0, P0.D.values < 0)
            keep = mixed & P0.D.notna().values
            row["complies_before_pct"] = 100 * win0[keep].mean() if keep.any() else np.nan
    rows.append(row)
T = pd.DataFrame(rows)
print(f"{BASE}: {len(runs)} complete-looking runs, {len(incomplete)} without scores.csv")
for r, why in incomplete:
    print(f"   INCOMPLETE {r}: {why}")
print(T.to_string(index=False, float_format="%.1f"))
assert T.complete.all() and (T.bad == 0).all() and T.both_orders.all(), "a run failed a check - see the table"

# per job, from the logs
jobs = []
for j in sorted({R["job"] for R in runs} | {r for r in os.listdir(BASE) if r.isdigit()}):
    log = f"{BASE}/{j}/output.log"
    if not os.path.isfile(log):
        continue
    txt = open(log).read()
    prog = re.findall(r"\[(\d\d):(\d\d):(\d\d)\] (\d+)/(\d+) rows", txt)
    rate = np.nan
    if len(prog) >= 2:
        sec = [int(h) * 3600 + int(m) * 60 + int(s) for h, m, s, *_ in prog]
        done = [int(p[3]) for p in prog]
        rate = (done[-1] - done[0]) / max((sec[-1] - sec[0]) % 86400, 1)
    jobs.append(dict(job=j, server=(re.search(r"started: .* on (\S+),", txt) or [None, "?"])[1],
                     gpus=(re.search(r"Found (\d+) GPUs", txt) or [None, "?"])[1],
                     runs=sum(R["job"] == j for R in runs), prompts=int(prog[-1][4]) if prog else 0,
                     prompts_per_s=rate, model_load_min=(re.findall(r"loading the model", txt).__len__()),
                     status=(re.search(r"Finished with status (\d+) after (\d+) min", txt) or [None, "?", "?"])[1],
                     total_min=(re.search(r"Finished with status (\d+) after (\d+) min", txt) or [None, "?", "?"])[2]))
print("\njobs:")
print(pd.DataFrame(jobs).to_string(index=False, float_format="%.1f"))
