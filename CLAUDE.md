# What Do LLMs Want?

MSc thesis project (Computer Science, Technion). Author: Lotan Amit. Supervisor: Dr. Nir Rosenfeld.
Full proposal: `obsidian_symlink/Thesis Proposal.pdf` (February 2026), plus the Overleaf/LaTeX source in the vault.

## How to talk to me

- **Keep answers short and simple.** English is not the user's first language. Use plain words, not academic or technical jargon. If a term is needed, explain it in one line.
- Give one recommendation, not a list of options.
- When you change analysis code, show the new numbers, not just the code.

## Read this first, every session

**This file holds only the rules that do not change.** Anything about the current state of the
project lives in `docs/` — do not add status, findings, or "what's next" here.

| File | What it is | How it is written |
|---|---|---|
| [docs/status.md](docs/status.md) | **Where we are, what we know, what is next.** Read it first. | Rewritten in place. Keep it current. |
| [docs/progress.md](docs/progress.md) | History of findings, newest on top. | Append-only. Never rewrite an old entry — correct it in a new one. |
| [docs/grum_formalization.md](docs/grum_formalization.md) | The method (GRUM, contracts as agents). | Rewritten as the method changes. |
| [docs/config_schema.md](docs/config_schema.md) | The `config.json` schema. | Rewritten when the schema changes. |

**Update these as part of the work, without being asked:**

- a finding comes out of an analysis → new dated entry in `progress.md`, **with the numbers**
- a decision or method changes → `progress.md` (why), and `grum_formalization.md` if it is the method
- anything finished, started, or newly open → move the line in `status.md`

Keep entries short. This is what stops us re-arguing settled points in a later session.

## Research purpose

The project studies LLM agents through the lens of **contract theory**. A user (principal) delegates a task to an LLM (agent) by giving instructions. These instructions are an **incomplete contract** — they never describe the user's full preferences.

**Main conjecture:** given a partial contract, the LLM will "complete" it with *its own* preferences.

Three phases:

1. **Passive elicitation** (current phase) — measure the LLM's own preferences.
2. **Active delegation** — move from *stated* preferences to *revealed* preferences (what the model actually picks when acting).
3. **Aggregate outcomes** — what happens in markets/games when many agents share the same hidden preferences.

Task complexity grows in this order:
1. **Simple** — items with no features (colors, foods, cars, stocks).
2. **Conditional** — preferences under a stated condition, put in the prompt (the `--constraints` flag of [scripts/data_collection.py](scripts/data_collection.py)).
3. **Conflicting** — items are feature vectors with trade-offs (the laptops sets: brand + screen + ram).

## Repo layout

| Path | What it is |
|---|---|
| `src/` | Shared code: agents, prompts, alternatives, BT models, plots. **Read-only, see rules below.** |
| `scripts/` | Data collection + SLURM launchers. |
| `data/` | **Results. Never modify or delete.** One folder per experiment set. |
| `experiments/` | Older results (Feb 2026, colors/foods/cars/stocks). Read-only, and gitignored. |
| `Notebooks/` | Analysis notebooks. Messy on purpose. |
| `docs/` | The four files above. AI-written; safe to rewrite. |
| `obsidian_symlink/` | Lotan's own notes, in Obsidian, mostly Hebrew. Not tracked by git. **Do not rewrite these** — they are his, and under-processed on purpose. Read them, and only edit when asked. |

Inside the vault: `Meeting Notes/` (with Nir — this is where tasks come from), `Notes/` (concepts), `Papers/`, `INBOX/` (unfiled), `Future Ideas/`, `Archive/`, `Presentations/`.
`Progress Notes/` is the old Hebrew log; it stopped 2026-06-04 and continues in `docs/progress.md`.

**One notebook per analysis.** When we start a new analysis, make a new notebook in `Notebooks/` — do not grow an old one.

Notebooks run from the repo root, not from `Notebooks/`. Every notebook starts with a cell that walks up to the root, so `data/...` paths and `import src...` both work. Copy that cell into any new notebook.

Which notebooks are live changes often — `docs/status.md` says. Older ones belong to the colors experiments and use the removed `src/experiment.py` path: do not assume they run, and do not update them.

## Workflow

Two machines. Do not mix them up.

**Cluster (SLURM)** — data collection, needs GPUs. Paths there are `/home/lotan.amit/...`.
Params are edited by hand in [scripts/create_slurms.py](scripts/create_slurms.py) (`exp_name`, model, size, alternatives, constraints), which writes `scripts/slurms.sh`. That calls [scripts/run_data_collection.sh](scripts/run_data_collection.sh) → [scripts/data_collection.py](scripts/data_collection.py) (one script for every run since the 2026-08-10 merge; the `*_robust*` duplicates are gone).

**The user launches the jobs from the server. Never try to run them from here** — no GPUs, no SLURM, wrong paths.

**Local (this WSL machine)** — analysis only, no GPU. Paths are `/home/lotanamit/...`.

Use the conda env python. Plain `python` is base conda and has **no statsmodels**:

```
~/miniconda3/envs/whatdo-llms-want/bin/python
```

### The data contract

Every run writes exactly:

```
data/<exp_name>/<slurm_job_id>/
    config.json    # model_family, model_size, alternatives, frame, constraints, templates
    scores.csv     # template_idx, score_a, score_b, a_<feature>..., b_<feature>...
```

Analysis code finds runs by **filtering `config.json`** (`load_scores_by_run` in [src/auxiliary.py](src/auxiliary.py)). Keep this contract and new runs appear in the notebooks automatically, with no code change.

- **A run directory with `config.json` but no `scores.csv` is an INCOMPLETE run — every loader must skip it.** `config.json` is written *before* the scoring loop, so a job that dies (disk full, OOM, timeout) leaves the folder behind. Those dirs are never deleted (`data/` is append-only), so after a rerun there are two folders for the same key: the dead one and the good one. A loader that only checks `config.json` will either crash on the missing `scores.csv` or trip the uniqueness assert. Check both files exist, and ideally that the row count is the expected one.

`config.json` follows the schema in [docs/config_schema.md](docs/config_schema.md) (applied 2026-08-10). The part that matters for analysis:

- **`constraints` is a dict `{feature: level}`**, with level strings copied exactly from `src/alternatives.py` — the same strings that appear in `scores.csv`. So "does this item satisfy the contract?" is `all(row[f"a_{f}"] == lvl for f, lvl in constraints.items())`, with no hard-coded mapping.
- `{}` means no constraint. Never `null`, never `"None"`.
- `constraints_id` is the derived slug used for filtering (`"none"`, `"screen=14-inch"`, `"ram=8GB+screen=14-inch"`).
- `frame` is the scene-setting sentence, kept separate from the constraint. Everything up to 2026-08-10 used `shopping` ("I am looking to buy a laptop.").
- **`frame` is part of a run's identity, not a detail** (2026-08-16). `constraints_id` alone is **not** a unique key: an unconstrained run is `"none"` whether its frame is `shopping` or `bare`. Every loader must key on **`(model_family, model_size, frame, constraints_id)`**, or filter `frame` explicitly and assert the rest is unique.
- **Hold the frame fixed inside any comparison.** All constrained runs use `shopping`; `bare` exists only unconstrained. So a contract comparison pins `frame="shopping"`, and a frame comparison pins `constraints={}`. Never vary both — the frame sentence alone moves brand weights by up to 4.6 log-odds.
- `prompt_prefix` records verbatim what was prepended to the template.

### Where the analysis code lives

| Module | What goes in it |
|---|---|
| [src/bt.py](src/bt.py) | **`FeatureBT`** — the feature Bradley-Terry fit used by the current laptop notebooks. One definition, imported everywhere; do not paste a copy into a notebook. |
| [src/pref_models.py](src/pref_models.py) | Older BT fitting functions (`fit_feature_based_bradley_terry`, `fit_item_bradley_terry`). Still used by the colour-era notebooks. |
| [src/plots.py](src/plots.py) | All plotting |
| [src/auxiliary.py](src/auxiliary.py) | Finding and loading runs from `data/` |
| [src/metrics.py](src/metrics.py) | Comparing runs (correlations, shift decomposition) |
| the notebook | Glue that changes per experiment |

## Rules for changing code

This is research code for a thesis. It will change many times before it is done. **Flexible beats correct-by-construction.** Production-quality code makes quick changes painful — that is the opposite of what is needed here.

- **`src/` is read-only.** Only edit it if explicitly told to. If a change there seems needed, ask first and make it as small as possible.
- **Never modify or delete `data/` or `experiments/`.** They cost GPU hours.
- **Do not refactor code you were not asked to touch.**
- No tests. No type hints everywhere. No new abstraction layers, config systems, or class hierarchies.
- No error handling for cases that cannot happen. Let it crash — a stack trace is more useful here.
- Hard-coded values in notebooks are fine.
- Prefer changing a notebook cell over changing `src/`.
