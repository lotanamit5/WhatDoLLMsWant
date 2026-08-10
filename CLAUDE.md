# What Do LLMs Want?

MSc thesis project (Computer Science, Technion). Author: Lotan Amit. Supervisor: Dr. Nir Rosenfeld.
Full proposal: `obsidian_symlink/Thesis Proposal.pdf` (February 2026), plus the Overleaf/LaTeX source in the vault.

## How to talk to me

- **Keep answers short and simple.** English is not the user's first language. Use plain words, not academic or technical jargon. If a term is needed, explain it in one line.
- Give one recommendation, not a list of options.
- When you change analysis code, show the new numbers, not just the code.

## Keep the progress log updated

`docs/progress.md` is the running log of the project. **Update it as part of the work, without being asked**, whenever:

- a finding comes out of an analysis (what we learned, with the numbers),
- a method or decision changes (and why),
- something is left open or unfinished.

Newest entry on top, with a date. Keep entries short. This is what stops us re-arguing settled points in a later session.

Do **not** put method descriptions in this file. Methods are still changing — they belong in `docs/progress.md`.

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

## Current status (August 2026)

Phase I. The active question is **consistency and robustness**: does the model have a stable preference, or does it change with prompt wording, option order, and constraints?

The working dataset is the synthetic laptops set (`laptops_robustness` in [src/alternatives.py](src/alternatives.py)): 5 brands x 3 screen sizes x 3 ram sizes = 45 items, full factorial.
Models: Qwen-2.5 (0.5B / 7B / 32B / 72B) and Gemma-3 (1B / 4B / 12B / 27B).

**For the live status, the current method, and everything still open, read [docs/progress.md](docs/progress.md) first.** The method is under review — do not treat it as settled.
Older Hebrew notes are in `obsidian_symlink/Progress Notes/` (you can read them).

## Repo layout

| Path | What it is |
|---|---|
| `src/` | Shared code: agents, prompts, alternatives, BT models, plots. **Read-only, see rules below.** |
| `scripts/` | Data collection + SLURM launchers. |
| `data/` | **Results. Never modify or delete.** One folder per experiment set. |
| `experiments/` | Older results (Feb 2026, colors/foods/cars/stocks). Read-only, and gitignored. |
| `Notebooks/` | Analysis notebooks. Messy on purpose. |
| `docs/` | `progress.md` — the running project log. |
| `obsidian_symlink/` | Thesis notes, papers, meeting notes (Obsidian vault, mostly Hebrew). Not tracked by git. |

**One notebook per analysis.** When we start a new analysis, make a new notebook in `Notebooks/` — do not grow an old one.

Notebooks run from the repo root, not from `Notebooks/`. Every notebook starts with a cell that walks up to the root, so `data/...` paths and `import src...` both work. Copy that cell into any new notebook.

Active notebook: [Notebooks/num_vs_txt.ipynb](Notebooks/num_vs_txt.ipynb) (laptops robustness).
The other notebooks are **not in use**. They belong to the earlier colors experiments and use the older `src/experiment.py` path. Do not assume they still run, and do not update them.

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

`config.json` follows the schema in [docs/config_schema_proposal.md](docs/config_schema_proposal.md) (applied 2026-08-10). The part that matters for analysis:

- **`constraints` is a dict `{feature: level}`**, with level strings copied exactly from `src/alternatives.py` — the same strings that appear in `scores.csv`. So "does this item satisfy the contract?" is `all(row[f"a_{f}"] == lvl for f, lvl in constraints.items())`, with no hard-coded mapping.
- `{}` means no constraint. Never `null`, never `"None"`.
- `constraints_id` is the derived slug used for filtering (`"none"`, `"screen=14-inch"`, `"ram=8GB+screen=14-inch"`).
- `frame` is the scene-setting sentence, kept separate from the constraint. Everything up to 2026-08-10 used `shopping` ("I am looking to buy a laptop.").
- `prompt_prefix` records verbatim what was prepended to the template.

### Where the analysis code lives

| Module | What goes in it |
|---|---|
| [src/pref_models.py](src/pref_models.py) | Bradley-Terry fitting |
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
