# Proposal: how `config.json` should record the prompt

Status: **proposal, not applied.** Written 2026-08-10 after a mislabelled run
(`laptops_robustness_50prompts/68337011`) cost an hour to identify.

## What is wrong today

```json
{
  "alternatives": "laptops_robustness",
  "constraints": "14_8",
  "constraints_text": "I am looking to buy a laptop. I prefer a 14-inch screen and 8 GB ram.",
  "model_family": "qwen", "model_size": "32", "slurm_job_id": "68374115",
  "templates": { "0": "...", "...": "..." }, "timestamp": "20260804_142927"
}
```

1. **`constraints` is an opaque slug.** `"14_8"` needs a lookup table to mean
   *screen = 14-inch, ram = 8GB*. Every analysis re-writes that table by hand.
2. **The slug is about to become ambiguous.** The next runs we want are a 16-inch screen and
   16GB ram. Both would be called `"16"`. This blocks the experiment.
3. **The frame is glued to the constraint.** *"I am looking to buy a laptop."* is itself a
   frame, and it lives inside `constraints_text`. We cannot vary it, and we cannot tell a
   frame-only run from a no-frame run.
4. **"No constraint" has three encodings in the data**: key missing, `null`, and the string
   `"None"`. Loading code has to handle all three.
5. **The prompt actually sent is not recorded.** This is what caused the 68337011 mess — the
   run had to be identified by matching 9900 rows of scores against another run.

## Proposed schema

```json
{
  "alternatives": "laptops_robustness",

  "frame": { "name": "shopping", "text": "I am looking to buy a laptop." },
  "constraints": { "screen": "14-inch", "ram": "8GB" },
  "constraints_text": "I prefer a 14-inch screen and 8 GB ram.",

  "prompt_prefix": "I am looking to buy a laptop. I prefer a 14-inch screen and 8 GB ram.",
  "constraints_id": "screen=14-inch+ram=8GB",

  "model_family": "qwen", "model_size": "32", "slurm_job_id": "68374115",
  "git_commit": "8801df4", "collection_script": "scripts/data_collection_robustness.py",
  "template_set": "options",
  "templates": { "0": "..." }, "timestamp": "20260804_142927"
}
```

### The one rule that matters

**`constraints` is a dict `feature -> level`, and the level strings are copied *exactly* from
`src/alternatives.py`.** Not `"14"`, not `"14in"` — `"14-inch"`, the same string that appears
in `scores.csv`.

That makes the central analysis question a one-liner, generic over any feature and any level,
with no hard-coded mapping anywhere:

```python
def satisfies(row, constraints, side="a"):
    return all(row[f"{side}_{feat}"] == level for feat, level in constraints.items())
```

This is exactly the satisfies/violates split behind the 2026-08-10 finding. Today it is
written by hand in the notebook and has to be rewritten for every new constraint.

### The other fields

| field | why |
|---|---|
| `constraints: {}` | the **only** encoding of "no constraint". Never `null`, never `"None"`. |
| `frame` | separated from the constraint, so the frame ladder (bare / shopping / self / third-person) is expressible. `{"name": "bare", "text": ""}` is the no-frame condition. |
| `prompt_prefix` | the exact string prepended, verbatim. Settles any future provenance doubt on its own. |
| `constraints_id` | derived slug for folder names and filtering. **Built from the dict, never typed by hand**, so it cannot drift from the truth or collide (`screen=16-inch` vs `ram=16GB`). |
| `git_commit`, `collection_script` | one `git rev-parse HEAD`; would have answered the 68337011 question instantly. |
| `template_set` | **added 2026-08-16.** Which set in `src/prompts.py` the templates came from: `"options"` (asks a question, ends `"Answer: "` — instruct models) or `"pretrained"` (ends mid-sentence, `"...I prefer Option "` — base models). The strings themselves are already in `templates`, so this is only so a run can be *filtered* by prompt style. Runs written before this date have no such key: treat a missing `template_set` as `"options"`, which is what they all used. |

### Building the text from the dict

Generate `constraints_text`, do not type it twice. Note the level string and the prose differ
(`"8GB"` -> `"8 GB ram"`), so each feature needs its own formatter:

```python
PHRASE = {
    "screen": lambda v: f"a {v} screen",     # "14-inch"  -> "a 14-inch screen"
    "ram":    lambda v: f"{v[:-2]} GB ram",  # "8GB"      -> "8 GB ram"
    "brand":  lambda v: f"a {v} laptop",
}

def constraints_text(constraints):
    if not constraints:
        return ""
    return "I prefer " + " and ".join(PHRASE[f](v) for f, v in constraints.items()) + "."

def constraints_id(constraints):
    return "+".join(f"{f}={v}" for f, v in sorted(constraints.items())) or "none"
```

Check before merging that this reproduces the four historical strings **character for
character**. Changing the wording changes the experiment.

## Migration of the 33 existing runs

A script `scripts/migrate_configs.py`, dry-run by default, `--apply` to write.

- Touches **`config.json` only**. Never `scores.csv`.
- Old slug -> new dict, using the historical `CONSTRAINT_ALIASES`:

  | old | `frame.name` | `constraints` |
  |---|---|---|
  | missing / `null` / `"None"` | `shopping` | `{}` |
  | `"14"` | `shopping` | `{"screen": "14-inch"}` |
  | `"8"` | `shopping` | `{"ram": "8GB"}` |
  | `"14_8"` | `shopping` | `{"screen": "14-inch", "ram": "8GB"}` |

- `constraints_text` and `prompt_prefix` are backfilled with the **actual historical string**,
  not a regenerated one.
- **Special case, `laptops_robustness_50prompts/68337011`:** currently has no `constraints`
  key, so it reads as unconstrained. It is not — it is the `14` condition with 43 wordings.
  Verified by matching all 9900 rows of its templates 0-4 against run `68337596` (agree to
  0.25 log-odds; against the true unconstrained run they differ by up to 25.8). Write:

  ```json
  "constraints": { "screen": "14-inch" },
  "note": "constraints recovered 2026-08-10 by row-matching against run 68337596; the run predates the constraints field. 43 templates, not 5."
  ```

- The runs in `data/qwen_pt`, `data/pmi_qwen`, `data/eos_fix_qwen`, `data/num_vs_txt` predate
  all of this and used no prefix at all: `frame: {"name": "bare", "text": ""}`,
  `constraints: {}`.

## Follow-up in `src/`

`load_scores_by_run` in [src/auxiliary.py](../src/auxiliary.py) filters on the old
`constraints` string. It needs a small change to filter on the dict. Ask before touching it —
`src/` is read-only by default.
