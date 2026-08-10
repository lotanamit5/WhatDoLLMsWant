# Progress Notes

Running log of what we did, what we found, and what is still open.
Newest entry on top. Written in English; the Hebrew thesis notes stay in the Obsidian vault
(`obsidian_symlink/Progress Notes/`).

Each entry: what changed, what we learned, what is still open.

---

## 2026-08-10 — Config schema changed, collection scripts merged

### Config: constraints are now a dict, not a slug

Schema written up in [docs/config_schema_proposal.md](config_schema_proposal.md) and
**applied to all 108 existing runs** with `scripts/migrate_configs.py` (config.json only;
`scores.csv` untouched and verified untouched; everything is in git if it needs undoing).

```json
"frame":         { "name": "shopping", "text": "I am looking to buy a laptop." },
"constraints":   { "screen": "14-inch", "ram": "8GB" },
"constraints_id": "ram=8GB+screen=14-inch",
"prompt_prefix": "I am looking to buy a laptop. I prefer a 14-inch screen and 8 GB ram."
```

Why it matters:

- **`constraints` levels are the exact strings from `alternatives.py`**, so the
  satisfies/violates test is one generic line and no longer hard-coded per constraint:
  `all(row[f"a_{f}"] == lvl for f, lvl in constraints.items())`.
  Re-running `apple_brand_decay.ipynb` off the dict reproduces the 2026-08-10 numbers
  exactly (violate −0.77 / satisfy +2.06, p = 1.3e-6) — the migration is faithful.
- **The old slugs could not express the next experiment.** A 16-inch screen and 16GB ram
  would both have been `"16"`. Now `screen=16-inch` and `ram=16GB`.
- `{}` is the only "no constraint" encoding (there were three: missing, `null`, `"None"`).
- `frame` is separated, so the frame ladder (`bare` / `shopping` / `self` / `third_person`)
  is expressible. Every run to date is `shopping`; **`bare` has never been run.**
- `prompt_prefix` records verbatim what was sent — the thing whose absence cost an hour.
- A typo like `ram=16 GB` now raises instead of silently producing a contract no item meets.

**`laptops_robustness_50prompts/68337011` is fixed**: `{"screen": "14-inch"}`, with a `note`
field recording that it was recovered by row-matching. It is a 43-wording version of the
`14` condition, not a baseline.

### Scripts merged

`data_collection_robustness.py` and `run_data_collection_robust.sh` were near-duplicates that
had drifted (one had constraints, the other had all templates and all item sets). Merged into
**`scripts/data_collection.py`** and **`scripts/run_data_collection.sh`**; the `*_robust*`
copies are deleted and `create_slurms.py` points at the survivor. New flags: `--constraints`
(`"screen=14-inch,ram=8GB"`), `--frame`, `--n_templates`.

### Readers updated

- `src/auxiliary.py::load_scores_by_run` filtered on the old string and after the migration
  silently returned only unconstrained runs (printing a misleading "job may still be
  running"). One line: it now reads `constraints_id`. The 9 `constraint_order` lists in
  `num_vs_txt.ipynb` were updated to match; all 4 qwen-32B runs load again.
- `brand_consistency.ipynb` keeps its old short slugs via a 2-line lookup; all 32 runs load.

### Open

- [ ] Still to run, in this order: **`screen=16-inch` and `ram=16GB` as single constraints**
      (16 runs) — the top level of each feature, completing the 2x2 against `14`/`8`. Only
      then `screen=16-inch,ram=16GB`. `ram=16GB` is the sharpest single test: it is where
      Apple is *strongest* unconstrained (+7.5), so if Apple still collapses there, C6 is
      purely about compliance and not about premium/budget framing.
- [ ] Then the frame ladder: `--frame bare` and `--frame self` (16 runs), plus a `qwen-pt`
      run on `laptops_robustness` — the base model has no chat template and no system
      prompt, which is the only true no-frame condition available.
- [ ] Explain the saturation caveat properly and agree what to do about it (deferred).

---

## 2026-08-10 — Re-read the 4 past presentations: what we promised vs. what we have

Converted `obsidian_symlink/Presentations/*.pptx` to markdown and read them, to avoid
re-arguing points already settled in meetings 1-4.

```
pip install 'markitdown[pptx]'
markitdown "What do LLMs Want 4.pptx" -o deck4.md
```

markitdown only extracts text. Most content in decks 3 and 4 is **images of text**, so the
markdown alone is misleading — the images have to be pulled out of the pptx (it is a zip:
`ppt/media/`, mapped to slides via `ppt/slides/_rels/slideN.xml.rels`) and read separately.

### Definitions from deck 3 that our current analysis never uses

These were agreed in meeting 3 and are missing from this log:

- **Delegation task.** User has utility `u`, communicates partial preferences `u' ⊂ u`,
  agent completes them with its own as `v|u'` and solves `x* = argmax_{x∈C} v_p(x)`.
- **Welfare gap** `ΔW(x) = u(x^opt) − u(x)`. This is the headline metric of Aim #2.
  Under the conjecture, `ΔW → 0` as the number of constraints `n → ∞`.
- **Robustness** = `{i : x*_i = y_i}`, which constraints the model actually honors.
- **Domain requirement #3:** for n ≈ 4-5, constraints must *gradually shrink* the welfare
  gap but **not close it**. This is a design target for the laptops set that we have never
  checked.

**Gap:** everything we have measured since (brand correlation, brand spread, contract
adherence) is about the agent's weights `v`, not about `ΔW`. We have never defined a user
utility `u` for the laptops set, so we cannot compute the welfare gap at all. Deck 3 also
notes the shortcut: *work backwards — choose `x*` first, then build `C` around it.*

### Deck 4's four next steps: 0 of 4 done

1. Token bias isolation (position vs. the tokens `1`/`2`) — still open, same as the PMI /
   label-bias item at the bottom of this file.
2. Pretrained vs. finetuned (does alignment inflate structural priors?) — not started for
   laptops. Deck 2's colors result is the only data point, and it is a near-total flip.
3. Weight-scale investigation (are BT magnitudes real, or an optimization artifact?) —
   partly answered on 2026-08-10 by the saturation finding: **they are not calibrated**.
4. Presentation context ("more context → stronger preference") — not started.

### Prior results worth not re-deriving

- Deck 2, colors: pretrained ranks blue≈green≈red > purple > yellow; instructed ranks
  **purple > yellow > green > blue > red**. Near-reversal between base and instruct.
- Deck 2: BT weights converge by ~30-40 templates. Our 5-template runs are below that;
  the 43-wording run is above it.
- Deck 3/4: a GRUM persona experiment already exists on **16 laptops** with personas
  {Student, Gamer, Business, Editor}. Personas shift utilities but keep the baseline shape.
  Different item set from the current 45-laptop factorial — not directly comparable.

### Open

- [ ] Define a user utility `u` for `laptops_robustness` so `ΔW` can be computed. Without it
      Aim #2 has no metric, and every number we report is about `v` alone.
- [ ] Check domain requirement #3 on the laptops set: does `ΔW` shrink gradually over
      `none → 14 → 8 → 14_8`, or does one constraint close it?

---

## 2026-08-10 — Why Apple falls: it is one brand, and the premium is conditional on compliance

New notebook: [Notebooks/apple_brand_decay.ipynb](../Notebooks/apple_brand_decay.ipynb).
Six conjectures for yesterday's "Apple collapses across phases" result, tested one by one.
Five rejected, one survives.

### The fact holds, and it needs no model

Apple win rate against a **same-spec** rival (same screen, same ram — brand is the only
difference). No Bradley-Terry involved:

| model | none | 14in | 8GB | 14in+8GB |
|---|---|---|---|---|
| qwen-32B | 78.6% | 70.3% | 56.9% | 48.6% |
| qwen-72B | 72.5% | 58.9% | 43.6% | 43.3% |
| gemma-27B | 86.9% | 64.4% | 47.8% | 34.7% |

### Verdicts

| conjecture | verdict | deciding number |
|---|---|---|
| C1 scale compression | **rejected** | Apple's scale-free z goes +1.5 → −1.8, rank 1 → 5. A rescale cannot change a rank. |
| C2 regression to the mean | **rejected** | see below |
| C3 position / wording / rerun noise | **rejected** | gamma cancels exactly; last in 43/43 wordings; rerun noise 0.003 |
| C4 additivity misspecification | premise true, **not the mechanism** | collapse survives inside a fixed spec cell |
| C5 any strong constraint | **rejected** | equal instruction strength, opposite brand effect (p = 0.016) |
| **C6 brand credit is conditional on compliance** | **supported** | satisfy +2.06 vs violate −0.77, p = 1.3e-6 |

### Finding: "brand preference is not robust" was too broad — it is one brand

Correlation of brand weights with the same model's unconstrained run, **all 5 brands** vs
**the 4 that are not Apple** (re-centered among themselves):

- all 5: mean −0.11, **67% negative**
- without Apple: mean **+0.92**, median +0.95, **0% negative** (21/21 runs)
- paired t-test p = 2.6e-9

ASUS, Dell, HP and Lenovo hold their order under every constraint. The whole "flip" reported
on 2026-08-09 is Apple moving. This **corrects that entry's headline.**

Regression-to-the-mean is not enough either: the pooled slope of (w_con − w_none) on w_none
is −1.07, but that slope is produced by Apple. Apple is a significant outlier from the line
itself (residual −0.70, p < 0.001); Dell sits on it (−0.07, p = 0.48).

### Finding: the mechanism — the brand premium survives only where the contract is met

Fit brand weights **inside a single (screen, ram) cell** — 5 laptops differing only in
brand, so nothing about constraint-satisfaction can differentiate them. Mean over 7 models,
189 cells:

| | Apple's weight | Apple's z | brand spread |
|---|---|---|---|
| no constraint | +5.64 | +1.49 | 10.5 |
| constrained, cell **satisfies** the contract | +2.06 | +0.52 | 7.2 |
| constrained, cell **violates** it | **−0.77** | −0.49 | 4.9 |

Welch p = 1.3e-6. Share of cells where Apple is still top brand: 89% → 49% (14in) → 27%
(8GB) → 17% (14in+8GB). The spread only halves, so the block is **not flattened** — Apple is
actively pushed down. Sharpest case: under `14_8` the only cell where Apple stays positive is
14-inch/8GB (+0.26), the exact contract match.

**Reading:** once the user states a spec, the model stops treating "Apple" as a reason to
prefer a laptop *unless that laptop meets the spec*. A non-compliant Apple is punished harder
than a non-compliant Dell. Directly relevant to the contract-theory frame: the agent's own
preference is not deleted by the contract, it is made **conditional on compliance**.

### Finding: the model already knows Apple's product line (before any constraint)

Within-cell Apple advantage in the **unconstrained** runs:
ram **+3.2 (4GB) → +6.3 (8GB) → +7.5 (16GB)**; screen **+6.6 (13in) → +4.9 (16in)**.
A 4GB or big-screen Apple is less Apple-like. So a single `brand_Apple` weight is an average
over cells the model treats very differently — worth remembering whenever we read one.

Related: the ram constraint disrupts brand more than the screen constraint at equal
instruction strength (brand corr +0.24 vs −0.23, paired p = 0.016; the constrained feature
moves 8.4 vs 8.2 log-odds in the two cases).

### Finding: C6 survives with no model and no magnitudes

Because of the saturation caveat below, C6 was re-tested using only **who won** — Apple's raw
win rate inside a fixed (screen, ram) cell, no Bradley-Terry, no magnitudes:

| | Apple's win rate |
|---|---|
| no constraint | 68.5% |
| constrained, cell **satisfies** the contract | 62.8% |
| constrained, cell **violates** it | **48.1%** |

Welch p = 6.8e-6, and the split holds inside each constraint separately (`8`: 61.8 vs 43.6).
A logistic BT on the win indicator agrees on the sign of Apple's weight in 27/28 runs.
So the conclusion does not depend on the log-prob scale.

### Finding: the measurement is saturated in 88-99% of comparisons

`score = logprob(label) − logprob(label | "Answer: ")`, so it is capped at −logprob(base).
In **88-99%** of all rows one of the two labels sits at that cap — the model is effectively
certain, and the margin we regress on is carried by the *loser's* tail log-probability.
Equally true with and without a constraint, so it does not produce the Apple effect, but our
"log-odds" are not calibrated log-odds. qwen-0.5B is the only model under 6% saturation —
and the only one with no measurable preference.

### Data hygiene: one run is mislabelled

`data/laptops_robustness_50prompts/68337011` has **no `constraints` field** in its
`config.json` (it predates the key), so `load_scores_by_run` reads it as unconstrained.
It is not. Its templates 0-4 reproduce run `68337596` (qwen-7B, `14`) to within 0.25 log-odds
per row, so it carries the "I prefer a 14-inch screen" preamble. It is a **43-wording version
of the `14` condition** — valuable as a wording-robustness check, wrong as a baseline.
Fix the config, or skip that folder when filtering on `constraints`.

Useful by-product: those two runs are the same 9900 prompts executed twice on the cluster.
Per-row bfloat16 jitter reaches 0.25 log-odds, but the fitted brand weights agree to
**0.003** — averaging kills it. Rerun noise is not a concern at the effect sizes we study.

### Open

- [ ] **Run the `16_16` constraint.** It is already in `CONSTRAINT_ALIASES`
      ([scripts/data_collection_robustness.py:35](../scripts/data_collection_robustness.py#L35))
      and has never been collected. 8 models x 1 constraint. It is the one experiment that
      splits the two live readings of C6: does *any* stated spec strip Apple's premium from
      non-compliant items, or only a **modest** spec? Every constraint we have ever run asks
      for a mid/low level (14-inch of 13/14/16; 8GB of 4/8/16), so we cannot tell them apart
      today.
- [ ] **Run a genuinely empty prompt.** Every run we have — including the ones we call
      "unconstrained" — carries the line *"I am looking to buy a laptop."* We have never
      measured brand preference without it, so we cannot say whether the Apple premium is the
      model's own or is created by the shopping frame.
- [ ] A constraint on a dimension **not in the item set** (e.g. "I prefer a lightweight
      laptop") would separate "any extra sentence" from "a spec the items can meet or miss".
- [ ] Does C6 generalise beyond Apple? Test it on the *second*-ranked brand per model, and on
      the earlier colors/foods/cars sets.

---

## 2026-08-09 — Brand preference is not robust (new notebook)

> **Superseded in part by the 2026-08-10 entry.** The headline below ("brand preference is
> not robust") is too broad: four of the five brands are stable (r = +0.92). The effect is
> Apple alone. Read the 08-10 entry with this one.

New notebook: [Notebooks/brand_consistency.ipynb](../Notebooks/brand_consistency.ipynb).
Covers the full grid — `laptops_robustness` (Qwen) + `laptops_robustness_gemma` (Gemma),
8 models x 4 constraints = **32 runs, no gaps**. Every run fitted with both BT models.
All weights mean-centered inside each feature block (sum-to-zero), so they are comparable
across runs.

### Finding: the brand preference flips under constraints that never mention brand

Correlation of the brand weights against the same model's unconstrained run:

| model | 14in | 8GB | 14in+8GB |
|---|---|---|---|
| qwen-0.5B | +0.91 | +0.92 | +0.94 |
| qwen-7B | -0.39 | -0.09 | -0.41 |
| qwen-32B | +0.75 | -0.09 | -0.36 |
| qwen-72B | -0.30 | -0.81 | -0.84 |
| gemma-1B | +0.94 | +0.45 | +0.68 |
| gemma-4B | +0.29 | -0.26 | -0.29 |
| gemma-12B | +0.31 | -0.47 | -0.52 |
| gemma-27B | +0.08 | -0.31 | -0.59 |

Mostly **negative** for the capable models. Not "weakly stable" — actively inverted.

**Apple is the clean case.** Top brand in every unconstrained run of both families
(+1.3 to +2.7), and negative in almost every constrained run. Consistent across Qwen and
Gemma, so it is not a single model's quirk.

qwen-0.5B is the exception at +0.9, but that is noise matched against noise: its brand
spread is ~0.1 vs 2-5 for the others, and its unconstrained R² is 0.085. No signal at all.

### Finding: constraints flatten the brand signal

Brand spread (best - worst brand weight):

| model | none | 14in | 8GB | 14in+8GB |
|---|---|---|---|---|
| qwen-32B | 3.82 | 1.47 | 1.67 | 2.16 |
| qwen-72B | 2.84 | 0.95 | 2.61 | 2.54 |
| gemma-27B | 4.88 | 1.35 | 2.30 | 2.90 |

The named feature soaks up the utility and brand shrinks.

### Finding: the additive feature model is a good description — except under two constraints

Full parameterization (45 free weights, one per laptop) vs feature parameterization
(9 weights, summed): Pearson 0.96-1.00, Spearman 0.94-1.00 on item utilities.
R² given up by the feature model is ~0.001-0.013 in single-constraint and unconstrained
runs, but jumps under the **double** constraint:

- qwen-7B `14_8`: 0.064   ·   gemma-12B `14_8`: 0.065   ·   qwen-32B `14_8`: 0.026

So "satisfies both constraints" is worth more than "satisfies screen" + "satisfies ram".
A real interaction, only when two constraints are active.

### Finding: contract adherence scales hard with model size

Weight on `14-inch` when the prompt asks for a 14-inch screen:

- qwen-0.5B: +0.02 → +0.30 (nothing)
- qwen-32B: +0.32 → **+13.78**
- gemma-27B: -0.37 → **+13.18**

Also: two constraints dilute each other. qwen-32B gives 14-inch +13.78 alone but +11.88
under `14_8`; 8GB +10.75 alone but +6.74 under `14_8`.

### Finding: positional bias is bigger than most of the preferences we measure

gamma runs about -5 to -11 for every model except qwen-0.5B (~-0.2), and does not shrink
with model size. Brand weights are 1-3. So the option-order effect is several times larger
than the thing we are trying to measure. This raises the priority of the PMI / label-bias
work already listed below.

### Finding: the reported p-values are too optimistic (~2.6-3x)

Added section 4 to the notebook. The fit treats all 9900 rows as independent, but each
unordered pair `{A,B}` appears as `(A,B)` and `(B,A)`, each repeated over 5 templates —
so **10 rows share every item pair**. There are 990 independent pairs, not 9900 rows.

Refitted with cluster-robust SEs (clustered by unordered pair):

- **18% of the residual variance is between-pair** — the rows inside a pair are clearly
  not independent, which is exactly what OLS assumes away.
- Standard errors inflate **x2.6 to x3.1** for every capable model.
- **Big effects survive**: ram, screen and gamma stay at p ~ 0 either way. Not artifacts.
- **Borderline brand terms do not**: qwen-32B unconstrained, `brand_Lenovo` goes from
  p = 4e-7 to **p = 0.061**. Some brand weights we would have called real are not.
- qwen-0.5B has inflation *below* 1 — but its R² is 0.085 and its effects are ~0.1
  log-odds. With n = 9900 even noise is "significant"; judge it by effect size.

**Working rule from now on:** don't read the printed p-values as-is. Either use clustered
SEs, or judge an effect against the positional bias (5-11 log-odds), which is bigger than
every brand effect we have measured.

### Finding: the positional bias decides 15-30% of all comparisons

Fig 1.1 now has a second row per family: for each feature, the **largest effect it can have
on one prediction** — which is exactly that feature's weight range, since a prediction is
`gamma + sum_f (w_f(A) - w_f(B))`. So feature ranges and `|gamma|` are directly comparable.
Result: **brand is always the smallest bar**, and the bias is 3-10x bigger than it.

Fig 1.2 is the blunt version — how often does the bias alone change who wins?

| model | none | 14in | 8GB | 14in+8GB |
|---|---|---|---|---|
| qwen-7B | 16.9% | 22.0% | 27.3% | 25.0% |
| qwen-32B | 15.5% | 16.2% | 27.3% | 19.2% |
| qwen-72B | 19.4% | 26.0% | 15.9% | 22.0% |
| gemma-12B | 20.8% | 26.2% | 27.3% | 29.6% |
| gemma-27B | 15.7% | 14.5% | 15.9% | 16.8% |

(qwen-0.5B hits 46% unconstrained — it has no preference, so position decides nearly a coin
flip's worth of comparisons.)

`|gamma|` is also ~0.5-1.2x the **median utility gap between two random laptops**. So the
bias is the same order of magnitude as the whole preference signal, not a small nuisance.

Also added gamma to Fig 1.4 (red ticks against brand spread) and to the stacked composition
bars. Same log-odds units everywhere, one axis, no second scale.

### Open

- [ ] qwen-0.5B has no measurable preference on anything. Decide whether to keep it in the
      analysis at all, or report it only as a floor/noise reference.
- [ ] Should the clustered SEs become the default in
      `src/pref_models.py::fit_feature_based_bradley_terry`? Right now that function prints
      the naive p-values, which we now know overstate confidence ~3x.
- [x] The brand flip needs a mechanism. Is it a real re-ranking, or an artifact of the
      constrained feature dominating the logit scale and compressing everything else?
      Worth checking whether the flip survives a per-run rescale.
      **Answered 2026-08-10:** a real re-ranking (rank 1 → 5, z +1.5 → −1.8; compression
      rejected), and the mechanism is that the brand premium is conditional on contract
      compliance. See the 08-10 entry.
- [ ] `fit_full_item_bradley_terry` currently lives in the notebook. Move to
      `src/pref_models.py` if we keep using it.

---

## 2026-08-09 — Repo cleanup + Bradley-Terry identifiability

### Done

- Added `CLAUDE.md` (project purpose, workflow, rules).
- Moved all notebooks into `Notebooks/`. Each notebook now starts with a cell that
  moves the working directory to the repo root, so `data/...` paths keep working.
- Started this progress log.
- **Moved 13 analysis functions out of `num_vs_txt.ipynb` into `src/`** (verbatim, no logic
  changed). The notebook now imports them and keeps only `run_laptop_ranking_experiment`,
  the glue that changes most often.

  | Where they went | Functions |
  |---|---|
  | `src/pref_models.py` | `fit_feature_based_bradley_terry`, `fit_item_bradley_terry` |
  | `src/plots.py` | `plot_preference_shift`, `plot_laptop_ranking_shift`, `plot_feature_weight_trajectories`, `plot_shift_attribution`, `plot_laptop_feature_weights` (+ helper `_diverging_attribution_barh`) |
  | `src/auxiliary.py` | `load_scores_by_run`, `find_any_scores_csv`, `get_item_feature_map` |
  | `src/metrics.py` | `compute_adherence_rank_correlations`, `decompose_item_shift` |

  Also fixed one pre-existing bug found on the way: `src/plots.py` had
  `from plot_utils import better_color_map`, which fails when imported as `src.plots` from
  the repo root. Changed to `from src.plot_utils import ...`. The module simply could not be
  imported before this.

  Verified end to end against `data/laptops_robustness` (`model_size=32`, all 3 constraints):
  loading, both BT fits, all plots, the correlations and the shift decomposition all run.

### Finding: why feature-based BT must pin one level per feature

Checked on `data/laptops_robustness/1271581` (Qwen-72B, no constraint).

In `fit_feature_based_bradley_terry` the design matrix is `dummies_a - dummies_b`.
Every laptop has exactly one brand, one screen, one ram — so inside each feature block the
columns sum to **exactly zero** on every row:

```
brand   levels=5  max|sum over levels| = 0.0
screen  levels=3  max|sum over levels| = 0.0
ram     levels=3  max|sum over levels| = 0.0
```

So the matrix is rank-deficient by exactly 3 = the number of features:

```
full-dummy X:  shape (9900, 11)  rank 8
dropped  X:    shape (9900,  8)  rank 8
```

Meaning: adding a constant to *all* levels of one feature changes nothing we can observe.
Bradley-Terry only identifies **differences** of utilities, never absolute levels.
One free constant per feature → one level per feature must be pinned.

Notes:
- This is **not** the usual dummy-variable trap. Here the block sums to *zero*, not to *ones*,
  so the problem exists even with no intercept, and there is one per feature (not one overall).
- The positional-bias intercept `gamma` is safe: `rank([1, X_dropped]) = 9` with 9 columns.
- `drop_first=True` and the `pd.concat([df_a, df_b])` before `get_dummies` are both correct
  and both load-bearing (the concat guarantees A and B get the same reference level).

### Finding: not dropping does not break the fit, but hides the choice

statsmodels does **not** error on the rank-deficient matrix. It falls back to `pinv` and returns
the minimum-norm solution — which turns out to be exactly the **sum-to-zero** gauge
(each feature block's weights sum to 0, verified to ~1e-15). Point estimates, standard errors,
and `df_resid` all match an explicit effects-coded fit to machine precision:

```
max |w_pinv - w_sumzero|   = 1.2e-14
max |se_pinv - se_sumzero| = 8.3e-17
```

So the inference is valid — but the gauge is invisible. Someone reading the coefficients as
"contrasts vs a reference level" would misread them by a constant.

### Finding: softmax is a valid gauge fix but a bad one here

Softmax is shift-invariant, so it *does* give a unique answer regardless of which solution you
picked. But `log softmax(w)_l = w_l - logsumexp(w)`, so it is just mean-centering with a
different constant — no new information. And it saturates on our scale (ram spans ~25 nats):

```
ram_16GB   w=  12.29   softmax=0.999990
ram_4GB    w= -13.10   softmax=0.000000
ram_8GB    w=   0.82   softmax=0.000010
```

All resolution is gone. Also our scores are PMI-normalized logprob differences, not calibrated
log-odds, so `exp()` of them has no probability meaning.

### Open / to do

- [ ] Switch `fit_feature_based_bradley_terry` to **explicit sum-to-zero coding**, so every
      level gets a coefficient and a standard error (including the currently-invisible
      reference level), and the gauge is intentional instead of implicit.
- [ ] `plot_feature_weight_trajectories` silently drops one level per feature
      (`brand_ASUS`, `screen_13-inch`, `ram_16GB`). The dashed zero line *is* their trajectory,
      but it is unlabeled. Add them back explicitly.
- [ ] References are chosen by **string sort**, so ram's reference is `16GB`, not `4GB`
      (`'16GB' < '4GB' < '8GB'`), and brand's is `ASUS` only because uppercase sorts first.
      Fragile: adding a `32GB` level would silently flip the whole ram panel.
      Pin references explicitly with `pd.Categorical`.
- [ ] Phrase cross-run claims in gauge-invariant terms. "Did brand bias flatten?" should be
      answered with the **spread** of the block (SD or max-min, including the reference at 0),
      not by looking at single lines — a single line is always a contrast against the
      reference, which may itself be moving.
- [ ] Same caveat for `decompose_item_shift`: the per-item total delta is safe, but the split
      into `brand + screen + ram` is gauge-dependent in level.

Verification scripts (throwaway): `scratchpad/check_bt.py`, `scratchpad/check_gauge.py`.

---

## Current elicitation method (under review)

This is what the pipeline does **today**. It is not settled — reviewing and possibly changing
it is active work.

1. Show the model two options in a prompt (`src.prompts.options_comparisons`, first 5 templates).
2. Read the logprob of the answer token `"1"` vs `"2"` — not generated text.
3. Normalize by PMI: subtract the logprobs from a bare context ([src/agent.py:152](../src/agent.py#L152)).
4. Ask both orders of every pair (`itertools.permutations`) to measure positional bias.
5. Fit Bradley-Terry on `score_a - score_b` with OLS, plus an intercept for positional bias.
   Proof that the OLS trick works: `obsidian_symlink/Notes/הוכחה לטריק ה-OLS עבור BT עם BIAS.md`

Open questions on the method itself (from the Hebrew progress notes, 2026-05 to 2026-06):

- **Positional bias.** Is it a real preference, or just a token prior for `"1"` / `"2"`?
  Idea from the notes: the model may be using "always pick the first" as its way to *randomize*,
  because it has no real way to flip a coin. A kind of reward hacking.
- **Which null context for PMI?** Candidates: masked prompt template / last line only /
  empty string. Calibration test proposed in the notes: compare two *identical* options and see
  which null context gives a zero margin.
- **Label scheme.** Try `X/Y`, `A/B`, `1st/2nd`. Try adding a third "I don't care / equal" option.
- **Sanity check** that the option tokens are actually in the model's top-k completions.
- **PriDe-style prior correction** ([arxiv 2309.03882](https://arxiv.org/abs/2309.03882)):
  measure logits on meaningless options to get a prior bias vector, then correct the real results.
