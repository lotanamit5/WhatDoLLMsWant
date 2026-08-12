# Progress Notes

Running log of what we did, what we found, and what is still open.
Newest entry on top. Written in English; the Hebrew thesis notes stay in the Obsidian vault
(`obsidian_symlink/Progress Notes/`).

Each entry: what changed, what we learned, what is still open.

---

## 2026-08-12 — Drop PMI. It corrupts γ and it moved the adherence numbers by up to 33 points.

Follow-up to the γ bug. **PMI is on for every qwen and gemma run** — `load_qwen2_5_agent` and
`load_gemma3_agent` build `InstructedHFAgent(model_id)` with the default `normalize_pmi=True`.
Only `qwen-pt` escapes it (its PMI block is commented out).

### What PMI does: subtracts one constant, nothing more

The base context is `prompt.split('\n')[-1]`, which is always the literal `"Answer: "`. So:

```
margin_PMI = raw_logit − C,     C = logP("1"|base) − logP("2"|base)
```

`C` is large and positive — the model strongly expects `"1"` after a bare `"Answer: "`:
gemma-1B +10.25, gemma-4B +10.50, gemma-12B +9.75, qwen-7B +6.00, qwen-72B +5.49,
gemma-27B +4.99, qwen-32B +2.25.

| what | affected? | why |
|---|---|---|
| BT feature weights | **no** | a constant lands entirely in the intercept |
| γ | **yes** | γ *is* the intercept |
| anything using `sign(margin)` | **yes** | the decision threshold becomes `C`, not 0 |
| position-corrected `D = (m(x,y) − m(y,x))/2` | **no** | `C` cancels exactly |

`sign(margin)` flips for 0.7%–16% of rows (gemma-1B worst).

### This already bit us: the adherence numbers are wrong

`wins()` in `lexicographic.ipynb` — behind every adherence number in the 08-10 entry — uses
`m > 0`. Recomputing the `ram=8GB` ceteris-paribus table position-corrected (75 pairs each):

| model | conflict, reported 08-10 | **position-corrected** |
|---|---|---|
| qwen-0.5 | 93.3 | **100.0** |
| qwen-7 | 65.3 | **98.7** |
| qwen-32 | 50.0 | **17.3** |
| qwen-72 | 86.7 | **100.0** |
| gemma-1 | 10.0 | **0.0** |
| gemma-4 | 2.0 | **0.0** |
| gemma-12 | 50.0 | **22.7** |
| gemma-27 | 84.7 | **100.0** |

**The "did it hear / will it pay" split survives and gets sharper.** The agree direction is now
exactly **100.0 for all eight models** (was 96–100). But:

- The conflict behaviour is **near-binary**: a model either fully pays (98.7–100) or fully
  refuses (0–22.7). The intermediate 50s and 65s were position bias and the PMI offset leaking
  into `sign(margin)`.
- **The size story breaks for qwen.** Position-corrected, qwen-32B is at **17.3%**, between
  qwen-7B at 98.7% and qwen-72B at 100%. Not monotone. Gemma still is (0 → 0 → 22.7 → 100).
- qwen-0.5B is at 100% on both directions — but it has essentially no RAM preference to give
  up, so this is not obedience in the same sense.

**"Contract adherence scales hard with model size" (2026-08-09) does not survive for qwen and
must be re-derived** across all constraints before it is used again.

### Decisions

- **Drop PMI.** It only ever subtracts a constant, which γ already absorbs. It costs a second
  forward pass per prompt to recompute the *same* base string 9900 times, it corrupts γ, and it
  moves every `sign(margin)` threshold off zero. Set `normalize_pmi=False` in the loaders.
- **Use position-corrected `D` for every "who wins" question.** Immune to PMI, to γ, and to any
  other constant. Then the PMI choice stops mattering at all.
- **No re-collection needed.** `C` is recoverable from existing data for the seven saturating
  models. Not for qwen-0.5B, which never saturates — its `C` must be measured directly.

### Applied same day

- **`normalize_pmi=False`** in `load_qwen2_5_agent` and `load_gemma3_agent`
  ([src/agent.py](../src/agent.py)) — authorised by Lotan. Affects future runs only.
- **`figs4deck5.ipynb`: `winrate()` is now position-corrected.** One shared function feeds
  Figs 5.3, 5.4, all four 5.5 variants and 10b.3, so one change fixes every adherence figure.
  Post-processing only — **`scores.csv` files are untouched** (pending Lotan's confirmation).
  All 37 deck-5 figures regenerated.

Verified six ways before changing anything: (1) the base context is provably the single string
`"Answer: "`; (2) the old `wins()` reproduces all 8 published numbers exactly; (3) the two "50.0"
models pick option 1 in **0.0%** of rows — they always answered "option 2" whatever the content,
which is what a 50.0 was made of; (4) the two-order merge is exact, 75 pairs, no rows lost;
(5) adding an arbitrary constant to `score_a` leaves the corrected number unchanged, so it does
not depend on recovering `C`; (6) an independent BT fit with a position intercept reproduces
`mean_D` exactly and agrees in sign for all 8 models.

Corrected override rate (Fig 5.3, conflict set):

| model | screen=14-inch | ram=8GB | both |
|---|---|---|---|
| qwen-7B | 7.0 | 78.1 | 80.3 |
| qwen-32B | 92.5 | 77.2 | 69.6 |
| qwen-72B | 89.5 | 100.0 | 84.3 |
| gemma-1B | 0.5 | 1.1 | 11.7 |
| gemma-4B | 0.0 | 3.8 | 65.3 |
| gemma-12B | 70.9 | 58.0 | 98.7 |
| gemma-27B | 92.1 | 99.4 | 87.7 |

The deck-5 headline changes from *"gemma goes 10% → 16% → 54% → 93%"* to
**"gemma goes 1% → 4% → 58% → 99%"** — a sharper story, and gemma's size trend survives.
qwen does not: 78% → 77% → 100%. Also note **qwen-7B obeys a screen contract only 7% of the
time** while obeying a ram contract 78% — the reverse of every other model.

Baseline sanity after the fix: the "contract did nothing" line is still 50 (measured 50.5 screen,
50.6 ram, vs 51.2 / 50.3 before), and the silent-contract control (4GB vs 16GB) is **100.0 for
all 7 models** — the RAM veto is untouched where the contract says nothing.

### Open

- [ ] Same fix still needed in `lexicographic.ipynb` (its own `wins()`) and anywhere else
      `sign(margin)` is used as "the model chose this one".
- [ ] **Pending Lotan's confirmation:** rewrite the `scores.csv` files to remove the PMI offset,
      so the stored data needs no post-processing. `C` is recoverable for the 7 saturating
      models; **not** for qwen-0.5B, which never saturates — its `C` must be measured directly.
- [ ] Why is qwen-32B the outlier on the ceteris-paribus conflict (17.3%)? It also has the
      largest weight spread and the most saturation. Worth one look.
- [ ] qwen-7B's screen contract at 7.0% is strange enough to check separately.

---

## 2026-08-12 — The weight scale is a nuisance parameter. The question's premise was wrong.

Open since 2026-06-10 / 06-18: *is the trend in the scale of the weights an optimization
artifact or real signal?* Five conjectures tested in parallel.
Notebook: [Notebooks/weight_scale_vs_model_size.ipynb](../Notebooks/weight_scale_vs_model_size.ipynb).

**Answer: neither — there is no trend to explain, and the scale does not measure preference.**

### Finding: there is no monotonic trend

Spearman(size, total spread) never reaches significance: ρ=0.67 p=0.083 (all 8), ρ=0.50 p=0.267
(drop qwen-0.5B), ρ=0.20 p=0.714 (also drop gemma-1B). Among the top 6 models **6 of 15
size-ordered pairs go the wrong way**. The linear-in-log slope is carried entirely by
qwen-0.5B: 13.81 (p=0.014) with it, 7.09 (p=0.073) without; dropping any *other* model moves
the slope by less than 4.

The number that settles it — within-model SD across the 4 constraint conditions vs
between-model SD:

| set | within | between | ratio |
|---|---|---|---|
| all 8 | 3.40 | 12.59 | 3.70 |
| drop qwen-0.5B | 3.63 | 5.97 | 1.64 |
| **drop qwen-0.5B + gemma-1B** | **3.90** | **4.00** | **1.03** |

**Among the top 6 models, "which model" is no bigger a lever on the weight scale than "which
prompt".** qwen-72B (32.5) below qwen-32B (43.7) is real: pair-clustered bootstrap CIs are 8
points apart and do not touch.

**Design flaw to remember:** with 4 sizes per family the smallest possible two-sided Spearman p
is **0.083**. A within-family scaling claim can never be significant with this design. Future
scaling work needs more sizes, not more rows.

### Finding: the log-odds does not depend on how big the difference is

The decisive result. Position-free median log-odds:

| swap | qwen-7 | qwen-32 | qwen-72 | gemma-1 | gemma-4 | gemma-12 | gemma-27 |
|---|---|---|---|---|---|---|---|
| ram 4 vs 8 (2×) | 22.53 | 29.13 | 22.77 | 13.52 | 20.05 | 20.30 | 24.27 |
| **ram 4 vs 16 (4×)** | **21.70** | **29.04** | **22.49** | **14.07** | **21.73** | **20.80** | **23.75** |

Additivity ratio `L(4→16) / [L(4→8)+L(8→16)]` = **0.49–0.54 for all 8 models**, where a cardinal
utility predicts 1.0. Reproduced independently in the notebook (0.50–0.54).

This is a **categorical response**: the model emits one fixed magnitude whenever "more RAM wins",
so direct = S, two-step = 2S, ratio 0.5. It holds for qwen-0.5B too, which has **zero
saturation** — so it is not squashing against a ceiling.

**Consequence:** the dummy fit reports `ram_4GB = −26.5, ram_8GB = −12.6, ram_16GB = 0`, making
4GB look twice as bad as 8GB. The data says 4-vs-8 and 4-vs-16 are equally decided. **The ladder
comes from the design — how often each level wins — not from preference strength.**

### Finding: bigger models are louder, not different

- SVD of the 7 weight vectors: **rank-1 explains 99.64%** of the variance.
- Weight vectors never turn more than **9.1°** (median 5.2°) while length changes up to **113%**.
- Spearman of the 45-laptop ranking across models: **0.971–0.995**. Where two models disagree on
  a winner, those pairs have mean |margin| only **18%** of average — noise on near-ties.
- Correlation of weight spread with held-out accuracy, excluding qwen-0.5B: **−0.04**. With
  template-agreement error: −0.22. Size does slightly better (−0.46, −0.58) but that is almost
  all gemma, which starts below the ceiling.
- **qwen-7B is the best of all 8 on held-out accuracy (0.990)**, while 32B and 72B are worse with
  equal or larger spreads.
- RAM veto rate is exactly **1.000** for six of seven working models — lexicographic structure is
  already complete at 4B and cannot increase.

### Rejected: saturation, and the circularity trap that nearly hid it

There is **no logprob floor**. The loser's logprob is smooth over 30 log units; only 0.06–0.18%
of rows sit within 0.5 of a run's minimum. What saturates is the **winner** token (exactly 0.0
for 82–98.5% of rows), which costs no information because p₁+p₂=1. **Tobit equals OLS to 2
decimals** for all 7 models; only 0.02–0.05% of rows are at a limit.

`frac(|margin|>10)` predicts total spread with r=0.96 — but both are computed from the *same*
margins. Measured on **disjoint rows** (saturation on the 900 brand-only rows, spread on the
other 9000) it drops to r=0.268 (p=0.17, n.s.) and does not beat size. Neither variable is
causal: there is one latent decisiveness scalar T per model, spread ≈ T × shape and
saturation ≈ P(T·shape > 10).

### Rejected: template pooling — but it overturns qwen-0.5B

The design is perfectly balanced (1980 rows per template), so template dummies are **exactly
orthogonal** to the differenced feature dummies. Template FE change every weight by **0.000**.
*This closes the open item "make template a random effect and see if the leakage slopes move" —
they do not move.*

What pooling does break is the residual SD, and only for the two smallest models. **qwen-0.5B's
R² goes 0.085 → 0.920** with template intercepts. Its per-template γ ranges over 1.0, which is
3× its entire feature spread of 0.33.

Also: **5 templates is enough.** Subsampling 5 of the 43 in the `screen=14-inch` run gives
SE ≈ 1.0 on total spread, unbiased, against a between-model range of 22.8.

### Two measurement bugs

1. **γ is reported wrong.** The base context is `prompt.split('\n')[-1]`, which for every
   template and every item pair is the literal string **`"Answer: "`** — so it is **one constant
   `C` for the whole run**. (Checked directly: all 5 templates end in `"Answer: "`, not in a
   newline. An earlier draft of this entry said the base context was the empty string; that was
   wrong, but it does not change the conclusion, because all that matters is that `C` is
   constant.)

   Since `margin = true_logit − C`, the OLS intercept estimates `γ_true − C`. Recover `C` from
   the saturated rows: where `p(1) → 1` we have `logP(1|prompt) → 0`, so
   `max(score_a) ≈ −logP(1|base)`. Hence `C = −max(score_a) + max(score_b)` and
   `γ_true = γ_reported + C`. Verified in the notebook:

   | model | γ reported | C | **γ corrected** |
   |---|---|---|---|
   | gemma-1 | −6.94 | +10.25 | **+3.31** ← sign flips |
   | gemma-4 | −8.51 | +10.50 | **+1.99** ← sign flips |
   | gemma-12 | −10.52 | +9.75 | −0.77 |
   | gemma-27 | −7.83 | +4.99 | −2.84 |
   | qwen-7 | −9.05 | +6.00 | −3.05 |
   | qwen-32 | −6.98 | +2.25 | −4.73 |
   | qwen-72 | −9.44 | +5.49 | −3.95 |

   **This is not a small correction: two of the eight change sign, and the rest shrink by
   2–3×.** So "positional bias decides 15–30% of comparisons" needs re-deriving.

   The recovery is only valid where the run saturates. Validity check `exp(la)+exp(lb) ≤ 1`
   (the two answer tokens hold all the mass): median **1.000** for all 7 signal models, but
   **1.369** for qwen-0.5B — impossible, because it never saturates. **γ cannot be corrected
   this way for qwen-0.5B.** For that model the base context must be measured directly.
2. **The pooled OLS shrinks small effects 3–5×.** Fitting on design cells where only one feature
   differs: qwen-7 brand 0.86 → 4.81 (5.6×), gemma-27 brand 4.88 → 20.11 (4.1×). Wherever RAM
   differs the answer is pinned regardless of brand, so brand gets no leverage and OLS averages
   it away. **The "RAM dominates" gap is partly manufactured by the pooled fit.** The
   lexicographic ordering survives; the size of the gap is overstated.

### Corrections to earlier entries

- **2026-08-09, "qwen-0.5B has no measurable preference on anything" — wrong.** Its weights sit
  33 SD above a permutation null and its template-FE R² is 0.920. It has a real, reproducible,
  *tiny* preference. What is true: its signal never beats its own noise (`spread/resid_SD` =
  0.80 vs 4.5–5.3) and it is genuinely inconsistent per-choice (order agreement 0.6%, 453
  3-cycles of 14190). Both hold; they are not in conflict. Treat it as a different scale regime,
  not a broken measurement.
- **2026-08-10, "the measurement is saturated in 88–99% of comparisons"** — true of the winner
  token, but it is **not** censoring the margin. The loser has full dynamic range. The open item
  "a tobit is the honest response model" is now closed: tobit changes nothing.
- **2026-08-10, "PMI is a no-op"** — correct for the weights, but it is *not* a no-op for γ.

### Decisions

- Stop reporting raw weight scale as a preference magnitude. Report the **shape** (cosine 0.99
  across all models) plus **one decisiveness number** per model — which is not monotone in size.
- Normalize by **`margin_SD`**, not `resid_SD`. `resid_SD` is worse than not normalizing at all
  (CV 0.19 vs 0.095 over 28 runs). The "spread/resid_SD ≈ 5" pattern was a coincidence of the 7
  unconstrained runs.
- `total_spread / margin_SD` is monotone in size in 6/8 family × constraint cells (raw: 0/8) —
  but **no individual block is monotone** (ram is the worst, mean ρ = −0.26), so treat it as a
  composite statistic, not evidence that one thing grows.
- `ram_spread / margin_SD` = **1.514, CV 0.053** across a 72× parameter range. On the dominant
  axis the scale is pure nuisance.

### Spin-off worth its own analysis

Under `screen=14-inch`, in margin-SD units, bigger models shift weight **off** their own RAM
preference **onto** the stated contract:

| | gemma-1 | gemma-4 | gemma-12 | gemma-27 | qwen-7 | qwen-32 | qwen-72 |
|---|---|---|---|---|---|---|---|
| screen (contract) | 0.286 | 0.372 | 1.135 | **1.324** | 0.552 | 1.199 | 1.152 |
| ram (own preference) | 1.366 | 1.472 | 0.963 | **0.713** | 1.403 | 0.870 | 0.984 |

This is the scale-free version of "contract adherence scales with model size" — and unlike the
weight scale, it is a genuine preference result.

Separately, C5 found that **template wording moves the contract-vs-preference balance**. Same
model, same items, same contract: template 35 *"Which aligns more with your preferences…"* gives
ram 12.8 / screen 15.0 (contract wins); template 20 *"Between the two, do you prefer…"* gives
ram 27.0 / screen 9.0 (own preference wins 3×). Early evidence for the instruction-strength
experiment. All 43 current templates are paraphrases of "which do you prefer" — that experiment
needs new templates.

### Open

- [ ] **Next experiment this points to: a wider feature ladder** (RAM 4/8/64/256GB, or a price
      ladder). Everything above says the response is categorical, but the tested range is narrow.
      If `L(4→256) ≈ 2 × L(4→16)`, the magnitude *does* carry cardinal information and the
      categorical reading fails. Cheap, and it settles the question.
- [ ] Fix γ in the reported outputs (subtract the PMI part).
- [ ] Decide whether to report brand/screen from design cells instead of the pooled fit.
- [ ] No within-model replicate at fixed size exists, so we cannot say how much of the length
      variation is run-to-run noise. Two qwen-32B runs with different seeds would settle it.
- [ ] Every scale-free quality measure ceilings above ~4B. A harder item set (no single feature
      decides) is needed to tell whether that ceiling is real.

---

## 2026-08-11 — Notes reorganized: `status.md` for state, this file for history

`progress.md` had become both the log and the to-do list — ~35 open checkboxes spread over 8
`### Open` sections, with two conflicting "next runs" lists. Decision:

- **`docs/status.md`** — where we are, what we know, what is next. Rewritten in place.
- **`docs/progress.md`** (this file) — append-only history. Never rewritten. `### Open`
  sections stay as the record of what was open *at the time*; the live list is `status.md`.

Processed all 22 meeting notes + the thesis proposal into `status.md`. What that turned up:

- **Nir's asks are not tracked anywhere.** Three from 2026-08-04 are untouched: full vs
  feature parametrization (likelihood), the "I prefer"/"I must" instruction-strength ladder,
  and a gibberish-feature control group. The control group was also asked for by Itay on
  05-20 — the same request twice, ten weeks apart.
- **Experiment 1 (num vs txt) is half-collected**: `laptops_num_vs_txt` has qwen-7B only,
  the plan called for 7B *and* 72B. No correlation number has ever been reported.
- **Nir's Experiment 4 was never done** — when a contract makes both laptops bad, is
  positional bias very strong? It is the cleanest test of the "bias = indifference" reading,
  and it needs no new data.
- **Four questions have gone unanswered across two or more meetings**: the weight-scale/model-size
  trend (06-10, 06-18), GPT commerce for domain feel (05-12, 07-22), choices vs induced order
  (07-22), and the domain-requirements checklist (02-26, two open checkboxes).
- **Drift from the proposal**, now recorded: we score the answer token, not perplexity; we use
  a fixed template set, not runtime LLM-generated prompts; we swapped Qwen-3 for Gemma-3.

Also: `Meeting Notes/26-07-22` had ~79 lines of an unrelated project pasted into it, burying
Nir's four numbered experiments. Removed (133 → 54 lines).

### Decisions taken

- **"soft-robustness" means graded compliance, not weak wording.** Under a `14-inch`
  contract, a 13-inch is a near miss and a 16-inch is a bigger one: score adherence by
  **distance from the requested level**, not pass/fail. Both screen and ram are ordered, so
  this needs no new data. Open question it answers: does the hard 2%–87% "will it pay?" gap
  soften into a gradient? The *"I prefer"* vs *"I must"* ladder is a separate experiment.
- **Two tracks in parallel.** Track 1: collect the missing runs on the cluster. Track 2:
  analyse existing data meanwhile. `scripts/create_slurms.py` now emits **40 jobs** — GRUM
  Phase A (`screen=13-inch`, `screen=16-inch`, `ram=4GB`, `ram=16GB` × 8 models) plus the
  never-run `--frame bare`. All into `data/laptops_robustness/`, gemma included: runs are
  found by filtering `config.json`, so one folder beats two.
- **Qwen-3 later, not now.** Two families (Qwen-2.5, Gemma-3) carry the size-scaling claim.
  The proposal promised Qwen-3; revisit after Phase A.
- **`CLAUDE.md` now holds only static rules.** Everything time-varying moved to `status.md` —
  it had been claiming `num_vs_txt.ipynb` was the active notebook long after work moved on.
- `config_schema_proposal.md` → `config_schema.md`. It has been the applied spec since 08-10,
  not a proposal.

`create_slurms.py` takes a **list** of parameter dicts now, not one dict, because model family
and size do not cross (qwen has no 1B, gemma has no 0.5B) and the bare-frame batch varies a
different flag. Later batches (brand contracts, `num_vs_txt` on 72B) are sitting commented out
in the same list.

---

## 2026-08-10 — Adherence splits cleanly into "did it hear?" and "will it pay?"

Comparing the two adherence figures in `figs4deck5.ipynb` (5.5 vs 10b.3) showed they measure
different pair sets, and the difference is the interesting part.

Every contract creates two kinds of comparison:

- **agree direction** — the contract and the model's own order want the same laptop
  (`ram=8GB`: 8GB vs 4GB). Costs nothing to obey.
- **conflict direction** — they want different laptops (`ram=8GB`: 8GB vs 16GB). Obeying
  means giving up something the model wants.

Ceteris paribus, under `ram=8GB`:

| model | agree (8 vs 4) | conflict (8 vs 16) | Fig 5.5 (the average) |
|---|---|---|---|
| qwen-0.5B | 96.0 | 93.3 | 94.7 |
| qwen-7B | **100.0** | 65.3 | 82.7 |
| qwen-32B | **100.0** | 50.0 | 75.0 |
| qwen-72B | **100.0** | 86.7 | 93.3 |
| gemma-1B | 99.3 | **10.0** | 54.7 |
| gemma-4B | **100.0** | **2.0** | 51.0 |
| gemma-12B | **100.0** | 50.0 | 75.0 |
| gemma-27B | **100.0** | 84.7 | 92.3 |

**Every model with real signal is at 100% on the agree direction.** They all hear the
instruction. What varies — from 2% to 87% — is whether they will *pay* for it. Fig 5.5's
single number is exactly the mean of the two columns (verified: 77.3 = 77.3), so it is
diluted by a direction on which everyone scores full marks.

This corrects the reading of gemma-1B/4B as "failing to follow the contract". gemma-4B obeys
**100%** when obeying is free and **2%** when it costs a RAM step. That is not a
comprehension failure — it is an unwillingness to trade.

Same split under `screen=14-inch`: agree 73-100, conflict 17-100. Smaller gap, because
screen was only a tendency to begin with.

**Consequence for the deck:** report the two directions separately, not their average.
"Does it hear the instruction?" and "will it give something up for it?" are different
questions and only the second one discriminates between models.

Secondary note: Fig 10b.3 lets brand and screen vary inside the conflict pairs (2250 rows vs
150 ceteris paribus). On average that barely matters (ram 58.7 vs 55.3; screen 82.0 vs 79.1)
but per-model it moves by up to 14-20 points (gemma-4B ram: 15.9 vs 2.0), so the
ceteris-paribus version is the one to quote.

---

## 2026-08-10 — Correction: a ram contract is NOT targeted, and "cancels" was too strong

Prompted by a fair objection to `s05_override`: it conditions only on RAM, so it cannot tell
"the contract promoted the feature it named" apart from "the contract changed everything".
New figure `s05_what_moves` measures all three features **ceteris paribus** — pairs that
differ in one feature and are tied on the other two (90 ordered pairs x 5 templates = 450
rows per cell).

### Finding: the screen contract is surgical, the ram contract is global

Mean change in "% won by the preferred level", over the 7 models:

| contract | ram | screen | brand |
|---|---|---|---|
| `14in` | **−2** | **−30** (named) | −4 |
| `8GB` | **−17** (named) | **−17** | **−17** |
| `14in+8GB` | −16 (named) | −30 (named) | −11 |

Naming the **screen** moves the screen by 30 points and leaves ram (−2) and brand (−4)
essentially alone. Naming the **ram** moves its own target *less* (−17) and drags screen and
brand down by exactly as much. So a ram contract is not a targeted instruction — it flattens
the whole preference structure.

### Correction: "the contract cancels the brand preference" overstated it

The GRUM leakage slope of −0.94 was read as cancellation. But **pure shrinkage toward zero
also gives a slope of −1**, so that number cannot separate the two. Checking directly on the
spec-tied stratum:

| | brand spread | correlation with the no-contract brand vector |
|---|---|---|
| no contract | 10.4 | — |
| `14in` | 5.2 | **+0.74** (1 of 7 negative) |
| `8GB` | 3.7 | **+0.19** (3 of 7 negative) |
| `14in+8GB` | 4.2 | **−0.00** (4 of 7 negative) |

So the dominant effect is **flattening** (spread falls 2-3x), with genuine re-ordering on top
only under the ram and double contracts. Under the screen contract it is almost pure
shrinkage. The honest phrasing is "the contract **flattens** the brand preference, and under
a ram contract also scrambles what is left" — not "cancels", and not a clean sign flip.

Supporting detail: median |margin| in the brand stratum falls 11.2 → 5.8 under `ram=8GB`
(the model becomes genuinely less decisive about brand), while Apple's ceteris-paribus win
rate crosses **below** 50% in 3 of 7 models — so it is compression *plus* inversion, not
either alone.

This does not disturb the priority-order result (that is measured on win rates), nor the
C5/C6 ram-vs-screen asymmetry. It sharpens what the asymmetry *is*: naming ram is a blunt
instrument, naming screen is a precise one.

---

## 2026-08-10 — Deck 5 figures updated and distilled

[Notebooks/figs4deck5.ipynb](../Notebooks/figs4deck5.ipynb) now carries a **running order**
at the top: 13 figures to present, the other 18 marked backup. Re-runs clean, 31 figures.

**Added** (all model-free or gauge-safe, per the magnitudes caveat):

- `s05_override` — adherence as the **override rate**, not as a log-odds weight. Replaces
  `s05_adherence_scale` as the headline for "does it listen".
- `s10b_priority` — the priority order measured: RAM veto / screen tendency / brand
  tie-breaker, with the veto / tendency / no-effect bands drawn.
- `s10b_additivity_broken` — the 8x brand-spread split. The one figure that justifies
  dropping the additive reading.
- `s10b_contract_power` — a contract beats a tendency completely, a veto only partly.
- `s11_leakage` — the GRUM cancellation slope (ram −0.94, screen −0.54) as a deck figure.
  Computed directly as `(w_contract − w_none)·w_none / (w_none·w_none)` on the spec-tied
  stratum: for a saturated `x` this **is** the GRUM `B` row, and it reproduces
  `grum.ipynb` to 2 dp. No need to port the pooled fit into the deck notebook.

**Corrected two stale figures:**

- `s12_additivity` was titled "additivity holds, except under two constraints". That claim
  is wrong-headed: the full-vs-feature R² gap is dominated by ram, so it cannot see the
  brand-level failure at all. Retitled and captioned as a diagnostic with a blind spot.
- `s14_next_runs` was built around "ram=16GB is the sharpest test". Rewritten around the
  GRUM identification argument — the 4 missing single contracts (`screen=13-inch`,
  `screen=16-inch`, `ram=4GB`, `ram=16GB`), which take the contract space from rank 3 to 7.
  The old ram=16GB argument is kept as a secondary reason on the same slide.

Two standing warnings recorded at the top of the notebook: never put a BT magnitude on a
slide as if it were a trade-off, and `s05_adherence_scale` / `s12_additivity` are superseded.

---

## 2026-08-10 — GRUM: contracts as "agents". The leakage block *is* the Apple collapse.

Moving from "one Bradley-Terry fit per phase" to **one model across phases**, so that the
interaction between contracts and features is a parameter instead of a comparison of tables.

> **Formalization:** [docs/grum_formalization.md](grum_formalization.md).
> **Numbers:** [Notebooks/grum.ipynb](../Notebooks/grum.ipynb).

The mapping: a **contract is a GRUM agent**. `U_ij = delta_j + x_i' B z_j + eps_ij`, with
`i` = experiment condition, `j` = laptop, `x` = the contract, `z` = item features. Because `x`
and `z` are over the *same* feature space, `B` splits into two halves that mean different
things — **diagonal blocks = compliance** (did it do what it was told), **off-diagonal blocks
= leakage** (did
it change a preference it was never asked about). Statement 2 of the thesis conjecture is then
one Wald test on a block.

### The MC-EM machinery is not needed — this is one OLS

With `delta_j = a' z_j`, the observation equation is `y = gamma_i + (a + B' x_i)' dz + noise`.
So **the per-condition BT weight vector is exactly `w_i = a + B' x_i`**, and the whole GRUM is
an OLS of the margin on `[condition dummies | dz | dz (x) x]`.

**Verified: the pooled fit reproduces the per-condition BT weights to ~1e-14** for all 8 models
once `x` is saturated. It is a reparameterization, not a new estimator — so nothing already
established is put at risk by adopting it.

The paper needs Gibbs/MC-EM because it sees only *rankings*, so utilities are latent. We see a
continuous cardinal margin per pair; that problem does not exist for us. **Most of `grum4llm/`
solves a problem we do not have** — only its experimental-design half is worth reusing.

### Finding: a ram contract cancels the brand preference; a screen contract cancels half

Slope of the leakage row on `delta` over the whole 5-brand block (both sum-to-zero, so the slope
is gauge-safe and scale-free). `-1.0` = the contract exactly cancels the brand preference:

| | mean over the 7 models with real signal |
|---|---|
| contract on **ram** | **-0.94** |
| contract on **screen** | **-0.54** |

Apple, on the spec-tied stratum: `delta` +7.31 and `B[ram=8GB]` **-6.07** for qwen-32B; +10.79
and **-11.25** for gemma-27B. The leakage is negative in **8/8** models for both contracts.

So C6 and the C5 ram/screen asymmetry are now **two coefficients** instead of a table of
correlations — and the reading sharpens: the contract does not *shrink* the brand premium, it
**cancels** it (`delta + B` sits near zero for every capable model). Consistent with the
lexicographic entry below: naming ram installs an absolute veto above brand, naming screen only
a partial one.

`delta` on the spec-tied stratum reproduces the "same ram + same screen" column of the entry
below exactly (7.31, 10.79). Same measurement, new parameterization.

### Decisions taken

- **`x` encodes "no constraint" as zero**, not as a reference level. Then `x = 0` gives
  `U = delta`, so **`delta` is literally the contract-free preference**. No constant column in
  `x` — it would be confounded with `delta`.
- **Compound contracts get one shared saturation scalar, not a free conjunction vector.**
  See the generalization section below — this corrects an earlier draft of this entry that
  treated the conjunction as a free parameter per double.
- **Fit each block in the stratum where it is operative.** Given the lexicographic finding
  below, the brand block is fitted on **spec-tied** pairs only (900 rows/condition). Under a
  priority rule a lower-priority utility is identified *only* inside ties on the higher-priority
  ones. Pooling all 9900 pairs averages a live regime with a dead one.
- **The GRUM does not carry the priority-order claim.** Priority order stays model-free (win
  rates, override rates). GRUM's job is the interaction structure, measured where the preference
  is real. Response stays the continuous margin, not hard wins — with hard wins the RAM veto
  perfectly separates and the MLE diverges (Ford's condition / Thm 2 in the GRUM paper).
- SEs clustered by unordered pair throughout. Note the p-values are useless here anyway: with
  ~990 clusters, qwen-0.5B's 0.04 coefficient still prints `<1e-16`. Judge by effect size
  relative to `delta`.

### Finding: the model GENERALIZES to a contract it never saw — direction yes, magnitude no

This is the part that makes it a model and not a description, and it is what a saturated `x`
cannot do. The only held-out test the data supports: fit on `{none, screen=14, ram=8}`, predict
`screen=14+8`. Additivity forces `w_hat(14+8) = w(14) + w(8) - w(none)`, no free parameters.

Writing `S = w(contract) - w(none)` for the shift, and `gen R² = 1 - ||S-Ŝ||²/||S||²`:

| | gen R² | slope |
|---|---|---|
| all features, all pairs | 0.745 | 0.762 |
| brand block, spec-tied | 0.773 | 0.704 |

- **Direction generalizes**: ~0.75 with nothing fitted on the held-out contract.
- **Magnitude over-shoots**: slope < 1 in 7/8 models, `||Ŝ|| > ||S||` almost everywhere.
  **Two contracts together do LESS than the sum of what each does alone** — sub-additive, the
  opposite of the "conjunction bonus" reading.

### Finding: one shared scalar λ ≈ 0.7 fixes it — 0.77 → 0.97

Shrink the shift by `λ^(q-1)` where `q` = how many features the contract names. One λ fitted
**across all models**:

| | λ | additive | with λ | ceiling |
|---|---|---|---|---|
| all features | 0.676 | 0.745 | 0.893 | 0.924 |
| brand, spec-tied | 0.710 | 0.773 | **0.971** | 0.982 |

One parameter, essentially at the ceiling. A free conjunction vector per double would cost 312
parameters and transfer to nothing; λ costs one and transfers to all 39 doubles.

**Caveat:** we only see `q ∈ {1,2}`, so the functional form is *not* identified — `λ^(q-1)` and
`q^(-α)` fit identically and diverge at `q=3` (0.50 vs 0.58). **One triple contract settles it.**

### Finding: leakage has no shape of its own — it is (scalar) × (−δ)

Cosine between the two fitted leakage rows on the brand block: **0.92 mean, ≥0.97 in six of
seven models**; each vs `−δ`: 0.86 (screen), 0.94 (ram). So every contract shrinks the brand
preference *along its own direction*, and contracts differ only in how much.

That licenses the structured form — for a feature `g` the contract does **not** name:

```
w_g(x) = (1 - κ_g(x))·a_g ,   κ_g(x) = λ^(q-1) · Σ_{f named} κ_{f→g}
```

with `κ_ram→brand = 0.94`, `κ_screen→brand = 0.54`. Check: `0.71 × (0.94+0.54) = 1.05` — the
double cancels the brand preference and slightly reverses it, matching the observed net.

**Load-bearing and untested:** that `κ_{f→g}` depends on the *feature*, not the *level*. Every
constrained feature has so far been run at exactly one level. If it holds, 11 rows of `B`
collapse to 6 scalars and unseen contract levels become predictable.

### Why this matters: contract space is exponential, parameters need not be

96 possible contracts (`6×4×4`) = 1 none + 11 singles + 39 doubles + 45 triples. **We have run
4 — 4%.** Adding one 4-level feature takes it to 480.

| model | params for all 96 | |
|---|---|---|
| saturated, one `w` per contract | 1056 | hopeless |
| GRUM, free `B` main effects | 88 | ok |
| + free conjunction per double | +312 | does not generalize |
| **structured `B` + λ** | **18** | = a(8)+ρ(3)+κ(6)+λ(1) |

Structured params grow as `F²` (features), the space as `∏(L_f+1)`. With a 4th feature: 18 → 28
while the space goes 96 → 480. **Run the singles exhaustively (linear, 11); sample the
combinations (exponential, 84) — predicting those is the model's job.**

### Next runs, in priority order

| phase | contracts | conditions | runs (×8) | buys |
|---|---|---|---|---|
| **A** | `screen=13`, `screen=16`, `ram=4`, `ram=16` | 4 | 32 | **tests whether κ is level-independent** |
| **B** | `brand=Apple` + 2-3 others | 3-4 | 24-32 | does a *brand* contract leak into specs? |
| **C** | 4-6 doubles over all 3 feature-pairs + 2 triples | 6-8 | 48-64 | fits λ, identifies its form at `q=3` |
| **D** | ~8 random contracts, never fitted | 8 | 64 | honest held-out number |

**Phase A first** — cheapest, and if κ is level-dependent the 18-parameter model collapses and
everything downstream changes. It also supersedes the older open items (covers `ram=16GB` and
`screen=16-inch` anyway).

Caution when picking by D-optimality: ridged log-det *looks* like it prefers doubles (4.62 vs
3.21), but the 4 best doubles are **rank 6, not 7** —
`(13in+4GB)+(16in+16GB) = (13in+16GB)+(16in+4GB)` once the conjunction indicator is on. The ridge
hid it. **Check the rank, not just the score.**

**Prediction to check when Phase A lands:** if the mechanism is "naming a feature promotes it
above brand", the ram slopes stay near -0.94 and the screen slopes near -0.54 *regardless of
level*. If instead the slope tracks *how much the model has to give up*, `ram=16GB` (contract and
preference agree) comes out much closer to 0.

### Open

- [ ] **Is κ level-independent?** The 18-parameter model rests on it. Phase A.
- [ ] **Functional form of the saturation** — `λ^(q-1)` vs `q^(-α)`. Phase C.
- [ ] Put **model covariates** (family, log size) into `x` too, so one GRUM covers all 8 models
      and the size-scaling of compliance is an interaction coefficient.
- [ ] The margin is censored at the logprob floor in 88-99% of rows; a tobit is the honest
      response model. Winsorizing already showed directions are stable, so this is a robustness
      check, not a blocker.
- [ ] Template is pooled; make it a random effect and see if the leakage slopes move.
- [ ] gemma-1B breaks the parallel-leakage pattern (cosine 0.54 vs ≥0.97 elsewhere). Small-model
      artifact, or a real difference in how weak models handle contracts?
- [ ] GRUM gives us a better-shaped `v`. It still does **not** give `ΔW` — that needs a user
      utility `u`, which the laptops set does not have.

---

## 2026-08-10 — The models are LEXICOGRAPHIC, not additive. RAM is a veto, not a weight.

Chasing the "OLS loss function" worry from the entry below. That worry was wrong, and what
replaced it is much more important: **the additive utility model is the wrong model.**

> **Walkthrough:** [Notebooks/explainer_lexicographic.ipynb](../Notebooks/explainer_lexicographic.ipynb)
> defines every term used here (veto / tendency / tie-breaker / exchange rate / compliance /
> conflict set / override rate), writes the two competing models in LaTeX, and explains what
> each figure is measured from. Read that first; the results notebook is
> [lexicographic.ipynb](../Notebooks/lexicographic.ipynb).
>
> It also contains the cleanest single falsification of additivity, which is worth
> restating here: fit the brand weights **twice on the same run**, once on the 6750 pairs
> where RAM differs and once on the 3150 where RAM is tied. Additivity says the two must
> match. Mean brand spread is **0.88 vs 7.39 — 8.4x apart**. A "part" whose size depends on
> the company it keeps is not a separate part.
>
> And the reason the additive fit still looked fine: for **every** model, the largest
> possible brand swing plus the largest possible screen swing is smaller than the smallest
> RAM step (e.g. qwen-32B: 3.82 + 4.65 = 8.47 < 15.70). The fitted additive model already
> behaves lexicographically. It is not wrong about *who wins* — it is wrong about what its
> own numbers mean.

### The loss-function worry does not survive checking

- **Not heteroscedastic.** Residual SD by |fitted| bucket: 9.12 (0-5), 9.39 (5-10),
  7.91 (10-20), 7.54 (20+). Only 1.2x, and *smaller* at the extremes, not larger.
- **Extreme rows do not flip anything.** Dropping every row with |margin| > 20 (keeps 46-74%
  of the data): Apple's rank unchanged in **7/7** models, brand block correlation 0.971.
- Leverage in OLS depends on the design matrix, not on `y`, and our design is a balanced
  full factorial — so large-margin rows never had extra pull in the first place.

What is true is that the residual is huge in absolute terms: **SD ~8 log-odds**, against
brand effects of 1-3. We only resolve brand at all because n = 9900.

### Finding: RAM is an absolute veto

When the two laptops differ in RAM, **the one with more RAM wins 6750 out of 6750
comparisons — 100.0%**, for qwen-7B/32B/72B and gemma-4B/12B/27B. Not 99%. Every single one.

Split by step, so this is not "4GB is absurd":

| model | 4 vs 8 | 8 vs 16 | 4 vs 16 | screen, ram tied |
|---|---|---|---|---|
| qwen-7B | 100.0 | 100.0 | 100.0 | 58.8 |
| qwen-32B | 100.0 | 100.0 | 100.0 | 74.6 |
| qwen-72B | 100.0 | 100.0 | 100.0 | 69.0 |
| gemma-4B | 99.8 | 100.0 | 100.0 | 69.9 |
| gemma-12B | 100.0 | 100.0 | 100.0 | 77.9 |
| gemma-27B | 100.0 | 100.0 | 100.0 | 88.9 |
| gemma-1B | 82.4 | 91.6 | 99.7 | 57.7 |

Apple wins **0.0%** of the comparisons where it has less RAM (mean 2.89% including the noisy
gemma-1B), and **100%** where it has more.

So the priority order is **ram (absolute) > screen (partial, 58-89%) > brand (only when both
are tied)**. Screen is compensatory — brand can sometimes beat it. RAM never loses.

Not a string-position artifact: the item text is `"{screen} {brand} laptop with {ram} RAM"`,
so the order in the string is screen, brand, ram — which is not the priority order.

### What this breaks

An additive utility says features trade off: enough brand advantage beats a RAM
disadvantage. **There is no such trade-off.** So `brand_Apple = +2.21` against a ram spread
of 25 implies an exchange rate that does not exist at any price.

And the single brand weight is an average over two regimes that have nothing in common:

| comparison set used | qwen-32B | gemma-27B | Apple's rank |
|---|---|---|---|
| all 990 pairs | +2.21 | +2.74 | 1 |
| same ram | +5.16 | +7.98 | 1 |
| same ram + same screen | +7.31 | +10.79 | 1 |

Brand and ram are exactly orthogonal in the design (max |r| = 0.0000), so this is not a
collinearity artifact — brand genuinely has **no** effect when ram differs and a large one
when it does not. Apple's win rate makes the same point with no model: **50.0%** when ram
differs, **67-87%** when ram is tied.

**Apple's rank is 1 in all three sets for all 7 models.** Ranks are invariant; magnitudes
are not even approximately meaningful.

### Consequences

- **Never quote a brand weight without the comparison set it came from.** It can triple.
- The additive R² of ~0.85 is not evidence that the additive model is right. It is high
  because ram and screen dominate; the additive form gets the big things right and the
  brand block is a rounding error inside it.
- This reinforces C6 rather than undermining it. Under a contract, "satisfies the contract"
  slots in at the top of the priority list, pushing brand further down — which is exactly
  the gradient (1.16 → 2.29 → 3.46) measured in the entry below.
- It also explains the near-total saturation: a lexicographic rule produces certainty, so
  the log-odds go to ±25 whenever the top-priority feature differs.

### Answered the same day: does the contract change the order?

Tested in [Notebooks/lexicographic.ipynb](../Notebooks/lexicographic.ipynb). **Prediction was
half right.** The contract does take the top slot — but only when it is not fighting a veto.

The test is a direct conflict. Under `ram=8GB`, the contract and the model's own order point
at different laptops in exactly the 8GB-vs-16GB pairs (contract says 8GB, the model's own
rule says 16GB). Under `screen=14-inch` with ram tied, the same conflict is 14in vs 16in.

% of conflicts won by the **compliant** laptop:

| model | 8GB vs 16GB, no contract → under `ram=8GB` | 14in vs 16in, no contract → under `screen=14-inch` |
|---|---|---|
| qwen-7B | 0.0 → **60.0** | 39.3 → **96.4** |
| qwen-32B | 0.0 → **59.1** | 29.2 → **100.0** |
| qwen-72B | 0.0 → **84.8** | 34.0 → **100.0** |
| gemma-1B | 8.4 → 10.0 | 45.5 → 37.1 |
| gemma-4B | 0.0 → 15.9 | 24.8 → 49.7 |
| gemma-12B | 0.0 → **54.5** | 20.5 → **100.0** |
| gemma-27B | 0.0 → **93.2** | 11.2 → **100.0** |

- **Against the screen *tendency*** (58-89% unconstrained), the contract installs a **new
  absolute veto** — 96-100% for the five capable models. The order genuinely changes.
- **Against the RAM *veto***, the contract does not take the top slot. It **breaks** the veto
  (0% → 54-93%) but nothing replaces it: RAM and compliance now genuinely trade off. The
  model becomes *more* compensatory under a contract, not differently lexicographic.
- **Control:** where the contract is silent (4GB vs 16GB, neither complies) the RAM veto is
  untouched, 97-100%. And where contract and preference agree (8 vs 4), 100%. So the contract
  acts only in the conflict region — it does not make the model generally erratic.

**Reading: a contract's power depends on what it is fighting.** Asking for something the
agent was flexible about is obeyed completely; asking it to give up the thing it always wants
is obeyed only partly. This is the mechanism behind the C5/C6 ram-vs-screen asymmetry.

### Finding: "contract override rate" is a better adherence metric than the weight

The % of genuine conflicts the contract wins. Model-free, a percentage, immune to the scale
problem. On `ram=8GB` it scales sharply with size:

- **gemma: 1B 10% → 4B 16% → 12B 54% → 27B 93%** (monotonic)
- qwen: 7B 60% → 32B 59% → 72B 85%

The small models are not vaguely "less obedient" — they fail at exactly one thing:
**overruling their own strongest preference on request.** Where nothing has to be given up,
even gemma-1B complies at 97-100%.

### Open

- [ ] Refit with a model that can express this — fit brand **within** spec-tied comparisons
      only, and report ram/screen as an order rather than as weights. The additive fit stays
      useful for ranking, not for exchange rates.
- [ ] **`ram=16GB` is now the sharpest missing run for a second reason**: it is the case
      where the contract and the model's own order **agree**. Prediction: near-100%
      compliance for every model including gemma-1B, because nothing has to be sacrificed.
      That separates "obeying" from "giving something up".
- [ ] Does the override rate keep rising above 72B, or saturate?
- [ ] Check it on the earlier colors/foods/cars sets — is lexicographic behaviour general,
      or an artifact of these three very legible laptop features?

---

## 2026-08-10 — C6 in ranks only: even the *compliant* Apple loses its place

Fig 10.4 in [Notebooks/figs4deck5.ipynb](../Notebooks/figs4deck5.ipynb). Motivated by the
scale note below — magnitudes are not trustworthy, rankings are — so C6 was redone with no
magnitudes at all.

### The C6 ordering survives ordinally

Apple's rank among the 5 brands inside a fixed `(screen, ram)` cell, 7 models x 9 cells:

| | mean rank | median | Apple is rank 1 |
|---|---|---|---|
| no contract | 1.16 | 1 | 89% |
| contract, cell **satisfies** | 2.29 | 2 | 49% |
| contract, cell **violates** | 3.46 | 4.5 | 25% |

So C6 does not depend on the log-odds scale. Good.

### But the pooled split hides which contract did it

Splitting by phase (Fig 10.4 is now one panel per phase — the phases contribute very
unequal numbers of cells: 3, 3 and **1** satisfying cell respectively):

| phase | satisfies: mean rank / % rank 1 | violates: mean rank / % rank 1 |
|---|---|---|
| `14in` | **1.62 / 67%** | 2.67 / 40% |
| `8GB` | 2.71 / 38% | 3.67 / 21% |
| `14in+8GB` | 3.00 / 29% (n = 7) | 3.91 / 16% |
| (no contract) | 1.16 / 89% | — |

**Under the screen contract the premium largely does survive on compliant items**
(1.16 → 1.62, still top brand in 2 of 3 cells). Under the ram contract it does not
(→ 2.71, top in 38%). The pooled "2.29 / 49%" was an average of these two very different
regimes.

This is the same asymmetry as C5 (ram disrupts brand more than screen at equal instruction
strength) and it now has an obvious explanation from the lexicographic entry above: **ram is
a veto, screen is only partial.** Naming ram promotes an absolute filter above brand; naming
screen promotes a feature that brand can still sometimes beat.

So the honest one-line version of C6 is: *stating a spec pushes brand down the priority
order, and how far depends on how absolute that spec already was.*

### But "the premium survives where the contract is met" was too generous

The earlier entry read `satisfy +2.06` as the premium surviving. Ordinally it does not:
Apple is top brand in only **49%** of satisfying cells, down from 89%. A positive
sum-to-zero weight does not mean rank 1.

The sharp test: take the feasible set `C` (the laptops that satisfy the contract) and ask
where Apple's best compliant laptop sits **among the compliant laptops only** — so
non-compliant items cannot contribute at all.

| model | 14in (\|C\|=15) | 8GB (\|C\|=15) | 14in+8GB (\|C\|=5) |
|---|---|---|---|
| qwen-7B | 5 | 9 | 5 |
| qwen-32B | 2 | 9 | 5 |
| qwen-72B | 5 | 13 | 5 |
| gemma-1B | 2 | 6 | 3 |
| gemma-4B | 4 | 5 | 5 |
| gemma-12B | 3 | 11 | 5 |
| gemma-27B | 4 | 11 | 5 |

**Before the contract it is rank 1 in 21/21.** After, mean rank 3.6 / 9.1 / 4.7. Under the
double contract `14in+8GB`, `C` is 5 laptops differing only in brand, and Apple is **last in
6 of 7 models**.

**Revised reading of C6:** the contract does not merely strip Apple's premium from
non-compliant items. It demotes Apple *everywhere*, just much harder where the contract is
missed. The compliance split is a gradient (1.16 → 2.29 → 3.46), not "premium kept vs
premium lost". The contract-theory sentence should be "the agent's preference is
**re-weighted, and conditioned on compliance**", not "preserved on compliant items".

This does not change any verdict in the 08-10 conjecture table — C1-C5 are still rejected,
C6 is still the surviving explanation — but it changes what C6 claims.

### Open

- [ ] Is the demotion of the *compliant* Apple a brand effect or a spec effect? Under `8GB`,
      `C` still varies in screen, so screen weights contribute. `14in+8GB` is the clean case
      (\|C\|=5, brand only) and it is the worst one — which argues brand, not spec. Worth
      confirming on `screen=16-inch,ram=16GB` when it is collected.

---

## 2026-08-10 — The scores ARE log-odds (correcting the saturation caveat), and PMI is a no-op

### Correction: the margin is a genuine log-odds

The earlier saturation note said our "log-odds" are not calibrated log-odds. **That was
wrong.** Checked on `laptops_robustness/68373841` (qwen-32B, unconstrained):

- `P("Option 1") + P("Option 2") = 1.000` in every row (median and 5th percentile both
  1.000). The two labels absorb the whole distribution, so conditioning on them loses
  nothing.
- `score_a - score_b = (lp1 - lp2) - (base_1 - base_2)`, and `lp1 - lp2` is exactly the
  log-odds of choosing option 1. The base term is a **single global constant** (+2.2505),
  identical in all 5 templates, so `gamma` absorbs it completely.

So the margin is a real log-odds of the binary choice, shifted by a constant. Nothing to fix.

### Finding: PMI normalization currently does nothing

The null context is `prompt.split('\n')[-1]`, and every prompt ends `"Answer: "` — the same
string regardless of template, items, model or constraint. So the PMI base is one constant
per label per run: `base_1 = -0.9312`, `base_2 = -3.1816`, difference +2.2505 in **all five
templates**. Subtracting it shifts every margin by that constant, which the intercept eats.

**PMI is a no-op for every feature weight we report.** Its only effect is to move `gamma` by
2.25. This raises the priority of the open question "which null context for PMI?" — right
now the answer is "it does not matter, because it is not doing any work".

Worth noting what it *does* say: with no content at all the model answers "Option 1" with
p = 0.39 and "Option 2" with p = 0.04 — a strong prior for **option 1**. Yet `gamma` is
about **-7**, favouring option 2. So the positional bias is not a label prior; it survives
removing one, and points the other way.

### The real remaining problem, stated properly

Not the scale — the **loss function**. 59-87% of rows have `|margin| > 10` (P > 0.99995),
and up to 44% exceed 20. Behaviourally, log-odds 15 and 30 are the same event ("always").
But OLS treats that 15-unit gap as exactly as important as 0 → 15, which is the difference
between a coin flip and near-certainty. So the fit spends most of its effort in the region
where nothing is happening.

How much does it matter? Winsorizing the margin at ±10 and refitting:

- every weight shrinks by roughly half (expected — smaller range)
- brand block correlates **0.967** with the uncapped fit
- **Apple's rank is unchanged in 23/28 runs**; the 5 changes are all near-ties in qwen-7B
  and gemma-1B, the two models with the smallest brand effects

**So: directions are safe, magnitudes are not.** Do not quote a brand weight as if its size
were meaningful. Rankings, signs and win rates are the trustworthy outputs.

### Can we calibrate the scale?

Not in the usual sense. Calibration needs a ground truth, and for a *preference* there is
no fact of the matter — nothing says whether the model "really" prefers Apple at 99% or
99.99%. Classical temperature/Platt scaling has nothing to fit against. Three things are
genuinely available instead:

1. **Calibrate the zero point** — the identical-options test (already on the list below).
   Compare an item with *itself*; the true preference is exactly 0, so any margin is pure
   bias. Measures `gamma` + label bias directly instead of inferring it. 45 items x 5
   templates = 225 prompts, minutes of GPU.
2. **Fix the loss, not the data** — logistic BT on who-won, or winsorize. Both tested, both
   agree with the current fit on direction.
3. **External validity** — the only real answer. Does the elicited preference predict what
   the model *does* when actually asked to recommend a laptop? That is Phase 2 (active
   delegation), and it validates the elicitation regardless of what the number 25 means.

---

## 2026-08-10 — Figures for deck 5, and the first measurement of deck 3's "robustness"

New notebook: [Notebooks/figs4deck5.ipynb](../Notebooks/figs4deck5.ipynb). One header per
planned slide, 24 figures, all written to `figs/deck5/*.png` for dropping into PowerPoint.
It re-derives everything from `data/` off the new config schema — no numbers are copied
from the other notebooks.

### Finding: contract robustness is all-or-nothing, and it is not about size

Deck 3 defined **robustness** as `{i : x*_i = y_i}` — does the agent's own top choice obey
the contract? Never computed until now. Taking `x* = argmax` of the fitted additive utility
over all 45 laptops:

| model | 14in | 8GB | 14in+8GB | rank of best compliant laptop (8GB) |
|---|---|---|---|---|
| qwen-0.5B / 7B / 32B / 72B | 100% | 100% | 100% | 1 |
| gemma-12B / 27B | 100% | 100% | 100% | 1 |
| **gemma-1B** | 0% | **0%** | 0% | **16** |
| **gemma-4B** | 100% | **0%** | 0% | **16** |

(`top1` compliance; the top-5 numbers tell the same story.) Six of eight models put a
compliant laptop first **and** fill their whole top-5 with compliant ones. The two small
Gemmas fail outright — gemma-1B's best compliant laptop sits at rank 16 of 45 under `8GB`
and rank 18 under `14in+8GB`, i.e. it would hand the user something that ignores the request.

Two cautions before this goes on a slide:

- **qwen-0.5B "passing" is not adherence.** Its constrained-feature weight is only +0.30,
  but its brand spread is ~0.1, so a tiny push is enough to decide the argmax. It complies
  because it has no competing preference, not because it listened. This is the opposite
  failure mode from gemma-1B and should be labelled as such.
- Robustness is measured on the **fitted utility**, so it inherits the saturation caveat.
  It is a statement about the elicited `v`, not about a generated recommendation.

**Reading:** adherence measured as a *weight* (0.5B +0.30 → 32B +13.78) looks like a smooth
scaling curve. Measured as a *choice* it is a step function — you either get a compliant
argmax or you do not, and the two small Gemmas are the only failures. Worth saying out loud
that the two metrics disagree about what "the model listens" means.

### Still not ΔW

This is only half of deck 3's Aim-2 metric. The welfare gap still needs a user utility `u`,
which does not exist for this item set. Open item from the previous entry stands.

---

## 2026-08-10 — Config schema changed, collection scripts merged

### Config: constraints are now a dict, not a slug

Schema written up in [docs/config_schema.md](config_schema.md) and
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
