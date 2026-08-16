# Status — What Do LLMs Want?

**2026-08-11.** Built by processing all 22 meeting notes, the thesis proposal, and
`progress.md`. Items marked **[?]** still need a verdict: keep, or drop.

**Working mode right now: two tracks in parallel.** Track 1 — the missing runs are queued in
`scripts/slurms.sh` (40 jobs), Lotan launches them on the cluster. Track 2 — while they run,
analyse the data we already have (section 4B).

This file = where we are and what is next. It is rewritten in place, not appended to.
- History of findings → [progress.md](progress.md) (append-only, never rewritten)
- Method → [grum_formalization.md](grum_formalization.md)
- Data contract → [config_schema.md](config_schema.md)

---

## 1. Where we are

**Phase I (passive elicitation), on the hardest task complexity of the three.**

The proposal's ladder is *simple → conditional → conflicting*. We are on **conflicting**
(laptops as feature vectors: brand × screen × ram) and doing **conditional** on top of it
(contracts in the prompt). The simple sets (colors/foods/cars/stocks) are done and parked.

The live question is **consistency and robustness**: is there a stable preference, and what
survives a contract?

### Drift from the proposal — decide whether to update the proposal or come back

| Proposal (Feb 2026) | What we actually do | Verdict |
|---|---|---|
| Perplexity of the item string as the score | logprob of answer token `"1"` / `"2"` | better, keep — but the proposal text is now wrong **[?]** |
| Prompt templates generated at runtime by a separate LLM | fixed set, first 5 templates | **[?]** does the diversity claim still hold? |
| Qwen-2.5 **and Qwen-3** | Qwen-2.5 + Gemma-3, no Qwen-3 | keep Gemma; **add Qwen-3 later**, not now (decided 08-11) |
| BT, with GRUM as fallback if IIA/transitivity break | GRUM is now the main model | happened as planned |

---

## 2. What we know (settled — do not re-argue)

1. **The models are lexicographic, not additive.** RAM acts as a veto, not a weight. The
   additive fit is fine for ranking, not for exchange rates.
2. **Adherence splits in two** — *numbers revised 2026-08-12, the old ones used `sign(margin)`
   and were corrupted by PMI + position bias.* Position-corrected, the agree direction is
   **exactly 100.0 for all 8 models**, and the conflict direction is **near-binary**: a model
   either fully pays (98.7–100: qwen-7, qwen-72, gemma-27, qwen-0.5) or fully refuses
   (0–22.7: gemma-1, gemma-4, gemma-12, **qwen-32**). "Did it hear?" and "will it pay?" are
   different questions, and only the second discriminates.
   **Adherence is NOT monotone in size for qwen** (qwen-32B at 17.3% sits between qwen-7B at
   98.7% and qwen-72B at 100%); it still is for gemma.
3. **Apple's brand premium is conditional on compliance.** Under a contract, even the
   *compliant* Apple loses its rank (rank 1 → 5). It is one brand, not brand preference in
   general.
   **Sharpened 2026-08-16.** The decay is *not* a fitting artifact — it survives a model-free,
   within-spec-cell measurement (+7.45 → −0.88 under `ram=8GB`) — and it is *not* about the
   contract asking for a modest level: Apple falls under **every** constraint, including
   `ram=16GB` and `screen=16-inch`. What does predict it is **prompt specificity**: removing the
   frame sentence entirely (`bare`, no constraint) *raises* Apple by +1.35 on average, 7/8 models.
   Working hypothesis: Apple is the **default answer to an underspecified question**, and any
   added specificity moves the model off it. The off-dimension-constraint control (A6) would
   settle it and has never been run.
4. **The contract effect is one shared scalar**, λ ≈ 0.7. Leakage has no shape of its own —
   it is (scalar) × (−δ). The model generalizes to a contract it never saw in *direction*,
   not in magnitude.
5. **Positional bias decides 15–30% of all comparisons.** γ is now **corrected** (2026-08-16,
   `figs4deck5_fix.ipynb`): γ = γ_fit + C. Corrected, it runs −4.73 … +3.53 instead of −10.5 …
   −0.2. It still beats the whole brand range for 4 of 7 models, but **not for gemma-4B/12B/27B**,
   and **its sign is not universal** — gemma-1B and gemma-4B favour slot A, the rest slot B. The
   old all-negative picture was the PMI constant, not the model. qwen-0.5B's γ is **unknown**
   (never saturates).
6. **Naive p-values overstate confidence ~3×.** Use clustered SEs. Confirmed 2026-08-12:
   clustering the bootstrap on item-pairs widens CIs by a median 2.28×.
7. **PMI is a no-op for the weights, but not for γ and not for `sign(margin)`.** It subtracts
   one constant `C` per run (`C` = +2.25 to +10.50), which γ already absorbs. `sign(margin)`
   flips for 0.7–16% of rows. **Decision 2026-08-12: drop PMI** (`normalize_pmi=False`), and use
   position-corrected `D = (m(x,y) − m(y,x))/2` for every "who wins" question — `D` is immune to
   any constant. No re-collection needed; `C` is recoverable for the 7 saturating models.
8. The **winner** token is saturated in 88–99% of comparisons, but the margin is **not
   censored**: the loser has full dynamic range and tobit equals OLS to 2 decimals.
9. Colors: violate WST, but satisfy RUM. Alignment shifts colors toward purple, away from red.
10. **The BT weight scale is a nuisance parameter, not preference strength** (2026-08-12).
    The log-odds does not depend on how big the difference is — a 4× RAM gap gets the same
    log-odds as a 2× gap (additivity ratio 0.5 in all 8 models, including the one with zero
    saturation). Rank-1 explains 99.64% of the variance across weight vectors: bigger models
    are **louder, not different** (max angle 9.1°, length changes up to 113%). There is **no
    significant size trend**, and weight spread has correlation −0.04 with held-out accuracy.
11. **Of the two measurement bugs found 2026-08-12, the γ one is fixed** (2026-08-16, in
    post-processing — see 5 and `figs4deck5_fix.ipynb`). `C` is pooled over each model's 4 runs,
    slightly tighter than the 08-12 single-run figures (gemma-1B → +3.53, not +3.31).
    **Still open:** the pooled OLS shrinks brand 3–5×, so the "RAM dominates" gap is partly an
    artifact of pooling.
    **Also settled 2026-08-16:** the preference gap `g` behind κ must be measured **on the
    conflict pairs directly**, not read off the additive fit — the fit inflates κ exactly where
    the contract fights a strong preference. Direct, κ > 1 ⟺ win rate > 50% in all 16 cells.
12. **qwen-0.5B has a real but tiny preference** — *corrected 2026-08-12*, it was previously
    listed here as having none. Template-FE R² = 0.920 (pooled 0.085); weights 33 SD above a
    permutation null. But its signal never beats its own noise (`spread/resid_SD` = 0.80 vs
    4.5–5.3) and it is inconsistent per-choice (order agreement 0.6%). Different scale regime,
    not a broken measurement.
13. **Five templates is enough.** Subsampling 5 of 43 gives SE ≈ 1.0 on total spread, unbiased,
    against a between-model range of 22.8. Template FE change the weights by exactly 0.000
    (the design is perfectly balanced).
14. **The preferences are pretrained, not aligned in** (2026-08-16, qwen-7B base vs instruct).
    Scale-free weight vectors correlate at **r = 0.990** over 44 weights; alignment is **~9×
    louder** and nothing else. The Apple decay is in the base model at the *same* normalised
    size (swing 0.036 vs 0.039), and the positional bias is proportionally the same
    (|γ|/S = 0.088 vs 0.098). The base model also "adheres" (79% on the ram conflict) — but for
    it that is **text coherence, not obedience**, which weakens "the model obeys" as a reading
    of the aligned numbers too. One model, one size; needs the other sizes and gemma-pt.

---

## 3. Data we have

`laptops_robustness` (45 items = 5 brand × 3 screen × 3 ram):

| family | sizes | constraints collected |
|---|---|---|
| qwen-2.5 | 0.5, 7, 32, 72 | `none`, `screen=14-inch`, `ram=8GB`, `ram=8GB+screen=14-inch` |
| gemma-3 | 4, 12, 27 (+1B elsewhere) | same four |

**Every constraint we have ever run asks for a mid or low level.** No top-level constraint
(`ram=16GB`, `screen=16-inch`), no brand constraint, no off-dimension constraint, no
unframed prompt. This is the single biggest hole in the data.

Also: `laptops_num_vs_txt` — **qwen-7B only**, num + txt. The plan called for 7B *and* 72B.
Earlier sets: `pmi_qwen`, `qwen_pt` (colors/foods/cars/stocks/laptops/laptop_brands, 4 sizes each).
Note the old `qwen_pt` runs used the **instruct** templates (`"...Answer: "`) — checked in
`data/qwen_pt/68220852/config.json` — so they are not comparable to the new base-model runs.

**Base models, new 2026-08-16.** `laptops_robustness_pt` — qwen-7B base, four contracts,
`pretrained` templates. **Collected** (jobs 1305867–70). Analysed in
[pretrained_vs_instruct.ipynb](../Notebooks/pretrained_vs_instruct.ipynb).

**GRUM Phase A has landed** for 7 of 8 models — every model now has all 8 constraint conditions
**except qwen-72B**, which is missing `screen=13-inch` and `ram=16GB` (6 of 8). Re-run those two.

---

## 4. Next — ordered

### A. Runs to collect (cluster) — Track 1

**Queued now in `scripts/slurms.sh`: 4 jobs — qwen-7B *base* (`qwen-pt`) on the four
contracts**, into `data/laptops_robustness_pt/`. First use of the `pretrained` template set.

⚠️ **This replaced the 40-job Phase A + bare-frame batch in `slurms.sh`** (2026-08-16). Those
two batches are still in `create_slurms.py`, commented out — uncomment them and set
`exp_name` back to `laptops_robustness` to regenerate. They are items 1 and 2 below.

The 40 jobs, when regenerated, go to `data/laptops_robustness/` (gemma included — analysis
filters on `config.json`, so one folder is simpler than two):

1. **GRUM Phase A** — `screen=13-inch`, `screen=16-inch`, `ram=4GB`, `ram=16GB`, all 8 models
   (32 runs). Tests whether κ is level-independent; the 18-parameter GRUM rests on it. Also
   subsumes the older "top level of each feature" item and completes the 2×2 against the
   `14-inch` / `8GB` runs. `ram=16GB` is the sharpest single test: Apple is strongest there
   unconstrained, and it is the one case where contract and model preference *agree*.
2. **`--frame bare`, no constraint**, all 8 models (8 runs). Every run so far carries
   *"I am looking to buy a laptop."* — we have never measured brand preference without it.

**Prediction to check when Phase A lands:** if the mechanism is "naming a feature promotes it
above brand", the ram slopes stay near −0.94 and the screen slopes near −0.54 *regardless of
level*. If instead the slope tracks how much the model must give up, `ram=16GB` comes out
much closer to 0.

Queued behind it (commented in `create_slurms.py`, uncomment when Phase A lands):

3. **A brand constraint** (`brand=Apple`, `brand=Dell`, `brand=ASUS`) — does a brand contract
   leak into specs?
4. **`laptops_num_vs_txt` on 72B** — finishes Nir's Experiment 1, which has only ever had 7B.
5. **Instruction-strength ladder** (Nir, 08-04): user side *"I prefer"* vs *"I must"*; model
   side *"what do you prefer"* vs *"choose one"*. Needs new `FRAMES` entries plus a strength
   flag on the constraint sentence — a small change to `data_collection.py`, so ask first.
6. **Off-dimension constraint** (e.g. "I prefer a lightweight laptop") — separates "any extra
   sentence" from "a spec the items can meet or miss".
7. **Gibberish-feature control group** (Nir 08-04; Itay 05-20 "think hard about your control
   group"). Same request twice, never done. Needs a new item set in `alternatives.py`.
8. GRUM Phases B/C/D — doubles, triples, held-out contracts.
9. **A wider feature ladder** — RAM 4 / 8 / 64 / 256 GB, or a price ladder (new item set in
   `alternatives.py`). *Created by the 2026-08-12 finding:* the log-odds is currently flat in
   the size of the difference (a 4× RAM gap scores the same as a 2× gap), but the tested range
   is narrow. If `L(4→256) ≈ 2 × L(4→16)`, the magnitude *does* carry cardinal information and
   the categorical reading fails. Cheap, and it settles whether any BT weight we report can be
   read as a magnitude at all. **Arguably the highest-value run in this list.**
10. **A repeat run of one model at fixed size** (e.g. qwen-32B twice). There is no within-model
    replicate anywhere in the data, so run-to-run noise is currently unmeasurable.

### B. Analysis with data we already have — Track 2, start here

9. **Soft-robustness** (Nir 08-04, defined 08-11): compliance is **not** pass/fail. Under a
   `14-inch` contract, a 13-inch is a near miss and a 16-inch is a bigger one — score by
   **distance from the requested level**, not by a binary violation. Redo the adherence
   numbers this way and see whether the hard "did it pay?" gap (2%–87%) softens into a
   gradient. Needs no new data; screen and ram are both ordered.
10. **Nir's Experiment 4** (positional bias): when a contract makes *both* laptops bad, is the
    bias very strong? When one is clearly better, is it weak? Never done, needs no new data,
    and it is the cleanest test of "positional bias = the model's way to express indifference".
11. **Full-item vs feature parametrization** — how *similar* and how *good* (likelihood).
    Nir 08-04, untouched. Also move `fit_full_item_bradley_terry` out of the notebook.
12. **Report Exp 1's number**: corr(β_num, β_txt), Spearman/Kendall. Data exists for 7B; 72B
    is queued (A4), so report 7B now and add 72B when it lands.
13. **Variance decomposition**: how much variance from order (γ), from persona. *Template part
    done 2026-08-12: template FE change the weights by exactly 0.000.*
14. ~~**Does the weight scale grow with model size for a real reason?**~~ **Answered
    2026-08-12** — no, and there is no significant trend to explain. See `progress.md` and
    [Notebooks/weight_scale_vs_model_size.ipynb](../Notebooks/weight_scale_vs_model_size.ipynb).
    Three follow-ups it created: fix γ, decide on design-cell estimation for brand, and the
    **wider feature ladder** run (below, A9).
15. ~~**Rational-choice axioms on laptops**~~ — **done 2026-08-12** as part of the above.
    Above 0.5B, transitivity is essentially perfect (WST/MST/SST violations ≤0.6%, 1–45
    3-cycles of 14190). qwen-0.5B violates badly (SST 0.158, 453 3-cycles).
16. **Define a user utility `u` for laptops** so ΔW can be computed. Without it, Aim #2 has
    no metric and every number we report is about `v` alone. **Blocking.**

### C. Method fixes (small, known)

00. **Found and mostly fixed 2026-08-16.** `(model_family, model_size, constraints_id)` is **no
    longer a unique run key** — the bare-frame runs collide with the shopping runs on
    `constraints_id == "none"`, 8 collisions, one per model.
    **The rule** (now in `CLAUDE.md`): the frame is a dimension of the experiment and is held
    **fixed inside any comparison** — a contract comparison pins `frame="shopping"` (every
    constrained run is shopping), a frame comparison pins `constraints={}` (bare exists only
    unconstrained). There is no global "canonical baseline".
    **Done:** both `figs4deck5_fix.ipynb` and `pretrained_vs_instruct.ipynb` filter on frame and
    assert uniqueness; re-executed with every number unchanged (the bug was latent, not active).
    **Left:** `load_scores_by_run` in `src/auxiliary.py` cannot filter on frame — `frame` is a
    nested dict, so `{"frame": "shopping"}` never matches. Needs dotted-key support or a `frame=`
    argument. A `src/` change, so it needs the go-ahead.

0. **Highest priority, found 2026-08-12, partly done:** re-derive every adherence / override /
   violation number **position-corrected**. The published ones use `sign(margin)` and are off by
   up to 33 points; "adherence scales with model size" does not survive for qwen.
   **Done:** `figs4deck5.ipynb`'s shared `winrate()` (08-12), and Figs 5.5 D / 6.2 / 12.1 rebuilt
   from scratch in [figs4deck5_fix.ipynb](../Notebooks/figs4deck5_fix.ipynb) (08-16) with the
   corrected γ and the direct-pairs gap. `normalize_pmi=False` is set in `src/agent.py`.
   **Left:** `lexicographic.ipynb`'s own `wins()`, and deck-5 Figs 5.3 / 5.4 / 10b.3.
17. Explicit sum-to-zero coding in `fit_feature_based_bradley_terry`; pin references with
    `pd.Categorical` (today ram's reference is `16GB` by string sort — fragile).
18. Clustered SEs as the default.
19. PMI null-context calibration test: compare two *identical* options, pick the null context
    that gives a zero margin.
20. Label scheme: `A/B`, `1st/2nd`; add a third "equal / don't care" option; check the option
    tokens are actually in top-k.
21. PriDe-style prior correction ([arxiv 2309.03882](https://arxiv.org/abs/2309.03882)).
22. Tournament-graph connectivity check.
23. Move `fit_full_item_bradley_terry` from the notebook into `src/pref_models.py`.

### D. Phase II groundwork
24. **Play with GPT commerce** to get a feel for domains. Nir asked twice (05-12, 07-22, the
    second time in bold with "think about this a lot"). Never done.
25. **Personal taste vs generic product** — books vs laptops.
26. **Something the model plausibly has preferences about vs something alignment plausibly
    hides** (07-22). Untouched, and it is close to the core conjecture.
27. **Free choice, not pairwise**: *"I want 14-inch, pick a laptop"* — generation, then map
    the choice back to an item. This is the actual Phase II move. Today we only do the
    pairwise version.
28. Decide: do we measure **the choices themselves** or **the induced order**? (07-22, still open)
29. Complete the domain-requirements checklist and score candidate domains against it (02-26,
    two explicit checkboxes, never closed).

### E. Reading
30. The positional-bias papers Nir sent (NAACL 2024 order effects; ACL 2025 judge bias;
    PriDe). **[?]** still worth it, or has our own measurement overtaken them?
31. Papers doing *evaluation* on similar tasks; check Asta (AI2); userLLM paper (Itay, 05-20).
32. CP-nets book; "Short Introduction to Preferences"; Mallows mixtures; Lirong Xia (Reshef, 02-23).

---

## 5. Parked

- **Phase III**: congestion game — users with personas, model picks a travel itinerary, show
  congestion falls as users communicate more. Domain must be one where congestion is *not*
  already present in real life.
- CP-nets as the opposite extreme to the lexicographic assumption.
- Personas as `v_p` (turn on `v_cars`) — partly subsumed by GRUM's `x` vector.
- Sampling a winner from the logits per BT (04-16) — superseded by soft labels. **[?]** drop?
- Binary features as a starting point (07-22) — we went straight to 3 levels. **[?]** drop?
- Strategic/adversarial prompt edits to move preferences (Reshef, 02-23).
- Alignment as the A → B preference shift; what is A, and can we steer B? (11-03)
