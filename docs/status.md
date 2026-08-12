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
2. **Adherence splits in two.** Every model with real signal is at ~100% when obeying the
   contract is *free* (agree direction). What varies from 2% to 87% is whether it will *pay*
   for it (conflict direction). "Did it hear?" and "will it pay?" are different questions.
3. **Apple's brand premium is conditional on compliance.** Under a contract, even the
   *compliant* Apple loses its rank (rank 1 → 5). It is one brand, not brand preference in
   general.
4. **The contract effect is one shared scalar**, λ ≈ 0.7. Leakage has no shape of its own —
   it is (scalar) × (−δ). The model generalizes to a contract it never saw in *direction*,
   not in magnitude.
5. **Positional bias decides 15–30% of all comparisons** — bigger than most preferences we
   measure.
6. **Naive p-values overstate confidence ~3×.** Use clustered SEs.
7. **PMI normalization is currently a no-op.**
8. The margin is a genuine log-odds, but **saturated in 88–99% of comparisons**.
9. **qwen-0.5B has no measurable preference** on anything.
10. Colors: violate WST, but satisfy RUM. Alignment shifts colors toward purple, away from red.

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

---

## 4. Next — ordered

### A. Runs to collect (cluster) — Track 1

**Queued now in `scripts/slurms.sh`, 40 jobs.** Both batches go to `data/laptops_robustness/`
(gemma included — analysis filters on `config.json`, so one folder is simpler than two):

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
13. **Variance decomposition**: how much variance from order (γ), from template, from persona.
    Template is currently pooled — make it a random effect.
14. **Does the weight scale grow with model size for a real reason, or is it an optimization
    artifact?** Asked twice (06-10, 06-18), never answered.
15. **Rational-choice axioms on laptops** — WST/MST/SST, Block–Marschak. Done on colors only.
16. **Define a user utility `u` for laptops** so ΔW can be computed. Without it, Aim #2 has
    no metric and every number we report is about `v` alone. **Blocking.**

### C. Method fixes (small, known)
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
