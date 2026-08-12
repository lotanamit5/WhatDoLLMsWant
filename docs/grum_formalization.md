# GRUM for contracts — formalization

Status: **proposed**, 2026-08-10. Replaces "one Bradley-Terry fit per phase" with one model
fitted across phases, **which predicts contracts it was never shown**.
Real numbers in [Notebooks/grum.ipynb](../Notebooks/grum.ipynb).

> **How to read this.** Part I (§1–§4) explains the model with one tiny made-up example — four
> ice creams. No real data, no big tables. Part II (§5–§8) applies it to the laptops and gives the
> measured numbers. Part III (§9–§11) is the careful stuff: identification, the lexicographic
> threat, what is still open.

---

# Part I — The model, with a tiny example

## 1. What a GRUM is, in words

A **random utility model** says: a chooser gives every option a score ("utility"), and picks the
highest one. The score is not exactly known, so a bit of randomness is added.

A **General** RUM (GRUM) adds one thing: the score depends on **who is choosing** *and* **what
the option is**, through their features. Two people can rank the same options differently,
because the model knows something about each person.

That is the whole idea:

> **option features × chooser features → a score.**

For us the "chooser" is not a person. **It is the contract we put in the prompt.** Different
contract = different chooser = different ranking of the same laptops. That is the mapping this
whole document is built on, and §5 makes it precise.

---

## 2. The running example: four ice creams

Forget laptops for now. Our shop sells **4 ice creams**, made of two features:

- **flavour**: chocolate or vanilla
- **size**: large or small

|  | flavour | size |
|---|---|---|
| 🍫L | chocolate | large |
| 🍫S | chocolate | small |
| 🍦L | vanilla | large |
| 🍦S | vanilla | small |

We write each feature as $+1$ or $-1$, so each ice cream is a little vector $z$:

$$z = (\underbrace{\text{flavour}}_{+1\,=\,\text{choc}},\; \underbrace{\text{size}}_{+1\,=\,\text{large}})$$

so 🍫L $= (+1, +1)$, 🍫S $= (+1, -1)$, 🍦L $= (-1, +1)$, 🍦S $= (-1, -1)$.

### 2.1 The agent's own taste: $a$

Suppose the agent, asked with **no instructions at all**, likes chocolate a bit and large a lot.
We write that as a weight vector

$$a = (\,2,\; 3\,) \qquad \text{“chocolate is worth 2, large is worth 3”}$$

The score of an ice cream is just weights $\times$ features, $\;\delta = a^\top z$:

| | $z$ | score $a^\top z$ | |
|---|---|---|---|
| 🍫L | $(+1,+1)$ | $2 + 3 = \mathbf{5}$ | favourite |
| 🍦L | $(-1,+1)$ | $-2 + 3 = \mathbf{1}$ | |
| 🍫S | $(+1,-1)$ | $2 - 3 = \mathbf{-1}$ | |
| 🍦S | $(-1,-1)$ | $-2 - 3 = \mathbf{-5}$ | worst |

**This vector $a$ is "what the LLM wants".** It is the whole Phase-I question, in two numbers.

### 2.2 The contract: $x$

Now the user writes an instruction: **"I want a small one."**

The contract is also a little vector, $x$. It has **one slot for every single thing a contract
could ask for** — that is one slot per (feature, *level*), not one per feature. In our shop there
are four such things: chocolate, vanilla, large, small.

$$x \;=\; \bigl(\underbrace{x_1}_{\text{“chocolate”?}},\; \underbrace{x_2}_{\text{“vanilla”?}},\; \underbrace{x_3}_{\text{“large”?}},\; \underbrace{x_4}_{\text{“small”?}}\bigr), \qquad 1 = \text{yes},\quad 0 = \text{no}$$

Careful: the labels are the **questions**, the numbers are the **answers**. A slot is $0$
whenever the contract stays silent about it, so:

| the user writes | choc? | vanilla? | large? | small? | $x$ |
|---|---|---|---|---|---|
| *(nothing)* | – | – | – | – | $(0,0,0,0)$ |
| "I want a **small** one" | – | – | – | **yes** | $(0,0,0,1)$ |
| "I want a **vanilla** one" | – | **yes** | – | – | $(0,1,0,0)$ |
| "I want a **small vanilla** one" | – | **yes** | – | **yes** | $(0,1,0,1)$ |
| "I want **chocolate**" | **yes** | – | – | – | $(1,0,0,0)$ |
| "I want a **large chocolate**" | **yes** | – | **yes** | – | $(1,0,1,0)$ |

Two slots of the same feature can never both be 1 — nobody asks for a chocolate vanilla.

Our instruction is "I want a small one", so from here on $\;x = (0,0,0,1)$: only the *small* slot
is on. **No contract at all is $x = 0$**, all zeros. That is deliberate, and §4.1 explains why.

> This is exactly the real encoding. The laptops have $5 + 3 + 3 = 11$ slots — one per brand,
> screen size and ram size a contract could name.

### 2.3 The interaction matrix: $B$

Here is the important part. The contract does not choose an ice cream. **It changes the
weights.** How much it changes them is the matrix $B$: **one row per contract slot, one column
per weight.** Four slots, two weights, so $B$ is $4 \times 2$.

For this example, suppose $B$ is

$$B \;=\; \begin{pmatrix} +4 & +1 \\ -4 & -1 \\ +3 & +5 \\ -3 & -5 \end{pmatrix} \begin{matrix} \leftarrow\ \text{“chocolate”} \\ \leftarrow\ \text{“vanilla”} \\ \leftarrow\ \text{“large”} \\ \leftarrow\ \text{“small”} \end{matrix}$$

**Column 1 is the flavour weight, column 2 is the size weight.** Same thing as a table:

| $B$ | flavour wt | size wt |
|---|---|---|
| "chocolate" | $+4$ | $+1$ |
| "vanilla" | $-4$ | $-1$ |
| "large" | $+3$ | $+5$ |
| "small" | $-3$ | $-5$ |

Read a row as *"if the contract says this, add these numbers to the weights."* Row "small" is
$(-3,-5)$: it drags the size weight down by 5 (**compliance** — we asked for small) and the
flavour weight down by 3 (**leakage** — we said nothing about flavour).

### $B$ splits into blocks, and the blocks are the research question

$B$ is **not square** — $4\times 2$ here, $11\times 8$ for the laptops — so it has no diagonal
entries. What it has is **blocks**. Group the *rows* by which feature the contract names, and the
*columns* by which feature the weight belongs to:

| | affects **flavour** weight | affects **size** weight |
|---|---|---|
| contract names a **flavour** (choc, vanilla) | $\begin{matrix}+4\\-4\end{matrix}$ &nbsp; **compliance** | $\begin{matrix}+1\\-1\end{matrix}$ &nbsp; *leakage* |
| contract names a **size** (large, small) | $\begin{matrix}+3\\-3\end{matrix}$ &nbsp; *leakage* | $\begin{matrix}+5\\-5\end{matrix}$ &nbsp; **compliance** |

- **Diagonal blocks** ($f \to f$): the contract names a feature and moves *that* feature's
  weight. **Compliance.**
- **Off-diagonal blocks** ($f \to g$): the contract names one feature and moves a *different*
  one. **Leakage.**

Inside a diagonal block the pattern is "push up whatever was asked for": asking chocolate gives
$+4$, asking vanilla gives $-4$. One number sets the scale — call it $\rho_f$. Inside an
off-diagonal block, similarly, one number $\kappa_{f\to g}$ sets how much feature $f$'s contract
disturbs feature $g$.

That is where the small parameter count in §7.1 comes from: **one scalar per block**, so
$F$ compliance scalars and $F(F-1)$ leakage scalars — $F^2$ in total, no matter how many
*levels* each feature has.

The new weights are the old ones plus the rows the contract switched on:

$$\boxed{\;w \;=\; a \;+\; B^\top x\;}$$

### The multiplication, written out

$B$ is $4\times 2$ and $x$ is $4\times 1$, so we need $B^\top$ ($2 \times 4$) to get a
$2\times 1$ answer — the same shape as $a$. Transposing just turns the rows into columns:

$$B^\top \;=\; \begin{pmatrix} +4 & -4 & +3 & -3 \\ +1 & -1 & +5 & -5 \end{pmatrix}$$

The four columns are now, in order: **choc, vanilla, large, small**. Row 1 is the flavour weight,
row 2 is the size weight.

Now **"I want a small one"**, $x = (0,0,0,1)$:

$$B^\top x \;=\; \begin{pmatrix} +4 & -4 & +3 & -3 \\ +1 & -1 & +5 & -5 \end{pmatrix}\!\! \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix} \;=\; \begin{pmatrix} 4(0) + (-4)(0) + 3(0) + (-3)(1) \\ 1(0) + (-1)(0) + 5(0) + (-5)(1) \end{pmatrix} \;=\; \begin{pmatrix} -3 \\ -5 \end{pmatrix}$$

$$w \;=\; a + B^\top x \;=\; \begin{pmatrix} 2 \\ 3 \end{pmatrix} + \begin{pmatrix} -3 \\ -5 \end{pmatrix} \;=\; \begin{pmatrix} -1 \\ -2 \end{pmatrix}$$

**All the multiplication does is pick out the switched-on rows and add them up.** The zeros
delete the three rows we did not ask for; the single 1 keeps the "small" row. Nothing subtler is
happening — the matrix notation is just bookkeeping for "which rows apply".

That is clearest when *two* slots are on. For **"I want a small vanilla one"**, $x=(0,1,0,1)$:

$$B^\top x \;=\; \underbrace{\begin{pmatrix} -4 \\ -1 \end{pmatrix}}_{\text{“vanilla” row}} + \underbrace{\begin{pmatrix} -3 \\ -5 \end{pmatrix}}_{\text{“small” row}} \;=\; \begin{pmatrix} -7 \\ -6 \end{pmatrix}, \qquad w \;=\; \begin{pmatrix} -5 \\ -3 \end{pmatrix}$$

Every contract in the shop, from the one matrix:

| contract | $x$ | $B^\top x$ | $w = a + B^\top x$ |
|---|---|---|---|
| *(nothing)* | $(0,0,0,0)$ | $(0,\,0)$ | $(+2,\,+3)$ |
| "small" | $(0,0,0,1)$ | $(-3,-5)$ | $(-1,-2)$ |
| "vanilla" | $(0,1,0,0)$ | $(-4,-1)$ | $(-2,+2)$ |
| "small vanilla" | $(0,1,0,1)$ | $(-7,-6)$ | $(-5,-3)$ |
| "chocolate" | $(1,0,0,0)$ | $(+4,+1)$ | $(+6,+4)$ |
| "large chocolate" | $(1,0,1,0)$ | $(+7,+6)$ | $(+9,+9)$ |

> **Health warning.** Only the "vanilla" and "small" rows above correspond to experiments we have
> actually run. The "chocolate" and "large" rows are a **guess** — I filled them in as the exact
> mirror of the other two so the arithmetic works. §2.5 is about whether that guess is safe.

From here on we follow **"I want a small one"**, so $w = (-1,-2)$.

Look at what happened to each weight:

| weight | before | after | why |
|---|---|---|---|
| **size** | $+3$ (likes large) | $-2$ (likes small) | **compliance** — it did what it was told |
| **flavour** | $+2$ (likes chocolate) | $-1$ (likes vanilla) | **leakage** — nobody mentioned flavour |

**That is the entire research question, in one matrix.**

- The **diagonal blocks** of $B$ (contract about size $\to$ the size weight) are **compliance**.
- The **off-diagonal blocks** (contract about size $\to$ the *flavour* weight) are **leakage**.

### 2.4 Why leakage is the interesting block

Re-rank the shop with the new weights $w = (-1,-2)$, score $= w^\top z$:

| | score before | score after | |
|---|---|---|---|
| 🍦S | $-5$ | $\mathbf{+3}$ | now the favourite |
| 🍫S | $-1$ | $+1$ | |
| 🍦L | $+1$ | $-1$ | |
| 🍫L | $+5$ | $-3$ | was the favourite |

Now the key comparison. **Suppose there were no leakage** — the contract only touched the size
weight, so $B$'s second row were $(0, -5)$ instead of $(-3,-5)$. Then $w = (2,-2)$ and:

| | with leakage | **without** leakage |
|---|---|---|
| 🍫S | $+1$ | $\mathbf{+4}$ ← would win |
| 🍦S | $\mathbf{+3}$ ← wins | $0$ |

**Both agents obey the contract — both pick a small one.** But the leaky agent hands you
*vanilla*, and it was never told anything about flavour. It filled the silent part of the
contract with a preference of its own, and that preference is not even the one it started with.

That is the thesis conjecture, and it is testable as: **are the off-diagonal blocks of $B$ zero
or not?**

---

### 2.5 What if the contract asks for something the agent already wants?

Both contracts so far **fought** the agent's taste: it wanted large, we asked for small. But the
user could just as easily write **"I want chocolate"** — and the agent already likes chocolate.

That is $x = (1,0,0,0)$, which switches on the **"chocolate"** row of $B$. And here is the
problem: **we have never run that row.** In §2.3 I wrote it as $(+4,+1)$ so the arithmetic would
work — but that was a *guess*, not a measurement. It is one of two very different guesses:

**Hypothesis 1 — mirror** (what §2.3 assumed). Asking for chocolate is the exact opposite of
asking for vanilla, so $\;\text{row}_{\text{choc}} = -\,\text{row}_{\text{van}} = (+4,+1)$:

$$w \;=\; (2,3) + (4,1) \;=\; (\,6,\; 4\,)$$

**Hypothesis 2 — ceiling.** The agent *already* wants chocolate, so there is little left to
change and the row is small, say $(+1, 0)$:

$$w \;=\; (2,3) + (1,0) \;=\; (\,3,\; 3\,)$$

Both keep 🍫L on top — an agreeing contract does not visibly change the ranking, it only
sharpens it. But the **weights** differ by a factor of two, and so does the leakage: mirror says
asking for chocolate pushes the *size* weight up by $+1$, ceiling says it does nothing at all.

**"I want a large chocolate"** ($x = (1,0,1,0)$) stacks both rows, then shrinks by $\lambda$ as
in §4.2:

| | flavour wt | size wt | top |
|---|---|---|---|
| no contract | $+2$ | $+3$ | 🍫L |
| mirror, additive | $+9$ | $+9$ | 🍫L |
| mirror, $\times\lambda = 0.7$ | $+6.9$ | $+7.2$ | 🍫L |
| ceiling, additive | $+3$ | $+4$ | 🍫L |

> **This is the single most important open question in the whole document.** Mirror says the four
> rows of $B$ are really only two — so a level we never ran is predictable, and the 11 laptop
> rows collapse to a handful of scalars (§7.1). Ceiling says every level needs its own row, and
> then contract space is not compressible at all.
>
> **We cannot tell yet**, because every constrained feature has been run at exactly **one**
> level: `screen=14-inch` and `ram=8GB`. In laptop terms the missing case is `ram=16GB` — a
> contract that *agrees* with what the models already want (more ram). **This is what Phase A in
> §8 is for**, and why it is the first thing to run.

## 3. From choices to what we actually measure

We never see a score. We show the model **two options** and read a margin,
$y = \texttt{score\_a} - \texttt{score\_b}$.

That is fine, because a difference of scores is a score of the *difference*. Compare 🍫L against
🍦S under no contract:

$$\Delta z \;=\; z_{🍫L} - z_{🍦S} \;=\; (+1,+1) - (-1,-1) \;=\; (\,2,\; 2\,)$$

$$y \;=\; a^\top \Delta z \;=\; 2 \cdot 2 + 3 \cdot 2 \;=\; +10 \quad \text{(option 1 wins easily)}$$

and under the "small please" contract, with $w = (-1,-2)$:

$$y \;=\; w^\top \Delta z \;=\; -1 \cdot 2 + (-2) \cdot 2 \;=\; -6 \quad \text{(option 2 now wins)}$$

So everything runs on two objects: **$\Delta z$**, how the two options differ, and **$w$**, the
weights for the contract in force. Add one nuisance term $\gamma$ for the fact that option 2 gets
picked more just for being second, and the model is complete:

$$\boxed{\;y \;=\; \gamma_i \;+\; \bigl(\underbrace{a + B^\top x_i}_{w_i}\bigr)^{\!\top} \Delta z \;+\; \text{noise}\;}$$

### 3.1 Why this is just one regression

$(a + B^\top x_i)^\top \Delta z$ multiplies out into a main part plus one part per contract slot:

$$y \;=\; \gamma_i \;+\; \underbrace{a^\top \Delta z}_{\text{2 columns}} \;+\; \sum_{k=1}^{4} \underbrace{x_{ik}\,(B_{k\cdot})^\top \Delta z}_{\text{2 columns each}}$$

So one row of the design matrix is

$$\bigl[\;\text{condition dummies}\;\bigm|\;\Delta z\;\bigm|\;x_{i1}\Delta z\;\bigm|\;x_{i2}\Delta z\;\bigm|\;x_{i3}\Delta z\;\bigm|\;x_{i4}\Delta z\;\bigr]$$

**In one line: each interaction block is a copy of $\Delta z$ that is switched on only when that
contract slot is switched on.** So the coefficient fitted on that copy *is* that row of $B$.
Nothing more is happening.

For our 🍫L vs 🍦S row under "small please" ($x = (0,0,0,1)$), the design row is

$$\bigl[\;\dots\;\bigm|\;\underbrace{(2,2)}_{\Delta z}\;\bigm|\;\underbrace{(0,0)}_{\text{choc off}}\;\bigm|\;\underbrace{(0,0)}_{\text{van off}}\;\bigm|\;\underbrace{(0,0)}_{\text{large off}}\;\bigm|\;\underbrace{(2,2)}_{\textbf{small on}}\;\bigr]$$

Notice the three blocks of zeros. **A slot that is never switched on in any run contributes
nothing anywhere, so its row of $B$ can never be estimated** — that is §2.5's problem, and §9's,
in one picture.

**This is why we do not need the MC-EM machinery in `grum4llm/`.** The GRUM paper needs Gibbs
sampling because it only sees *rankings*, so the utilities are hidden. We read a real number off
the model for every pair. Nothing is hidden, so it is ordinary least squares.

> Verified on the real data: the pooled fit reproduces the separate per-condition Bradley-Terry
> weights to $10^{-14}$. It is a **reparameterization** of what we already do, not a new
> estimator — so nothing already established is put at risk.

---

## 4. Two choices that make the model work

### 4.1 "No constraint" is a zero, not a category

A slot of $x$ is $0$ when the contract is silent about it. So "no contract at all" is $x = 0$,
and then

$$x_i = 0 \quad\Longrightarrow\quad w_i = a .$$

**This is why $a$ means exactly "the preference with no contract".** If we instead added a column
for "unconstrained", it would be confounded with $a$ and neither would mean anything clean.

### 4.2 Two contracts at once: the model must *predict*, not fit

Now the user says **"I want a small vanilla one"** — two slots on, $x = (0,1,0,1)$. The model has
an opinion without being told:

$$w \;=\; a + B^\top x \;=\; \underbrace{(2,3)}_{a} + \underbrace{(-4,-1)}_{\text{“vanilla”}} + \underbrace{(-3,-5)}_{\text{“small”}} \;=\; (\,-5,\; -3\,)$$

This is the whole point of the model. Writing $S$ for the **shift** a contract causes,
$S = w - a$, additivity says

$$S_{\text{both}} \;=\; S_{\text{vanilla}} + S_{\text{small}}$$

**and that is a prediction we can check against a contract we never ran.**

An earlier draft of this document instead gave the compound contract its own free parameters.
That was a mistake: it fits perfectly and predicts nothing. §7 measures how well the real
prediction actually does, and it does well on **direction** but overshoots on **size** — so the
correct fix is one shared shrinking factor $\lambda$:

$$S_{\text{both}} \;=\; \lambda\,\bigl(S_{\text{vanilla}} + S_{\text{small}}\bigr), \qquad \lambda \approx 0.7$$

In the toy, that turns the predicted shift $(-7,-6)$ into $(-4.9,-4.2)$, so $w = (-2.9,-1.2)$.
**Two contracts together do less than the sum of what each does alone.**

---

# Part II — The laptops

## 5. The mapping, precisely

| GRUM | ice cream | ours |
|---|---|---|
| alternative $j$ | one of 4 ice creams | one of 45 laptops |
| agent $i$ | — | **the contract in the prompt** |
| $z_j$ | (flavour, size) | (brand, screen, ram) |
| $x_i$ | 4 slots: choc? vanilla? large? small? | 11 slots: 5 brands + 3 screens + 3 rams |
| $B$ rows we have run | 2 of 4 ("vanilla", "small") | **2 of 11** (`screen=14`, `ram=8GB`) |
| $a$ / $\delta_j$ | $(2,3)$ | the model's own preference, no contract |
| $B$ | $2\times 2$ | $K \times 8$ |

The real $z$ has **8** columns, not 2, because the features have more than two levels: brand has
5 levels $\to$ 4 columns, screen 3 $\to$ 2, ram 3 $\to$ 2. (Same idea as $\pm 1$; the last level
of each block is minus the sum of the others, so every level still gets a number.) In order:

$$\bigl(\underbrace{\text{ASUS},\ \text{Apple},\ \text{Dell},\ \text{HP}}_{\text{brand}},\; \underbrace{\text{13-inch},\ \text{14-inch}}_{\text{screen}},\; \underbrace{\text{4GB},\ \text{8GB}}_{\text{ram}}\bigr)$$

The full model is what it was in §3, with $\varepsilon$ written in:

$$U_{ij} \;=\; \delta_j \;+\; x_i^\top B\, z_j \;+\; \varepsilon_{ij}$$

> **What $\varepsilon$ is here.** In the GRUM paper it is *taste noise* — the agent's own
> randomness. Ours is not that: the scores are deterministic, and rerunning moves a fitted weight
> by $0.003$. Our residual is **template variation plus misspecification**. Worth saying in the
> thesis, because it changes what a standard error means. SEs stay **clustered by unordered item
> pair** (inflation $2.6$–$3.1\times$).

## 6. The same walk-through, on real data (qwen-32B)

Contract `ram=8GB`. Option 1 = **Apple, 14-inch, 16GB**, option 2 = **Dell, 14-inch, 8GB**.

**The contrast.** Apple $\to (0,1,0,0)$, 14-inch $\to (0,1)$, 16GB $\to (-1,-1)$; Dell
$\to (0,0,1,0)$, 14-inch $\to (0,1)$, 8GB $\to (0,1)$. So

$$\Delta z = (\,0,\, 1,\, -1,\, 0\;\mid\; 0,\, 0\;\mid\; -1,\, -2\,)$$

*"option 1 is Apple instead of Dell, same screen, and more ram."*

**The prediction.**

| term | value | meaning |
|---|---|---|
| $\gamma_i$ | $-8.32$ | positional bias |
| $a^\top \Delta z$ | $+18.75$ | **no contract**: the Apple/16GB wins easily |
| $(B_{\text{ram=8GB}})^\top \Delta z$ | $-24.21$ | **what the contract does** |
| **total** | $\mathbf{-13.78}$ | option 2 (Dell, 8GB) now wins |
| observed mean $y$ | $-14.77$ | over the 5 templates |

**Splitting that $-24.21$ is the point** — exactly the 🍫 / 🍦 split from §2.3:

$$\underbrace{1.59(-1) + 9.48(-2)}_{\textbf{ram} \;=\; -20.56 \;\;\text{compliance}} \;+\; \underbrace{(-2.95)(1) + 0.70(-1)}_{\textbf{brand} \;=\; -3.65 \;\;\textbf{leakage}} \;+\; \underbrace{0}_{\text{screen}} \;=\; -24.21$$

(coefficients shown to 2 dp; block totals exact)

The user asked about **ram**, and the model quietly moved **brand** by $-3.65$. Apple down
$2.95$, Dell up $0.70$. That is the Apple collapse as a single coefficient.

## 7. Does it generalize?

The data supports exactly one held-out test: fit on $\{$`none`, `screen=14`, `ram=8`$\}$, predict
`screen=14 + ram=8` — a contract the model never saw. As in §4.2, the prediction is forced:
$\hat S_{\text{both}} = S_{14} + S_{8}$, with no free parameters.

With $R^2_{\text{gen}} = 1 - \lVert S - \hat S\rVert^2/\lVert S\rVert^2$:

| | $R^2_{\text{gen}}$ | slope |
|---|---|---|
| all features, all pairs | $0.745$ | $0.762$ |
| brand block, spec-tied | $0.773$ | $0.704$ |

- **Direction generalizes**: $\approx 0.75$ with nothing fitted on the held-out contract.
- **Size overshoots**: slope below 1 in 7 of 8 models. Sub-additive, as flagged in §4.2.

**One shared scalar fixes it.** With $q_i$ = how many features the contract names,
$w_i = a + \lambda^{\,q_i-1} B^\top x_i$, fitting a single $\lambda$ across *all* models:

| | $\lambda$ | additive | with $\lambda$ | ceiling |
|---|---|---|---|---|
| all features | $0.676$ | $0.745$ | $0.893$ | $0.924$ |
| brand, spec-tied | $0.710$ | $0.773$ | $\mathbf{0.971}$ | $0.982$ |

**One parameter, $0.77 \to 0.97$ — essentially the ceiling.** A free conjunction vector per
double would cost 312 parameters and transfer to nothing; $\lambda$ costs one and transfers to
all 39 doubles.

> **Caveat.** We only observe $q \in \{1,2\}$, so the *shape* of the saturation is not identified:
> $\lambda^{q-1}$ and $q^{-\alpha}$ fit identically here and split at $q=3$ ($0.50$ vs $0.58$).
> **One triple contract settles it.**

### 7.1 Why it generalizes: leakage has no shape of its own

A free $B$ needs one row per (feature, level), so a level never run has no row and cannot be
predicted at all. But the fitted rows are nearly **parallel** — and parallel to $-\delta$:

| cosine | mean over 7 models |
|---|---|
| $B_{\text{screen=14}}$ vs $B_{\text{ram=8GB}}$ (brand block) | $0.92$ (six of seven $\geq 0.97$) |
| $B_{\text{screen=14}}$ vs $-\delta$ | $0.86$ |
| $B_{\text{ram=8GB}}$ vs $-\delta$ | $0.94$ |

**Every contract flattens the brand preference along its own direction**; contracts differ only
in *how much*. (gemma-1B is the exception at $0.54$, and also the weakest signal.)

That licenses a much smaller model. For a feature $g$ the contract does **not** name:

$$w_g(x_i) \;=\; \bigl(1 - \kappa_g(x_i)\bigr)\, a_g, \qquad \kappa_g(x_i) \;=\; \lambda^{\,q_i-1}\!\!\sum_{f \text{ named}} \kappa_{f \to g}$$

and for a feature $f$ **named** at level $\ell$: $\;w_f(x_i) = a_f + \rho_f e_\ell$.

- $\rho_f$ — **compliance**, one scalar per feature.
- $\kappa_{f\to g}$ — **leakage**, one scalar per ordered feature pair.
  Measured: $\kappa_{\text{ram}\to\text{brand}} = 0.94$, $\kappa_{\text{screen}\to\text{brand}} = 0.54$.
- $\lambda$ — saturation.

Check: $0.71 \times (0.94 + 0.54) = 1.05$, so $\kappa \approx 1$ and the brand preference is
**flattened to about zero** under the double contract — matching the measured brand spread
falling $10.4 \to 4.2$.

> **Load-bearing and untested:** that $\kappa_{f\to g}$ depends on the **feature**, not the
> **level**. Every constrained feature has so far been run at exactly one level. If it holds,
> `ram=4GB` tells you what `ram=16GB` does and 11 rows of $B$ collapse to 6 scalars.

## 8. Sampling: the space is exponential, the parameters need not be

A contract names each feature or stays silent, so the number of possible contracts is

$$\prod_{f}(L_f + 1) \;=\; 6 \times 4 \times 4 \;=\; \mathbf{96}$$

= 1 `none` + 11 singles + 39 doubles + 45 triples. **We have run 4 — about 4%.** Adding one
4-level feature (price) takes it to 480. Exhaustive collection is never an option.

| model | parameters for all 96 | |
|---|---|---|
| saturated, one $w$ per contract | $1056$ | hopeless |
| GRUM, free $B$, main effects | $88$ | fine |
| $\;$ + free conjunction per double | $+312$ | **does not generalize** |
| **structured $B$ + $\lambda$ (§7.1)** | $\mathbf{18}$ | $a(8)+\rho(3)+\kappa(6)+\lambda(1)$ |

The space grows like $\prod_f (L_f{+}1)$; the structured model grows like $F^2$ — in the number
of **features**, not levels. Add a 4th feature: space $96 \to 480$, parameters $18 \to 28$.

> **The sampling rule:** run the **singles exhaustively** — there are only $\sum_f L_f = 11$ and
> they carry all of $B$. **Sample the combinations** — there are 84, and predicting them is the
> model's job.

Priority order:

| phase | contracts | conditions | runs (×8) | buys |
|---|---|---|---|---|
| **A** | `screen=13`, `screen=16`, `ram=4`, `ram=16` | 4 | 32 | **tests whether $\kappa$ is level-independent** |
| **B** | `brand=Apple` + 2–3 others | 3–4 | 24–32 | does a *brand* contract leak into specs? |
| **C** | 4–6 doubles over all 3 feature-pairs, + 2 triples | 6–8 | 48–64 | fits $\lambda$, identifies its shape at $q{=}3$ |
| **D** | ~8 contracts sampled at random, never fitted | 8 | 64 | honest held-out number |

**Phase A first** — cheapest, and if $\kappa$ turns out level-dependent the 18-parameter model
collapses and everything after it changes.

Pick within a phase by D-optimality on the **contract-side matrix alone**: every condition runs
the same balanced factorial of items, so the information matrix factorizes and this is a 96-row
calculation, not a Fisher-information computation over the whole model.

> **One trap, found the hard way.** Ridged D-optimality *looks* like it prefers doubles (log det
> $4.62$ vs $3.21$), but the 4 best doubles reach rank **6 of 7**, because
> $(\text{13in}+\text{4GB}) + (\text{16in}+\text{16GB}) = (\text{13in}+\text{16GB}) + (\text{16in}+\text{4GB})$
> once the conjunction indicator is on, and the ridge hides it. **Check the rank, not the score.**

---

# Part III — The careful parts

## 9. Identification

$z$-space is in excellent shape: 45 items, full factorial, brand $\perp$ ram exactly.
$x$-space is the bottleneck, badly:

| | |
|---|---|
| conditions collected | `none`, `screen=14-inch`, `ram=8GB`, `screen=14-inch+ram=8GB` |
| $x$-space rank | **3 of 7** (reduced 3-slot encoding) |
| zero information on | `screen=13-inch`, `screen=16-inch`, `ram=4GB`, `ram=16GB`, **all 5 brands** |

**A free $B$ is unidentified in most of its rows.** More items do not help. More templates do not
help. Only new contracts help.

Gauge: each $z$ block is sum-to-zero, so every level gets a coefficient and a standard error
(including the one `drop_first=True` hides); $x$ needs no gauge, because "unconstrained" is the
zero vector; $\delta$ is pinned by the $x=0$ run, so **the unconstrained run is load-bearing** —
keep running it for every model.

## 10. The lexicographic threat

The models are **lexicographic**: more RAM wins 6750/6750. A GRUM is an additive utility, so it
inherits the criticism we just made of the additive BT fit. Two things go wrong if ignored:

1. **The likelihood diverges.** With hard win/loss outcomes and a perfectly separating feature,
   the MLE pushes the RAM coefficient to infinity — Ford's condition / Theorem 2 in the GRUM
   paper failing. It is why we see RAM weights of $\sim 25$.
2. **The brand block gets estimated where brand does nothing.** Under a priority rule brand only
   matters when ram and screen are tied; pooling all 9900 pairs averages a live regime with a
   dead one.

**The fix:** fit the brand block on **spec-tied** pairs (same screen, same ram — 900 rows per
condition). Not a hack: under a lexicographic rule a lower-priority utility is *identified only
inside ties on the higher-priority ones*. It is also why the spec-tied numbers generalize better
($0.97$ vs $0.89$).

Keep the continuous margin as the response, not hard wins, so separation cannot blow up the fit.
**Do not let the GRUM carry the priority-order claim** — that stays model-free (win rates,
override rates).

## 11. Results, and what is still open

Spec-tied stratum, SEs clustered by unordered pair. $\delta$ is Apple's weight with no contract;
the other columns are Apple's entries in the corresponding rows of $B$.

| model | $\delta_{\text{Apple}}$ | $B_{\text{screen=14-inch}}$ | $B_{\text{ram=8GB}}$ |
|---|---|---|---|
| qwen-0.5B | $-0.08$ | $+0.05$ | $-0.04$ |
| qwen-7B | $3.24$ | $-3.68$ | $-3.91$ |
| qwen-32B | $7.31$ | $-3.59$ | $-6.07$ |
| qwen-72B | $5.09$ | $-2.75$ | $-6.42$ |
| gemma-1B | $3.64$ | $-1.27$ | $-3.25$ |
| gemma-4B | $6.02$ | $-3.37$ | $-6.18$ |
| gemma-12B | $3.41$ | $-3.59$ | $-7.36$ |
| gemma-27B | $10.79$ | $-7.03$ | $-11.25$ |

$\delta$ reproduces the "same ram + same screen" column of the 2026-08-10 entry exactly
(qwen-32B $7.31$, gemma-27B $10.79$) — same measurement, new parameterization. **The leakage is
negative in 8/8 models for both contracts.** Summarized gauge-free as the slope of the leakage
row on $\delta$: **ram $-0.94$, screen $-0.54$**.

> **Say "flattens", not "cancels".** A slope of $-1$ comes equally from "shrink toward zero" and
> "invert", so it cannot separate them. Measured directly, brand spread falls $10.4 \to 3.7$
> under `ram=8GB` while correlation with the no-contract brand vector drops to $+0.19$ — so the
> dominant effect is **flattening**, with re-ordering only on top of it. The structured form
> already encodes this correctly: $\kappa \to 1$ sends $w_g$ to **zero**, not to $-a_g$.

**Open:**

- [ ] **Is $\kappa$ level-independent?** The 18-parameter model rests on it. Phase A.
- [ ] **Shape of the saturation** — $\lambda^{q-1}$ vs $q^{-\alpha}$. Phase C.
- [ ] $\kappa$ on the **log-odds scale** and on the **ceteris-paribus win-rate scale** disagree
      for the screen contract ($0.54$ "halves it" vs $-4$ points "surgical"). Decide which the
      thesis reports, or report both and explain the difference.
- [ ] The structured form assumes leakage is pure shrinkage along $-\delta$. The correction entry
      shows re-ordering *on top of* the shrinkage under the ram contract; if that matters,
      $\kappa$ needs a second component orthogonal to $\delta$.
- [ ] Put **model covariates** (family, $\log$ size) into $x$, so one GRUM covers all 8 models and
      the size-scaling of compliance becomes an interaction coefficient.
- [ ] The margin is censored at the logprob floor in 88–99% of rows; a tobit is the honest
      response model. Winsorizing showed directions are stable, so this is a robustness check.
- [ ] Template is pooled; make it a random effect and see if the leakage numbers move.
- [ ] gemma-1B breaks the parallel-leakage pattern ($0.54$ vs $\geq 0.97$). Artifact, or a real
      difference in how weak models handle contracts?
- [ ] GRUM gives us a better-shaped $v$. It still does **not** give $\Delta W$ — that needs a user
      utility $u$, which the laptops set does not have.
