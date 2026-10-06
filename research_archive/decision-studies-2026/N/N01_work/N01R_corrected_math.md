# N01-R — corrected mathematics

This file supersedes the corresponding sections of the first `N01_report.md`. It states (1) the
correct set algebra including the background, (2) the local-dependency argument that replaces the
earlier modular-sum claim, and (3) what the corrected argument does and does not establish.

---

## 1. The counterexample stands

The hand graph is a valid witness that a two-layer local proxy cannot express a certain
long-distance conditional preference:

* the four two-hop regions are pairwise disjoint —
  `R2(a)={a1,a2}`, `R2(b)={b1,b2}`, `R2(c)={c1,c2}`, `R2(d)={d1,d2}`
  (under either reading, with or without the source node);
* after three hops the four paths meet, at `P` and at `Q`;
* the true preferences are `δ_c = −6` and `δ_d = +6`.

**This already refutes the claim that two-hop disjointness forbids a true reversal.** The regions that
must stay apart for the proxy argument are the two-hop regions, while the couplings that create the
reversal happen at three hops. The original report's §4 misread its own specimen: it treated a
condition on `R2` as if it constrained `R`.

---

## 2. Correct set algebra, with the background

Let one live-edge world define the reachable sets `A, B, C, D` of the four candidates and
`W = R(U)` of the background seeds. The two measured differences are

```
X_c = |A \ (W ∪ C)| − |B \ (W ∪ C)|
X_d = |A \ (W ∪ D)| − |B \ (W ∪ D)|
```

The earlier report omitted `W`. Writing `α = A\W`, `β = B\W`, `γ = C\W`, `δ = D\W`, subtraction gives

```
X_c − X_d = |α ∩ δ| + |β ∩ γ| − |α ∩ γ| − |β ∩ δ|
```

Note what this substitution does and does not do. It makes the *expression* free of `W`, but the
background has **not disappeared** — it has been absorbed into the definitions of `α, β, γ, δ`, each
of which is `W`-dependent. Two different backgrounds therefore give different `α, β, γ, δ`, and the
overlap terms above are background-dependent. `W` is only invisible when the four sets are held fixed
for one world; it is not absent from the comparison.

### 2.1 Why two-hop disjointness does not force `X_c = X_d`

Two-hop disjointness does force `R2(a) ∩ R2(d) = ∅` and `R2(b) ∩ R2(c) = ∅` — but only in the
**deterministic** two-hop sense. In a live-edge world the reachability sets extend past two hops, so
`α ∩ δ` and `β ∩ γ` can be non-empty through longer paths. The saved data demonstrates this:

| where | value |
|---|---|
| main case 0, world 1000295 | `X_c = −20`, `X_d = −19` |
| main case 38, world 1001137 | `X_c = −36`, `X_d = −16` (gap 20) |
| frozen confirm batch | 68 of 81,920 (world, comparison) pairs have `X_c ≠ X_d` |

Those 68 are **measured counterexamples to the identity**, not evidence for it. The correct statement
is: the identity fails rarely under this sampling, which limits how easily a reversal can be found —
it does not show that a reversal is excluded.

---

## 3. The proxy argument, done properly: local dependency

The earlier version wrote a two-layer local proxy as the modular function
`Σ̂(S) = Σ_{v∈S} φ(v) + const` and then checked that identity on five random score vectors. **Both
steps were wrong.** A general two-layer GNN is *not* a modular function of the seed set: within its
receptive field it can represent seed–seed interactions, and "sum the per-node outputs" is not the same
as "sum per-node scores". The five-vector computation is an arithmetic illustration of a modular
function, **not** a structural verification of a GNN, and it must not be cited as one.

The correct argument is a local-dependency argument.

**Setup.** Fix the graph, the parameters, the number of layers `L`, and the background set `U`. Each
output node `v` is scored from its `L`-hop receptive field `N_L(v)`. The four configurations are
`S_ac`, `S_bc`, `S_ad`, `S_bd`, and they differ from one another only by which of `a, b` and which of
`c, d` is present.

**Claim.** Suppose no output node's `L`-hop receptive field simultaneously contains candidates from
the `{a,b}` side and the `{c,d}` side, i.e. for every `v`, the receptive field meets at most one of the
two candidate pairs. Then the four configurations induce mixed differences that vanish:

```
δ̂_c = σ̂(S_ac) − σ̂(S_bc) = δ̂_d = σ̂(S_ad) − σ̂(S_bd)
```

**Why.** For a fixed node `v`, let `φ_v(·)` be its score, a function of the seed indicator restricted
to `N_L(v)`. If `N_L(v)` meets only the `{a,b}` side, then replacing `c` by `d` inside `N_L(v)` changes
nothing, so `φ_v` takes the same value on `S_ac` and `S_ad`, and likewise on `S_bc` and `S_bd`. If
`N_L(v)` meets only the `{c,d}` side, then replacing `a` by `b` changes nothing, so `φ_v` agrees on
`S_ac` and `S_bc`, and on `S_ad` and `S_bd`. In both cases the contribution of `v` to
`σ̂(S_ac) − σ̂(S_bc) − σ̂(S_ad) + σ̂(S_bd)` is zero. Summing over `v` gives the claim. ∎

Under the hand graph's construction with `L = 2`, the hypothesis holds (the receptive fields are
disjoint), and the conclusion is exactly the obstruction: the proxy cannot separate the two
preferences that the real process separates by ±6.

### 3.1 What this does NOT establish

* It proves a relation among **four specified configurations**. It does **not** prove that the model is
  additive over seed sets in general, and it must not be stated that way.
* It says nothing about other backgrounds, other budgets, or seed sets outside these four.
* It is a statement about a **model class** ("any scorer whose per-node receptive field is confined to
  `L` hops and whose total is a sum of per-node scores"), derived at the class level.
* Because the identity is a property of the class, it also transfers to the official SIMBA
  `SpreadPredictor` **if** that implementation is of this class — but that was never instantiated here
  (no torch, no network), so the empirical leg remains open. See `N01_report.md` §2.1.

---

## 4. Consequence for the N01-S probe

Relaxing the structural condition to allow cross-side coupling **also removes the hypothesis of the
local-dependency argument** (candidates from both sides can now enter a single receptive field). So
after N01-S, the proxy identity `δ̂_c = δ̂_d` is **no longer guaranteed**.

Therefore N01-S cannot be used to conclude that "the original GNN cannot express the true ordering,"
even if it had found a reversal. This is the reason N01-S should not be extended as designed, and it
is why the current round stops at correcting the evidence rather than continuing with that design.
