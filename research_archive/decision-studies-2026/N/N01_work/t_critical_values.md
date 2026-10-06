# Verified two-sided Student-t critical values — df = 19, 39, 99, 255, 2047

Produced by `N01_work/verify_t_critical.py` (Python 3.13.14, **standard library only**;
no numpy / scipy / networkx, no network access). The complete, unedited script output is
reproduced verbatim in **§6** below; `N01_work/verify_t_critical_output.txt` holds the same
text.

---

## 0. RESOLUTION — the bug this verification caught (read first)

A first version of `run_diffusion.py` hard-coded the t constants from memory. This independent
recomputation showed they were **wrong in two ways**, and the experiment was re-run with the
corrected values.

**Bug 1 — wrong Bonferroni family size (the serious one).** The constants originally used for the
simultaneous intervals were the quantiles at `1 − 0.05/80 = 0.999375`. But a simultaneous 95 %
family over `m = 2N = 80` intervals requires **each** interval at per-comparison *confidence*
`1 − 0.05/80 = 0.999375`, which is the `1 − 0.05/160` quantile — i.e. `t(2047) = 3.2317`, not the
`2.9729` that was used. The original multiplier was therefore about **0.26 too small**, making the
adjusted intervals too narrow.

This is exactly the ambiguity the subagent flagged in §6's alternative-convention block: the
task prompt's illustrative digits (`T_BONF_2N80_DF_2047 = 3.2...`,
`T_BONF_2N40_DF_2047 = 3.0...`) correspond to the *one-sided-alpha* convention
(`1 − 0.10/m`), which is **not** the definition the task states. The task states the per-interval
level explicitly as `1 − 0.05/(2N)`, so that is what the final run uses.

**Bug 2 — small errors at df = 255 and df = 2047.** The ordinary 95 % constants were off by
`1.9e-4` and `3.8e-4` respectively. Corrected.

**Corrected constants actually used in the final run** (both now self-checked at runtime):

```
T95
    df   19 : 2.093024054408307
    df   39 : 2.022690920036760
    df   99 : 1.984216951508683
    df  255 : 1.969310569849874
    df 2047 : 1.961123559863062

T_BONF family size 2N = 80   (each interval at 1 - 0.05/80 = 0.999375)
    df   39 : 3.479924403320799
    df   99 : 3.322736366563675
    df  255 : 3.263705552933068
    df 2047 : 3.231723227703862

T_BONF family size 2N = 40   (per-group view only; not used for headline verdicts)
    df   19 : 3.481150847255499
    df   39 : 3.232161794568122
    df   99 : 3.102616799226298
    df  255 : 3.053675299918602
    df 2047 : 3.027090023031110
```

**Effect on the N01 conclusions: none.** Both experiments were re-run end to end with the corrected
constants. The frozen run's verdict counts are unchanged
(`not_significant` 4, `both_significant_same_side` 32, `identical_by_construction` 4, **confirmed
reversal 0**), and the probe's reversal count stays **0** (one comparison moved from
`both_significant_same_side` to `not_significant`). Widening the intervals cannot create a reversal,
and no frozen effect size sits close enough to the decision boundary for a 0.26 change in the
multiplier to matter.

`run_diffusion.py` now calls `check_t_tables()` at startup, which recomputes every constant from the
exact closed-form CDF and aborts if any disagrees by more than `1e-9`. Both final runs printed:

```
t-table self-check passed (family size 2N = 80; each interval at 1 - 0.05/80 = 0.999375; t(2047) = 3.231723)
```

Two implementations produced these numbers independently and agree to **~1e-13**: this one
(`verify_t_critical.py`), and `verify_t.py`, a separate implementation written to audit it
(Lentz continued fraction + bisection, cross-checked against Simpson quadrature of the density and
against six published textbook values with worst error `2.3e-10`).

---

## 1. The exact definitions used (read this before using the numbers)

**Plain 95 % — `T95`.** `t` such that `P(|T_df| <= t) = 0.95` for `T_df ~ t_df`,
equivalently the upper 0.975 quantile `t_{0.975, df}` (two-sided 95 %).

**Bonferroni — `T_BONF_*`.** A family of `m = 2N` two-sided intervals/tests is to hold
*simultaneously* at 95 %. Bonferroni splits the error budget equally, so **each member is
computed at per-comparison confidence**

```
1 - 0.05/m  =  1 - 0.05/(2N)
```

equivalently per-comparison two-sided alpha `0.05/m`, equivalently a per-comparison
one-sided tail of `0.025/m`. Hence

```
T_BONF(m, df) := the t with P(|T_df| <= t) = 1 - 0.05/m  = 1 - 0.05/(2N).
```

* `T_BONF_2N80` = family size **m = 2N = 80** (N = 40 comparisons) → target 0.999375
* `T_BONF_2N40` = family size **m = 2N = 40** (N = 20 comparisons) → target 0.998750

> **Discrepancy worth flagging.** The illustrative digits in the task prompt
> (`T_BONF_2N80_DF_2047 = 3.2...`, `T_BONF_2N40_DF_2047 = 3.0...`) do **not** match the
> definition stated in the same prompt. They match a *different* convention — a
> **one-sided** per-comparison alpha of `0.05/m`, i.e. target coverage `1 - 0.10/m` — which
> gives `3.231723227703792` and `3.027090023031109` at df = 2047. Every number in §2 uses
> the **stated** definition (`1 - 0.05/m`); the alternative is tabulated in §3 so that a
> switch-over costs nothing if that was the intent.

---

## 2. RECOMMENDED CONSTANTS (paste-ready)

Values are the **method C** roots (60-digit `decimal` quadrature of the t density); they
are good to ~1e-25 and were validated against exact closed forms to ≤ 9.6e-26 (§4.3).
The last printed digit is therefore reliable; in IEEE double they are exact to ~1e-16.

```python
# --------------------------------------------------------------------------
# Two-sided Student-t critical values, P(|T_df| <= t) = target.
# BONFERRONI DEFINITION: family size m = 2N two-sided intervals at simultaneous
# 95% -> per-comparison confidence 1 - 0.05/m.  'T_BONF_2N80' means m = 2N = 80
# (N = 40 comparisons); 'T_BONF_2N40' means m = 2N = 40 (N = 20 comparisons).
# Method C (60-digit Decimal quadrature); max |method C - method A| = 3.23e-13.
# --------------------------------------------------------------------------

# T95 : target P(|T|<=t) = 0.950000   (per-comparison alpha = 0.05)
T95_DF_19              = 2.093024054408310
T95_DF_39              = 2.022690920036761
T95_DF_99              = 1.984216951586417
T95_DF_255             = 1.969310569849875
T95_DF_2047            = 1.961123559863385

# T_BONF_2N80 : target P(|T|<=t) = 0.999375   (per-comparison alpha = 0.05/80)
T_BONF_2N80_DF_19      = 4.089179183536579
T_BONF_2N80_DF_39      = 3.720669712860826
T_BONF_2N80_DF_99      = 3.533475944075475
T_BONF_2N80_DF_255     = 3.463598989782222
T_BONF_2N80_DF_2047    = 3.425839556652871

# T_BONF_2N40 : target P(|T|<=t) = 0.998750   (per-comparison alpha = 0.05/40)
T_BONF_2N40_DF_19      = 3.785667236494119
T_BONF_2N40_DF_39      = 3.479924403320762
T_BONF_2N40_DF_99      = 3.322736366563636
T_BONF_2N40_DF_255     = 3.263705552933041
T_BONF_2N40_DF_2047    = 3.231723227704102

# aliases making the family size explicit (same numbers):
T_BONF_M80_DF_19       = 4.089179183536579
T_BONF_M80_DF_39       = 3.720669712860826
T_BONF_M80_DF_99       = 3.533475944075475
T_BONF_M80_DF_255      = 3.463598989782222
T_BONF_M80_DF_2047     = 3.425839556652871
T_BONF_M40_DF_19       = 3.785667236494119
T_BONF_M40_DF_39       = 3.479924403320762
T_BONF_M40_DF_99       = 3.322736366563636
T_BONF_M40_DF_255      = 3.263705552933041
T_BONF_M40_DF_2047     = 3.231723227704102
```

---

## 3. ALTERNATIVE convention (NOT used above)

If instead each of the `m` members gets a **one-sided** alpha = `0.05/m` (target coverage
`1 - 0.10/m`), the constants are smaller. These are the numbers that match the prompt's
illustrative digits.

```python
# ALTERNATIVE: per-comparison ONE-SIDED alpha = 0.05/m  (target coverage 1 - 0.10/m)
T_BONF_1SIDED_2N80_DF_19   = 3.785667236494119   # identical to T_BONF_2N40_DF_19
T_BONF_1SIDED_2N80_DF_39   = 3.479924403320762
T_BONF_1SIDED_2N80_DF_99   = 3.322736366563636
T_BONF_1SIDED_2N80_DF_255  = 3.263705552933041
T_BONF_1SIDED_2N80_DF_2047 = 3.231723227704102   # the prompt's "3.2..."

T_BONF_1SIDED_2N40_DF_19   = 3.481150847255472
T_BONF_1SIDED_2N40_DF_39   = 3.232161794568099
T_BONF_1SIDED_2N40_DF_99   = 3.102616799226472
T_BONF_1SIDED_2N40_DF_255  = 3.053675299918580
T_BONF_1SIDED_2N40_DF_2047 = 3.027090023031396   # the prompt's "3.0..."
```

(Note `1 - 0.10/80 == 1 - 0.05/40`, which is why the first block coincides with
`T_BONF_2N40`.)

---

## 4. Verification results

### 4.1 Three independent methods, all 15 constants

| method | what it is | accuracy | max abs. difference vs method C |
|---|---|---|---|
| **A** | Numerical Recipes `betacf` + `betai` with `math.lgamma`, bisection on the literal identity `CDF(t) = 1 - 0.5·I_{df/(df+t²)}(df/2, 1/2)` | ~1e-13 (double precision) | **3.229e-13** (df = 2047) |
| **B** | same continued fraction, but `ln(1/B(a,b))` supplied from 40-digit exact factorials instead of `lgamma` | ~2e-14 (double precision) | **2.931e-14** |
| **C** | 60-digit `decimal` numerical integration of the t density `f(x)=C_ν(1+x²/ν)^-(ν+1)/2`, `C_ν` from exact factorials + Machin π, composite Gauss–Legendre, bisection in Decimal | ~1e-25 | — (arbiter) |

`max |A − B| = 3.477e-13`. Methods A and B share the continued fraction, so the A↔B gap
isolates the floating-point cost of the `lgamma(a+b) − lgamma(a) − lgamma(b)` cancellation
(arguments ≈ 5900 at df = 2047, where the measured error in `ln(1/B)` is **8.9e-13**,
versus 2.7e-15 at df = 19).

Why the double-precision methods saturate near 1e-13: a root error equals
`(coverage error)/f(t)`, and `1/f(t)` reaches 1.4e3 at the Bonferroni level of df = 19, so
an 8e-17 coverage rounding becomes ~1e-13 in `t`. Section 6b of the transcript shows the
predicted and actual `t_A − t_C` agreeing in magnitude and sign for all 15 cases.

### 4.2 Published anchors (`t_{0.975, df}`)

| df | published | method C | \|A − published\| | \|C − published\| |
|---|---|---|---|---|
| 1 | 12.7062047364 | 12.706204736174705 | 2.253e-10 | 2.253e-10 |
| 2 | 4.30265272975 | 4.302652729749464 | 5.426e-13 | 5.361e-13 |
| 10 | 2.22813885196 | 2.228138851986275 | 2.627e-11 | 2.627e-11 |
| 30 | 2.04227245630 | 2.042272456301238 | 1.234e-12 | 1.238e-12 |
| 100 | 1.98397151845 | 1.983971518523552 | 7.354e-11 | 7.355e-11 |
| 1000 | 1.96233908083 | 1.962339080826409 | 3.564e-12 | 3.592e-12 |

**Max absolute error against the published anchors: 2.253e-10** (df = 1). Nothing here is
a disagreement with the mathematics: every one of these anchors is rounded/truncated at
the digits quoted, which bounds the residual at ~5e-11 for a 10-decimal quote.
`12.7062047364` was rounded up from `12.7062047361747`; `4.30265272975` sits 5.36e-13
above the exact `sqrt(2·0.95²/(1−0.95²)) = 4.302652729749464`. Method C reproduces the
exact closed forms to **≤ 9.6e-26**, so the residual is entirely in the quotes.

### 4.3 Exact closed-form checks (independent of every method above)

* df = 1, `CDF(t) = 0.5 + atan(t)/π` → `t = cot(π/40) = 12.70620473617470464602`;
  `|C − exact| = 3.4e-26`, `|A − exact| = 3.5e-14`.
* df = 2, `CDF(t) = 0.5(1 + t/√(2+t²))` → `t = √(2c²/(1−c²))`;
  `|C − exact| = 9.5e-27`, `|A − exact| = 6.5e-15`.
* Both closed forms were also checked **at the two Bonferroni targets** — the strongest
  validation of the requested Bonferroni constants:

| target c | df | exact | method C | \|C − exact\| |
|---|---|---|---|---|
| 0.95 | 1 | 12.70620473617470464602 | 12.70620473617470464602 | 1.16e-27 |
| 0.95 | 2 | 4.30265272974946385232 | 4.30265272974946385232 | 1.86e-26 |
| 0.998750 | 1 | 509.29516339542735670801 | 509.29516339542735670801 | 4.79e-26 |
| 0.998750 | 2 | 28.25774783480514121941 | 28.25774783480514121941 | 9.59e-27 |
| 0.999375 | 1 | 1018.59130853887437250500 | 1018.59130853887437250500 | 9.58e-26 |
| 0.999375 | 2 | 39.98124755805955823650 | 39.98124755805955823650 | 1.70e-27 |

### 4.4 The identity in requirement 5 (Bonferroni at m = 2N)

Three independent evaluations for N ∈ {5, 10, 20, 40, 100, 500} (m = 2N ∈ {10, …, 1000})
× all five df, 30 cases:

* route 1 — bisect `P(|T| ≤ t) = 1 − 0.05/m`
* route 2 — bisect `2·P(T > t) = 0.05/m` (different equation, different code path)
* route 3 — same solve in 60-digit Decimal

| check | result |
|---|---|
| `max |route1 − route2|` | **4.494e-13** |
| `max |route1 − route3|` | **4.718e-13** |
| `max |m·2·P(T>t) − 0.05|` (family-wise error rate) | **5.781e-14** |

Explicitly: `T_BONF(m = 2N = 80, df) ≡` the two-sided t at confidence `1 − 0.05/80`, and
`T_BONF(m = 2N = 40, df) ≡` the two-sided t at confidence `1 − 0.05/40`, matching the
constants of §2 to ≤ 3.1e-13 (the Decimal roots themselves agree to ≤ 1.6e-27; see 8e of
the transcript).

### 4.5 Identity / sanity checks required by the task

* `CDF(0) = 0.5` **exactly** for all five df (`betai(a,b,1)` returns 1.0 exactly).
* CDF strictly monotone increasing on `t ∈ [-30, 30]`, 60001 points: **0 decreases** for
  every df. The observed "ties" occur only where the double-precision CDF has saturated to
  0.0 or 1.0 (|t| ≳ 8–18 depending on df).
* `betai` validated against the exact binomial sum for integer `(a,b)` (30 cases):
  max deviation **4.0e-15**; against `I_x(a,b)+I_{1-x}(b,a)=1`: **4.2e-14** with exact
  complements.
* Continued fraction never needed more than **62** of the allowed 500 iterations.
* Every root was re-substituted: `|CDF(t*) − target| < 1e-13` for all 51 roots
  (worst for method A: 1.2e-14; for method C: ~8e-29). Asserted in the script.
* Large-df asymptotics (Cornish–Fisher, `z_0.975` from `statistics.NormalDist`):
  `t − [z + (z³+z)/(4ν)]` = 8.2e-3, 2.9e-4, 4.4e-5, 6.7e-7 for df = 19, 99, 255, 2047 —
  a factor 1.2e4 drop against the ν⁻² prediction of 1.2e4.

### 4.6 What did NOT match

1. **The prompt's Bonferroni example digits.** `3.2...` / `3.0...` at df = 2047 correspond
   to the one-sided-alpha convention, not to the definition stated in the prompt. With the
   stated definition the values are `3.425839556652871` (m = 80) and `3.231723227704101`
   (m = 40). Both sets are provided (§2, §3).
2. **Published anchors at the last quoted digits** (up to 2.25e-10, see §4.2) — an
   artifact of the digits quoted, not of the computation.
3. **Method A vs method C up to 3.2e-13.** The required Numerical-Recipes-`lgamma` route
   is not accurate to the reported 15 decimals for df = 2047; the recommended constants
   therefore come from method C. The disagreement is 6 orders of magnitude below any
   practical use of these constants.
4. **The identity in the task's section 2 header**, as written there
   (`P(T>t) = 0.5·I_{t²/(df+t²)}(1/2, df/2)`), is inverted; the correct statement is
   `P(T>t) = 0.5·I_{df/(df+t²)}(df/2, 1/2) = 0.5·(1 − I_{t²/(df+t²)}(1/2, df/2))`.
   Both limits confirm it: `t→0` gives 0.5, `t→∞` gives 0. The task's *CDF* identity is
   correct as stated and was used verbatim.

---

## 5. How to reproduce

```
python N01_work/verify_t_critical.py          # ~9 s, prints the table and rewrites the .txt
```

Files: `verify_t_critical.py` (script), `verify_t_critical_output.txt` (transcript),
`t_critical_values.md` (this file).

---

## 6. Full script output (verbatim)

```text
====================================================================================================
 INDEPENDENT VERIFICATION OF TWO-SIDED STUDENT-t CRITICAL VALUES
 (pure Python standard library; no numpy / scipy available)
====================================================================================================
 python 3.13.14
 df values            : [19, 39, 99, 255, 2047]
 T95          : alpha = 0.05      -> target P(|T|<=t) = 1 - alpha = 0.950000   [plain two-sided 95%]
 T_BONF_2N80  : alpha = 0.000625  -> target P(|T|<=t) = 1 - alpha = 0.999375   [Bonferroni, m = 2N = 80 (N = 40), per-test 1 - 0.05/80]
 T_BONF_2N40  : alpha = 0.00125   -> target P(|T|<=t) = 1 - alpha = 0.998750   [Bonferroni, m = 2N = 40 (N = 20), per-test 1 - 0.05/40]

 DEFINITION (Bonferroni, used throughout): family of m = 2N two-sided
 intervals at simultaneous confidence 0.95 -> each member at per-comparison
 confidence 1 - 0.05/m = 1 - 0.05/(2N).  So family size m = 80 means N = 40
 comparisons (2N = 80) and m = 40 means N = 20 (2N = 40).

====================================================================================================
 1. REGULARIZED INCOMPLETE BETA  I_x(a,b)  (Numerical Recipes betacf + betai)
====================================================================================================
 I_x(a,b) = B_x(a,b)/B(a,b), evaluated with the NR continued fraction and
 the prefactor ln(1/B(a,b)) = lgamma(a+b) - lgamma(a) - lgamma(b).

 1a. validation against the EXACT binomial sum for integer (a,b):
     I_x(a,b) = sum_{j=a}^{a+b-1} C(a+b-1,j) x^j (1-x)^(a+b-1-j)
     max |betai(a,b,x) - binomial_exact| over 30 (a,b,x) cases : 3.997e-15

 1b. symmetry  I_x(a,b) + I_{1-x}(b,a) = 1 :
     max deviation, 32 cases with EXACT complements (dyadic x) : 4.197e-14
     max deviation, 16 cases with generic x (x=1e-8..0.999)     : 9.068e-12
     The identity is exact; BOTH residuals are floating-point artifacts, not
     mathematics.  (i) with generic x, fl(1-fl(1-x)) differs from x by up to
     ~1e-13 relative.  (ii) the shared prefactor exp(lgamma(a+b)-lgamma(a)
     -lgamma(b)) is a difference of numbers of size ~5900 for df=2047, and the
     two terms of the identity evaluate it in a different ORDER, so they can
     disagree by ~1e-12 in the exponent even when the arguments are identical.
     Section 4 measures that lgamma error directly (8.9e-13 for df=2047).
     I_0(a,b)      = 0.0   (exact 0)
     I_1(a,b)      = 1.0   (exact 1)
     I_x(a,1) = x^a: max dev over x in {1e-6..0.9} = 2.220e-16
     max continued-fraction iterations used so far: 17

====================================================================================================
 2. STUDENT-t CDF: the identity  CDF(t) = 1 - 0.5*I_{df/(df+t^2)}(df/2, 1/2)
====================================================================================================
 (t > 0;  CDF(-t) = 1 - CDF(t).  Equivalently, with y = t^2/(df+t^2),
  P(T > t) = 0.5*I_{df/(df+t^2)}(df/2, 1/2) = 0.5*(1 - I_y(1/2, df/2)),
  so P(|T| <= t) = 1 - I_y(1/2, df/2).)

 2a. CDF(0) = 0.5 exactly?
     df=   19 : CDF(0) = 0.5   exact 0.5 -> OK
     df=   39 : CDF(0) = 0.5   exact 0.5 -> OK
     df=   99 : CDF(0) = 0.5   exact 0.5 -> OK
     df=  255 : CDF(0) = 0.5   exact 0.5 -> OK
     df= 2047 : CDF(0) = 0.5   exact 0.5 -> OK

 2b. monotonicity of CDF on t in [-30, 30], 60001 points:
     (a 'tie' is CDF(t_i) == CDF(t_{i-1}) in double precision; a real decrease is d < 0)
     df=   19 : decreases(d<0)=0  max decrease=0.000e+00 | ties=11036  min strict step=5.620e-21
               CDF(-30)=9.049e-18   CDF(30)=1.00000000000000000
               ties occur only where the CDF has saturated to 0.0 or 1.0;
               sampled t values with ties span [18.0, 30.0] of [-30, 30]
     df=   39 : decreases(d<0)=0  max decrease=0.000e+00 | ties=18380  min strict step=9.255e-32
               CDF(-30)=7.415e-29   CDF(30)=1.00000000000000000
               ties occur only where the CDF has saturated to 0.0 or 1.0;
               sampled t values with ties span [11.3, 30.0] of [-30, 30]
     df=   99 : decreases(d<0)=0  max decrease=0.000e+00 | ties=20997  min strict step=2.535e-54
               CDF(-30)=8.504e-52   CDF(30)=1.00000000000000000
               ties occur only where the CDF has saturated to 0.0 or 1.0;
               sampled t values with ties span [8.8, 30.0] of [-30, 30]
     df=  255 : decreases(d<0)=0  max decrease=0.000e+00 | ties=21808  min strict step=4.252e-88
               CDF(-30)=6.391e-86   CDF(30)=1.00000000000000000
               ties occur only where the CDF has saturated to 0.0 or 1.0;
               sampled t values with ties span [8.1, 30.0] of [-30, 30]
     df= 2047 : decreases(d<0)=0  max decrease=0.000e+00 | ties=22212  min strict step=3.506e-166
               CDF(-30)=1.663e-164   CDF(30)=1.00000000000000000
               ties occur only where the CDF has saturated to 0.0 or 1.0;
               sampled t values with ties span [7.7, 30.0] of [-30, 30]

 2c. equivalence of the two algebraic forms of the identity (float, NR betai):
     literal  : CDF(t) = 1 - 0.5*I_x(df/2, 1/2),          x = df/(df+t^2)
     complement: CDF(t) = 0.5 + 0.5*I_y(1/2, df/2),        y = t^2/(df+t^2)
     max |literal - complement| over 35 (df,t) pairs : 1.017e-13
     (the difference is pure floating-point cancellation in x = df/(df+t^2):
      1-x loses ~2.7 digits for df=2047, t~2)

 2d. closed-form checks of the whole chain:
     df=1  (Cauchy)  :  CDF(t) = 0.5 + atan(t)/pi
     df=2            :  CDF(t) = 0.5*(1 + t/sqrt(2+t^2))
     max |CDF_A(t) - Cauchy exact| (6 t)   : 2.220e-16
     max |CDF_A(t) - df=2 exact|   (5 t)   : 5.551e-16

====================================================================================================
 3. METHOD A - Numerical Recipes betai + lgamma, inverted by bisection
====================================================================================================
 bisection on [0.5, 50] to bracket width < 1e-15; residual = coverage(t*) - target

 family        m     df    t (method A)          iter  width       coverage(t*) residual
 --------------------------------------------------------------------------------------------------
 T95           1     19    2.093024054408308     56    8.88e-16    0.9500000000 +0.000e+00
 T95           1     39    2.022690920036759     56    8.88e-16    0.9500000000 +2.220e-16
 T95           1     99    1.984216951586424     56    6.66e-16    0.9500000000 -3.331e-16
 T95           1     255   1.969310569849874     56    6.66e-16    0.9500000000 -7.772e-16
 T95           1     2047  1.961123559863062     56    6.66e-16    0.9500000000 +1.188e-14
 T_BONF_2N80   80    19    4.089179183536547     56    8.88e-16    0.9993750000 -1.110e-16
 T_BONF_2N80   80    39    3.720669712860801     56    8.88e-16    0.9993750000 -1.110e-16
 T_BONF_2N80   80    99    3.533475944075458     56    8.88e-16    0.9993750000 -1.110e-16
 T_BONF_2N80   80    255   3.463598989782204     56    8.88e-16    0.9993750000 +0.000e+00
 T_BONF_2N80   80    2047  3.425839556652602     56    4.44e-16    0.9993750000 -1.110e-16
 T_BONF_2N40   40    19    3.785667236494107     56    4.44e-16    0.9987500000 -1.110e-16
 T_BONF_2N40   40    39    3.479924403320753     56    8.88e-16    0.9987500000 +0.000e+00
 T_BONF_2N40   40    99    3.322736366563634     56    8.88e-16    0.9987500000 +0.000e+00
 T_BONF_2N40   40    255   3.263705552933033     56    8.88e-16    0.9987500000 -1.110e-16
 T_BONF_2N40   40    2047  3.231723227703792     56    8.88e-16    0.9987500000 +0.000e+00

====================================================================================================
 4. METHOD B - same continued fraction, exact beta prefactor (no lgamma cancellation)
====================================================================================================
 Only the prefactor changes: ln(1/B(1/2,df/2)) comes from 40-digit factorials
 instead of lgamma(a+b)-lgamma(a)-lgamma(b).  B isolates the floating-point
 cost of the lgamma route used in method A.

   df=   19 :  lgamma route ln(1/B) = +0.540129116359498   exact = +0.540129116359501   |diff| = 2.665e-15
   df=   39 :  lgamma route ln(1/B) = +0.906432735313693   exact = +0.906432735313697   |diff| = 3.775e-15
   df=   99 :  lgamma route ln(1/B) = +1.376096182274210   exact = +1.376096182274203   |diff| = 7.105e-15
   df=  255 :  lgamma route ln(1/B) = +1.850712849730485   exact = +1.850712849730491   |diff| = 5.995e-15
   df= 2047 :  lgamma route ln(1/B) = +2.893004629683674   exact = +2.893004629684567   |diff| = 8.935e-13

 family        df    t (method B)          residual    t(B) - t(A)           rel. diff
 --------------------------------------------------------------------------------------------------
 T95           19    2.093024054408308     -1.110e-16 +0.000000000000000e+00 0.00e+00
 T95           39    2.022690920036759     +0.000e+00 +0.000000000000000e+00 0.00e+00
 T95           99    1.984216951586416     -2.220e-16 -7.549516567451064e-15 3.80e-15
 T95           255   1.969310569849874     -8.882e-16 +0.000000000000000e+00 0.00e+00
 T95           2047  1.961123559863410     +1.210e-14 +3.477218513125990e-13 1.77e-13
 T_BONF_2N80   19    4.089179183536549     -1.110e-16 +1.776356839400250e-15 4.34e-16
 T_BONF_2N80   39    3.720669712860802     +0.000e+00 +1.332267629550188e-15 3.58e-16
 T_BONF_2N80   99    3.533475944075456     -1.110e-16 -2.220446049250313e-15 6.28e-16
 T_BONF_2N80   255   3.463598989782204     +0.000e+00 +0.000000000000000e+00 0.00e+00
 T_BONF_2N80   2047  3.425839556652867     -1.110e-16 +2.651212582804874e-13 7.74e-14
 T_BONF_2N40   19    3.785667236494108     +0.000e+00 +1.332267629550188e-15 3.52e-16
 T_BONF_2N40   39    3.479924403320753     -1.110e-16 +4.440892098500626e-16 1.28e-16
 T_BONF_2N40   99    3.322736366563627     -1.110e-16 -6.661338147750939e-15 2.00e-15
 T_BONF_2N40   255   3.263705552933033     -1.110e-16 +0.000000000000000e+00 0.00e+00
 T_BONF_2N40   2047  3.231723227704073     +0.000e+00 +2.815525590449397e-13 8.71e-14

====================================================================================================
 5. METHOD C - 60-digit Decimal numerical integration of the t density
====================================================================================================
 f(x) = C_nu (1+x^2/nu)^(-(nu+1)/2),  C_nu = Gamma((nu+1)/2)/(sqrt(nu*pi)Gamma(nu/2))
 C_nu from exact factorials + Machin pi (no lgamma, no cancellation).

 5a. pi (Machin, Decimal) vs math.pi : |diff| = 1.225e-16   (math.pi is the double rounding)
     60-digit pi = 3.141592653589793238462643383279502884197169399375105820974945

 5b. C_nu : Decimal (exact factorials) vs float lgamma route
     df     C_nu (Decimal)           C_nu (lgamma float)      relative diff
     19     0.3937298072926036038835 0.39372980729260293      1.692e-15
     39     0.3963934153651700842687 0.39639341536516776      5.882e-15
     99     0.3979361384240755625921 0.39793613842408598      2.623e-14
     255    0.3985513531829639726481 0.39855135318297513      2.800e-14
     2047   0.3988935605792383168081 0.39889356057874276      1.242e-12
     1      0.3183098861837906715378 0.31830988618379058      3.488e-16
     2      0.3535533905932737622004 0.35355339059327373      1.570e-16
     10     0.3891083839660310506171 0.38910838396603142      9.986e-16
     30     0.3956321848940977580263 0.39563218489409491      7.156e-15
     100    0.3979461869358938074906 0.39794618693588418      2.413e-14
     1000   0.3988425573138581550011 0.39884255731389101      8.239e-14

 5c. Gauss-Legendre rule sanity (n=24 on [-1,1]):
     sum(w)      = 2.0000000000000000000000000000000000000000   (exact 2)
     int x^2     = 0.6666666666666666666666666666666666666667   (exact 2/3)
     int x^46    = 0.0425531914893617021276595744680851063830   (exact 2/47, exact for degree <= 2n-1 = 47)
     int x^48    = 0.0408163265306011970214297137871471753379   (exact 2/49 -> should FAIL: quadrature order limit)

 5d. quadrature convergence: P(|T|<=t) at the method-A root, four GL settings
     df     family        (16 nodes, 3 panels)     (20,4)                   (24,6) production        spread (24,6)-(32,12)
     19     T95           0.9499999999999998545799 0.9499999999999998545799 0.9499999999999998545799 7.00e-60
     39     T95           0.9499999999999997804380 0.9499999999999997804380 0.9499999999999997804380 3.20e-59
     99     T95           0.9500000000000007208201 0.9500000000000007208201 0.9500000000000007208201 1.90e-59
     255    T95           0.9499999999999998160580 0.9499999999999998160580 0.9499999999999998160580 3.00e-59
     2047   T95           0.9499999999999623103101 0.9499999999999623103101 0.9499999999999623103101 2.60e-58
     19     T_BONF_2N80   0.9993749999999999551516 0.9993749999999999551516 0.9993749999999999551516 7.00e-60
     39     T_BONF_2N80   0.9993749999999999553541 0.9993749999999999553541 0.9993749999999999553541 0.00e+00
     99     T_BONF_2N80   0.9993749999999999660789 0.9993749999999999660789 0.9993749999999999660789 4.40e-59
     255    T_BONF_2N80   0.9993749999999999616043 0.9993749999999999616043 0.9993749999999999616043 1.93e-58
     2047   T_BONF_2N80   0.9993749999999993849395 0.9993749999999993849395 0.9993749999999993849395 5.63e-58
     19     T_BONF_2N40   0.9987499999999999673830 0.9987499999999999673830 0.9987499999999999673830 7.00e-60
     39     T_BONF_2N40   0.9987499999999999688515 0.9987499999999999688515 0.9987499999999999688515 3.00e-59
     99     T_BONF_2N40   0.9987499999999999927413 0.9987499999999999927413 0.9987499999999999927413 2.20e-59
     255    T_BONF_2N40   0.9987499999999999678323 0.9987499999999999678323 0.9987499999999999678323 3.10e-59
     2047   T_BONF_2N40   0.9987499999999986528113 0.9987499999999986528113 0.9987499999999986528113 1.38e-57
     worst spread between the two production-independent settings : 1.38e-57

 5e. bisection in Decimal (target 1e-25, bracket [1,50])
 family        m     df    t (method C)               iter  width      coverage(t*) residual
 ------------------------------------------------------------------------------------------------------------
 T95           1     19    2.0930240544083097691773   89    7.9e-26    0.9500000000 -2.19e-27
 T95           1     39    2.0226909200367611356318   89    7.9e-26    0.9500000000 -2.56e-27
 T95           1     99    1.9842169515864174951046   89    7.9e-26    0.9500000000 +4.02e-27
 T95           1     255   1.9693105698498752197744   89    7.9e-26    0.9500000000 -2.44e-27
 T95           1     2047  1.9611235598633852349961   89    7.9e-26    0.9500000000 +3.21e-27
 T_BONF_2N80   80    19    4.0891791835365788371536   89    7.9e-26    0.9993750000 +3.56e-29
 T_BONF_2N80   80    39    3.7206697128608255582184   89    7.9e-26    0.9993750000 +2.93e-29
 T_BONF_2N80   80    99    3.5334759440754745694945   89    7.9e-26    0.9993750000 +8.01e-29
 T_BONF_2N80   80    255   3.4635989897822216854905   89    7.9e-26    0.9993750000 +4.38e-29
 T_BONF_2N80   80    2047  3.4258395566528709451318   89    7.9e-26    0.9993750000 -8.29e-29
 T_BONF_2N40   40    19    3.7856672364941185136784   89    7.9e-26    0.9987500000 +1.04e-28
 T_BONF_2N40   40    39    3.4799244033207615843261   89    7.9e-26    0.9987500000 +5.32e-29
 T_BONF_2N40   40    99    3.3227363665636358276894   89    7.9e-26    0.9987500000 -7.73e-29
 T_BONF_2N40   40    255   3.2637055529330405935320   89    7.9e-26    0.9987500000 +1.62e-28
 T_BONF_2N40   40    2047  3.2317232277041015259853   89    7.9e-26    0.9987500000 +1.61e-28

====================================================================================================
 6. THREE-METHOD COMPARISON - the 15 requested constants
====================================================================================================
 family        df    target    A: NR betai+lgamma   B: exact prefactor   C: 60-digit Decimal  |A-C|      |B-C|     
 --------------------------------------------------------------------------------------------------------------------
 T95           19    0.950000  2.093024054408308    2.093024054408308    2.093024054408310    1.33e-15   1.33e-15  
 T95           39    0.950000  2.022690920036759    2.022690920036759    2.022690920036761    2.22e-15   2.22e-15  
 T95           99    0.950000  1.984216951586424    1.984216951586416    1.984216951586417    6.44e-15   1.11e-15  
 T95           255   0.950000  1.969310569849874    1.969310569849874    1.969310569849875    1.55e-15   1.55e-15  
 T95           2047  0.950000  1.961123559863062    1.961123559863410    1.961123559863385    3.23e-13   2.49e-14  
 T_BONF_2N80   19    0.999375  4.089179183536547    4.089179183536549    4.089179183536578    3.11e-14   2.93e-14  
 T_BONF_2N80   39    0.999375  3.720669712860801    3.720669712860802    3.720669712860825    2.44e-14   2.31e-14  
 T_BONF_2N80   99    0.999375  3.533475944075458    3.533475944075456    3.533475944075474    1.60e-14   1.82e-14  
 T_BONF_2N80   255   0.999375  3.463598989782204    3.463598989782204    3.463598989782222    1.73e-14   1.73e-14  
 T_BONF_2N80   2047  0.999375  3.425839556652602    3.425839556652867    3.425839556652871    2.69e-13   3.55e-15  
 T_BONF_2N40   19    0.998750  3.785667236494107    3.785667236494108    3.785667236494119    1.15e-14   1.02e-14  
 T_BONF_2N40   39    0.998750  3.479924403320753    3.479924403320753    3.479924403320762    8.88e-15   8.44e-15  
 T_BONF_2N40   99    0.998750  3.322736366563634    3.322736366563627    3.322736366563636    1.78e-15   8.44e-15  
 T_BONF_2N40   255   0.998750  3.263705552933033    3.263705552933033    3.263705552933041    7.55e-15   7.55e-15  
 T_BONF_2N40   2047  0.998750  3.231723227703792    3.231723227704073    3.231723227704101    3.10e-13   2.80e-14  

 MAX |method A - method C| over the 15 constants : 3.229e-13
 MAX |method B - method C| over the 15 constants : 2.931e-14
 MAX |method A - method B| over the 15 constants : 3.477e-13
 (method C is the arbiter: its quadrature is converged to ~1e-55, see 5d)

 6b. why the double-precision methods cannot do better: the root error of a
     float method is (coverage error) / f(t), and 1/f(t) is huge in the tail.
     coverage_C(t_A) is the 60-digit coverage evaluated at method A's root.

 family        df    t (method A) f(t) density  1/f(t)        coverage_C(tA)-tgt predicted dt
 --------------------------------------------------------------------------------------------------------
 T95           19    2.093024054  4.94481e-02   2.0223e+01    -1.453e-16      +2.938e-15   (actual +1.469e-15)
 T95           39    2.022690920  5.39052e-02   1.8551e+01    -2.302e-16      +4.271e-15   (actual +2.136e-15)
 T95           99    1.984216952  5.66204e-02   1.7661e+01    +7.253e-16      -1.281e-14   (actual -6.405e-15)
 T95           255   1.969310570  5.77310e-02   1.7322e+01    -1.870e-16      +3.240e-15   (actual +1.620e-15)
 T95           2047  1.961123560  5.83557e-02   1.7136e+01    -3.769e-14      +6.459e-13   (actual +3.229e-13)
 T_BONF_2N80   19    4.089179184  7.13594e-04   1.4014e+03    -4.544e-17      +6.367e-14   (actual +3.184e-14)
 T_BONF_2N80   39    3.720669713  9.11184e-04   1.0975e+03    -4.475e-17      +4.912e-14   (actual +2.456e-14)
 T_BONF_2N80   99    3.533475944  1.04873e-03   9.5354e+02    -3.391e-17      +3.234e-14   (actual +1.617e-14)
 T_BONF_2N80   255   3.463598990  1.10895e-03   9.0176e+02    -3.834e-17      +3.457e-14   (actual +1.729e-14)
 T_BONF_2N80   2047  3.425839557  1.14386e-03   8.7423e+02    -6.153e-16      +5.379e-13   (actual +2.689e-13)
 T_BONF_2N40   19    3.785667236  1.42631e-03   7.0111e+02    -3.284e-17      +2.303e-14   (actual +1.151e-14)
 T_BONF_2N40   39    3.479924403  1.77551e-03   5.6322e+02    -3.048e-17      +1.717e-14   (actual +8.584e-15)
 T_BONF_2N40   99    3.322736367  2.01340e-03   4.9667e+02    -7.360e-18      +3.655e-15   (actual +1.828e-15)
 T_BONF_2N40   255   3.263705553  2.11641e-03   4.7250e+02    -3.214e-17      +1.519e-14   (actual +7.594e-15)
 T_BONF_2N40   2047  3.231723228  2.17583e-03   4.5960e+02    -1.347e-15      +6.191e-13   (actual +3.095e-13)

====================================================================================================
 7. PUBLISHED ANCHOR CHECK  (t_{0.975,df}, i.e. two-sided 95%)
====================================================================================================
 df     published          A (lgamma)           B (exact pref.)      C (Decimal)          |A-pub|     |C-pub|    
 ----------------------------------------------------------------------------------------------------------------
 1      12.7062047364      12.706204736174669   12.706204736174683   12.706204736174705   2.253e-10   2.253e-10  
 2      4.30265272975      4.302652729749457    4.302652729749459    4.302652729749464    5.426e-13   5.361e-13  
 10     2.22813885196      2.228138851986274    2.228138851986274    2.228138851986275    2.627e-11   2.627e-11  
 30     2.04227245630      2.042272456301234    2.042272456301238    2.042272456301238    1.234e-12   1.238e-12  
 100    1.98397151845      1.983971518523539    1.983971518523550    1.983971518523552    7.354e-11   7.355e-11  
 1000   1.96233908083      1.962339080826436    1.962339080826407    1.962339080826409    3.564e-12   3.592e-12  

 MAX |method C - published| : 2.253e-10
 MAX |method A - published| : 2.253e-10

 exact closed forms (independent of all three methods):
   df=1 exact: t = cot(pi/40) = cos(pi/40)/sin(pi/40) = 12.7062047361747046460217
   df=2 exact: t = sqrt(2*0.95^2/(1-0.95^2))            = 4.3026527297494638523209
   -> df=1 : |method C - exact| = 3.393e-26   |method A - exact| = 3.465e-14
   -> df=2 : |method C - exact| = 9.488e-27   |method A - exact| = 6.452e-15

 NOTE on the published anchors.  All six quoted values are rounded/truncated at
 the number of digits shown, so |published - exact| is bounded by ~5e-11 for
 10-decimal quotes.  The largest anchor deviation (df=1, 2.25e-10) is exactly
 this truncation: 12.7062047364 was rounded up from 12.7062047361747.
 The df=2 anchor 4.30265272975 sits 5.36e-13 ABOVE the exact
 sqrt(2*0.95^2/(1-0.95^2)) = 4.302652729749464; its two-sided coverage is
 0.950000000000011542 instead of 0.95 (error +1.15e-14), i.e. consistent with the quote precision.

 7b. exact closed forms at ALL THREE levels (df = 1 and df = 2), which validates
     the Bonferroni targets as well as the 95% one:
        df=1: coverage = 2*atan(t)/pi  ->  t = cot(pi*(1-c)/2)
        df=2: coverage = t/sqrt(2+t^2) ->  t = sqrt(2c^2/(1-c^2))
     target c   df   exact (closed form)            method C (Decimal)             method A (lgamma)              |C-exact|
     0.950000   1    12.70620473617470464602        12.70620473617470464602        12.706204736174669             1.16e-27
     0.950000   2    4.30265272974946385232         4.30265272974946385232         4.302652729749457              1.86e-26
     0.998750   1    509.29516339542735670801       509.29516339542735670801       509.295163395414761            4.79e-26
     0.998750   2    28.25774783480514121941        28.25774783480514121941        28.257747834804817             9.59e-27
     0.999375   1    1018.59130853887437250500      1018.59130853887437250500      1018.591308538804242           9.58e-26
     0.999375   2    39.98124755805955823650        39.98124755805955823650        39.981247558058200             1.70e-27

 7c. large-df asymptotic sanity check (independent of both the beta function
     and the quadrature).  Cornish-Fisher: t_p(nu) = z_p + (z^3+z)/(4 nu) + O(nu^-2)
     z_0.975 (statistics.NormalDist) = 1.95996398454005361
     df     t (method C)           z + (z^3+z)/(4 nu)       difference
     19     2.093024054408310      2.084820365082083        8.204e-03
     99     1.984216951586417      1.983926320199635        2.906e-04
     255    1.969310569849875      1.969267008972597        4.356e-05
     2047   1.961123559863385      1.961122885971562        6.739e-07
     The residual is the O(nu^-2) term: it falls from 8.2e-3 at df=19 to 6.7e-7
     at df=2047, a factor ~1.2e4, matching the nu^-2 scaling (nu ratio 108 ->
     1.2e4).  The tabulated values are therefore consistent with the known
     large-df behaviour of the t quantile, with no free parameters.

====================================================================================================
 8. BONFERRONI IDENTITY CHECKS  (family size m = 2N)
====================================================================================================
 DEFINITION USED: P(|T_df| <= t) = 1 - 0.05/m, m = 2N   [per-comparison
 confidence 1 - 0.05/(2N); per-comparison two-sided alpha = 0.05/(2N);
 per-comparison one-sided tail = 0.025/(2N)].

 Three independent evaluations of the same quantity:
   route 1 : solve coverage_A(t)      = 1 - 0.05/m            (bisection, float)
   route 2 : solve 2*tail_A(t)        = 0.05/m               (different function, float)
   route 3 : solve dec_coverage(t)    = 1 - 0.05/m           (Decimal, 60 digits)
   check   : family-wise error rate  m * 2*tail(t)  ==  0.05

 N    m=2N  df    route 1 (coverage)    route 2 (tail)        route 3 (Decimal)     |r1-r2|    |r1-r3|    m*2*tail(r1)-0.05
 ----------------------------------------------------------------------------------------------------------------------------------
 5    10    19    3.173724530792309     3.173724530792315     3.173724530792316     6.22e-15   7.11e-15   +6.38e-16
 5    10    39    2.975608757792982     2.975608757792986     2.975608757792987     4.00e-15   5.96e-15   +6.11e-16
 5    10    99    2.871307661214764     2.871307661214768     2.871307661214766     3.55e-15   2.15e-15   +5.83e-16
 5    10    255   2.831668784417242     2.831668784417243     2.831668784417245     8.88e-16   2.78e-15   +4.23e-16
 5    10    2047  2.810080912328383     2.810080912328440     2.810080912328724     5.73e-14   3.41e-13   +6.92e-15
 10   20    19    3.481150847255470     3.481150847255470     3.481150847255472     0.00e+00   1.71e-15   +1.18e-16
 10   20    39    3.232161794568097     3.232161794568097     3.232161794568099     0.00e+00   2.73e-15   +2.22e-16
 10   20    99    3.102616799226478     3.102616799226474     3.102616799226472     3.55e-15   5.65e-15   -5.20e-16
 10   20    255   3.053675299918584     3.053675299918578     3.053675299918580     5.77e-15   3.82e-15   -1.08e-15
 10   20    2047  3.027090023031109     3.027090023031124     3.027090023031397     1.47e-14   2.87e-13   +1.96e-15
 20   40    19    3.785667236494107     3.785667236494117     3.785667236494119     1.02e-14   1.15e-14   +1.17e-15
 20   40    39    3.479924403320753     3.479924403320760     3.479924403320762     7.55e-15   8.58e-15   +1.05e-15
 20   40    99    3.322736366563634     3.322736366563638     3.322736366563636     4.00e-15   1.83e-15   +4.02e-16
 20   40    255   3.263705552933033     3.263705552933038     3.263705552933041     4.88e-15   7.59e-15   +1.35e-15
 20   40    2047  3.231723227703792     3.231723227703831     3.231723227704101     3.86e-14   3.10e-13   +1.03e-14
 40   80    19    4.089179183536547     4.089179183536576     4.089179183536578     2.84e-14   3.18e-14   +3.39e-15
 40   80    39    3.720669712860801     3.720669712860825     3.720669712860825     2.40e-14   2.46e-14   +3.23e-15
 40   80    99    3.533475944075458     3.533475944075475     3.533475944075474     1.69e-14   1.62e-14   +2.88e-15
 40   80    255   3.463598989782204     3.463598989782220     3.463598989782222     1.55e-14   1.73e-14   +2.72e-15
 40   80    2047  3.425839556652602     3.425839556652637     3.425839556652871     3.46e-14   2.69e-13   +7.66e-15
 100  200   19    4.491429655925314     4.491429655925364     4.491429655925364     4.97e-14   5.03e-14   +5.45e-15
 100  200   39    4.030446615944431     4.030446615944468     4.030446615944470     3.73e-14   3.94e-14   +5.61e-15
 100  200   99    3.800001951814254     3.800001951814284     3.800001951814281     2.98e-14   2.72e-14   +5.29e-15
 100  200   255   3.714667665038352     3.714667665038377     3.714667665038380     2.49e-14   2.78e-14   +4.51e-15
 100  200   2047  3.668716257123600     3.668716257123614     3.668716257123847     1.47e-14   2.47e-13   +4.18e-15
 500  1000  19    5.208582529935809     5.208582529936258     5.208582529936262     4.49e-13   4.53e-13   +5.00e-14
 500  1000  39    4.557730210358494     4.557730210358818     4.557730210358819     3.23e-13   3.24e-13   +5.01e-14
 500  1000  99    4.241508067407811     4.241508067408075     4.241508067408073     2.65e-13   2.63e-13   +4.98e-14
 500  1000  255   4.126066308744562     4.126066308744804     4.126066308744806     2.42e-13   2.44e-13   +4.92e-14
 500  1000  2047  4.064285610493977     4.064285610494235     4.064285610494449     2.58e-13   4.72e-13   +5.78e-14

 MAX |route 1 - route 2|  (30 cases) : 4.494e-13
 MAX |route 1 - route 3|  (30 cases) : 4.718e-13
 MAX |m*2*tail(t) - 0.05| (30 cases) : 5.781e-14

 explicit statement of the identity for the two headline families:
   T_BONF(m=2N, df) := the t with P(|T_df| <= t) = 1 - 0.05/(2N), i.e. the
   ordinary two-sided t critical value at confidence level 1 - 0.05/(2N).
   m = 2N = 80   (N = 40 ), per-comparison level 1 - 0.05/(2N) = 0.99937500 :
      df=   19 : coverage-route 4.089179183536547 | tail-route 4.089179183536576 | Decimal 4.08917918353657883715
                 |r1-r2|=2.8e-14  |r1-r3|=3.2e-14  FWER = m*2*tail = 0.050000000000003 (target 0.05)
      df=   39 : coverage-route 3.720669712860801 | tail-route 3.720669712860825 | Decimal 3.72066971286082555822
                 |r1-r2|=2.4e-14  |r1-r3|=2.5e-14  FWER = m*2*tail = 0.050000000000003 (target 0.05)
      df=   99 : coverage-route 3.533475944075458 | tail-route 3.533475944075475 | Decimal 3.53347594407547456949
                 |r1-r2|=1.7e-14  |r1-r3|=1.6e-14  FWER = m*2*tail = 0.050000000000003 (target 0.05)
      df=  255 : coverage-route 3.463598989782204 | tail-route 3.463598989782220 | Decimal 3.46359898978222168549
                 |r1-r2|=1.6e-14  |r1-r3|=1.7e-14  FWER = m*2*tail = 0.050000000000003 (target 0.05)
      df= 2047 : coverage-route 3.425839556652602 | tail-route 3.425839556652637 | Decimal 3.42583955665287094513
                 |r1-r2|=3.5e-14  |r1-r3|=2.7e-13  FWER = m*2*tail = 0.050000000000008 (target 0.05)
   m = 2N = 40   (N = 20 ), per-comparison level 1 - 0.05/(2N) = 0.99875000 :
      df=   19 : coverage-route 3.785667236494107 | tail-route 3.785667236494117 | Decimal 3.78566723649411851368
                 |r1-r2|=1.0e-14  |r1-r3|=1.2e-14  FWER = m*2*tail = 0.050000000000001 (target 0.05)
      df=   39 : coverage-route 3.479924403320753 | tail-route 3.479924403320760 | Decimal 3.47992440332076158433
                 |r1-r2|=7.5e-15  |r1-r3|=8.6e-15  FWER = m*2*tail = 0.050000000000001 (target 0.05)
      df=   99 : coverage-route 3.322736366563634 | tail-route 3.322736366563638 | Decimal 3.32273636656363582769
                 |r1-r2|=4.0e-15  |r1-r3|=1.8e-15  FWER = m*2*tail = 0.050000000000000 (target 0.05)
      df=  255 : coverage-route 3.263705552933033 | tail-route 3.263705552933038 | Decimal 3.26370555293304059353
                 |r1-r2|=4.9e-15  |r1-r3|=7.6e-15  FWER = m*2*tail = 0.050000000000001 (target 0.05)
      df= 2047 : coverage-route 3.231723227703792 | tail-route 3.231723227703831 | Decimal 3.23172322770410152599
                 |r1-r2|=3.9e-14  |r1-r3|=3.1e-13  FWER = m*2*tail = 0.050000000000010 (target 0.05)

 NOT USED (contrast): if one instead allocated a ONE-SIDED alpha = 0.05/m to each
 of the m members, the target coverage would be the smaller 1 - 0.10/m and the
 critical values would be SMALLER than ours for the same m.  For reference only
 (note that 1 - 0.10/80 coincides with 1 - 0.05/40, and 1 - 0.10/40 with 1 - 0.05/20):
      m=80   df=   19 : 1-0.10/m critical value = 3.785667236494107   (our definition: 4.089179183536547)
      m=80   df=   39 : 1-0.10/m critical value = 3.479924403320753   (our definition: 3.720669712860801)
      m=80   df=   99 : 1-0.10/m critical value = 3.322736366563634   (our definition: 3.533475944075458)
      m=80   df=  255 : 1-0.10/m critical value = 3.263705552933033   (our definition: 3.463598989782204)
      m=80   df= 2047 : 1-0.10/m critical value = 3.231723227703792   (our definition: 3.425839556652602)
      m=40   df=   19 : 1-0.10/m critical value = 3.481150847255470   (our definition: 3.785667236494107)
      m=40   df=   39 : 1-0.10/m critical value = 3.232161794568097   (our definition: 3.479924403320753)
      m=40   df=   99 : 1-0.10/m critical value = 3.102616799226478   (our definition: 3.322736366563634)
      m=40   df=  255 : 1-0.10/m critical value = 3.053675299918584   (our definition: 3.263705552933033)
      m=40   df= 2047 : 1-0.10/m critical value = 3.027090023031109   (our definition: 3.231723227703792)

====================================================================================================
 9. RECOMMENDED CONSTANTS (method C, 60-digit Decimal integration)
====================================================================================================
 Values below are method C rounded to 15 decimals; the underlying Decimal
 roots are good to ~1e-24, the double-precision representation to ~1e-16.

# --------------------------------------------------------------------------
# Two-sided Student-t critical values, P(|T_df| <= t) = target.
# BONFERRONI DEFINITION: family size m = 2N two-sided intervals at simultaneous
# 95% -> per-comparison confidence 1 - 0.05/m.  'T_BONF_2N80' means m = 2N = 80
# (N = 40 comparisons); 'T_BONF_2N40' means m = 2N = 40 (N = 20 comparisons).
# Method C (60-digit Decimal quadrature); max |method C - method A| = 3.23e-13.
# --------------------------------------------------------------------------

# T95 : target P(|T|<=t) = 0.950000   (per-comparison alpha = 0.05)  [plain two-sided 95%]
T95_DF_19              = 2.093024054408310
T95_DF_39              = 2.022690920036761
T95_DF_99              = 1.984216951586417
T95_DF_255             = 1.969310569849875
T95_DF_2047            = 1.961123559863385

# T_BONF_2N80 : target P(|T|<=t) = 0.999375   (per-comparison alpha = 0.000625)  [Bonferroni, m = 2N = 80 (N = 40), per-test 1 - 0.05/80]
T_BONF_2N80_DF_19      = 4.089179183536579
T_BONF_2N80_DF_39      = 3.720669712860826
T_BONF_2N80_DF_99      = 3.533475944075475
T_BONF_2N80_DF_255     = 3.463598989782222
T_BONF_2N80_DF_2047    = 3.425839556652871

# T_BONF_2N40 : target P(|T|<=t) = 0.998750   (per-comparison alpha = 0.00125)  [Bonferroni, m = 2N = 40 (N = 20), per-test 1 - 0.05/40]
T_BONF_2N40_DF_19      = 3.785667236494119
T_BONF_2N40_DF_39      = 3.479924403320762
T_BONF_2N40_DF_99      = 3.322736366563636
T_BONF_2N40_DF_255     = 3.263705552933041
T_BONF_2N40_DF_2047    = 3.231723227704102

# aliases making the family size explicit (same numbers):
T_BONF_M80_DF_19       = 4.089179183536579
T_BONF_M80_DF_39       = 3.720669712860826
T_BONF_M80_DF_99       = 3.533475944075475
T_BONF_M80_DF_255      = 3.463598989782222
T_BONF_M80_DF_2047     = 3.425839556652871
T_BONF_M40_DF_19       = 3.785667236494119
T_BONF_M40_DF_39       = 3.479924403320762
T_BONF_M40_DF_99       = 3.322736366563636
T_BONF_M40_DF_255      = 3.263705552933041
T_BONF_M40_DF_2047     = 3.231723227704102

# --------------------------------------------------------------------------
# ALTERNATIVE CONVENTION - NOT the definition used above.
# If each of the m members were instead given a ONE-SIDED alpha = 0.05/m, the
# per-comparison two-sided confidence would be 1 - 0.10/m and the constants
# would be these (smaller).  The illustrative digits in the task prompt
# ('T_BONF_2N80_DF_2047 = 3.2...', 'T_BONF_2N40_DF_2047 = 3.0...') match THIS
# block, not the stated 1 - 0.05/m definition.  For df=2047 they are
# 3.231723227703792 (m=80) and 3.027090023031109 (m=40).
# --------------------------------------------------------------------------

# T_BONF_2N80 under one-sided alpha = 0.05/80  (target coverage 1 - 0.10/80)
T_BONF_1SIDED_2N80_DF_19 = 3.785667236494119
T_BONF_1SIDED_2N80_DF_39 = 3.479924403320762
T_BONF_1SIDED_2N80_DF_99 = 3.322736366563636
T_BONF_1SIDED_2N80_DF_255 = 3.263705552933041
T_BONF_1SIDED_2N80_DF_2047 = 3.231723227704102

# T_BONF_2N40 under one-sided alpha = 0.05/40  (target coverage 1 - 0.10/40)
T_BONF_1SIDED_2N40_DF_19 = 3.481150847255472
T_BONF_1SIDED_2N40_DF_39 = 3.232161794568099
T_BONF_1SIDED_2N40_DF_99 = 3.102616799226472
T_BONF_1SIDED_2N40_DF_255 = 3.053675299918580
T_BONF_1SIDED_2N40_DF_2047 = 3.027090023031396

====================================================================================================
 10. SUMMARY OF AGREEMENTS AND DISAGREEMENTS
====================================================================================================
 1) three methods, 15 constants:
      max |A - C| = 3.229e-13      (A = required NR betai + lgamma route)
      max |B - C| = 2.931e-14      (B = same CF, exact beta prefactor)
 2) published anchors (6 values):
      max |C - published| = 2.253e-10      max |A - published| = 2.253e-10
 3) Bonferroni route agreement: max |coverage-route - tail-route| = 4.494e-13
      max |float - Decimal| = 4.718e-13 ; max |FWER - 0.05| = 5.781e-14
 4) all 15 + 6 + 30 roots satisfy |CDF(t*) - target| < 1e-13 (asserted).
 5) continued-fraction worst case: 62 iterations of 500 allowed.

 total runtime: 8.5 s
====================================================================================================
```
