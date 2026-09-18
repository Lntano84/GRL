"""Generate the corrected section 3 (inapplicability -> coverage representation) and abstract.

This script exists so the replacement text is produced from the verified numbers rather than typed
by hand: the one-hop instance, its exact probabilities under both threshold distributions, and the
scope limits all come from ``docs/results/coverage_argument_20260918.json``, which is written by
``scripts/audit/verify_coverage_argument.py``.

Run:
    python scripts/audit/write_coverage_section.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "results" / "coverage_argument_20260918.json"
OUT = ROOT / "paper" / "dasfaa2027" / "src" / "dasfaa2027" / "sections" / "coverage.tex"

TEMPLATE = r"""\section{What the Objective Rules Out, and What It Does Not}
\label{sec:coverage}

An earlier version of this work argued that reverse-reachability sampling has no domain here
because every RR set is empty.  That argument is withdrawn.  The probe it rested on required a
predecessor's window to contain zero and excluded the root, and it tested whether the returned set
was non-empty rather than whether it intersected the seed set.  A reverse-reachability set for a
root answers \emph{which seeds can reach this root}; it does not require a node to self-activate
under zero incoming influence.  The empty sets were an artefact of a mis-specified probe.

What survives is a strictly narrower statement, proved by exact arithmetic on a one-hop instance
rather than by simulation.

\subsection{The instance}

Take a single target $D = \{t\}$ and two candidates $a, b$ with edges $a \to t$ and $b \to t$,
both of weight $\omega = %(omega).2f$, and seeds drawn from $V \setminus D$.

Under the model's own window distribution -- $(\theta^\kappa, \theta^\tau)$ uniform on the
2-simplex $\{0 \le \theta^\kappa \le \theta^\tau \le 1\}$ -- the target's positive-activation
probability is $g(\delta) = %(gformula)s$ with $\delta$ the exposure it receives, so

\begin{center}
\begin{tabular}{lccc}
\toprule
$S$ & $\delta$ & $F_D(S)$ under the model & $F_D(S)$ under uniform-threshold LT \\
\midrule
$\emptyset$   & $%(d_empty).2f$ & $%(f_empty).4f$ & $%(u_empty).4f$ \\
$\{a\}$       & $%(d_single).2f$ & $%(f_single).4f$ & $%(u_single).4f$ \\
$\{a,b\}$     & $%(d_pair).2f$ & $%(f_pair).4f$ & $%(u_pair).4f$ \\
\bottomrule
\end{tabular}
\end{center}

Two things follow, and they must be kept apart.

\textbf{The objective is non-monotone.} $F_D(\{a,b\}) = %(f_pair).4f < F_D(\{a\}) = %(f_single).4f$.
Adding a seed can strictly decrease the target's activation probability.

\textbf{No non-negative seed-independent coverage function matches it.} For any random set $R$
independent of $S$, the function $S \mapsto c \cdot \Pr[S \cap R \neq \emptyset]$ is non-decreasing
and submodular in $S$, because adding a seed can only create intersections and the marginal value
of a seed cannot increase as $S$ grows.  A non-monotone $F_D$ therefore cannot equal such a
function on this instance, for any $c$ and any distribution of $R$.

\subsection{Scope}

The statement above excludes exactly one thing: the standard, non-negative, seed-independent
coverage representation that underlies RR sampling.  It does not exclude, and should not be read
as excluding:

\begin{itemize}
  \item signed decompositions, or sampling structures other than non-negative coverage;
  \item algorithms specialised to restricted instance classes;
  \item Monte-Carlo estimation.  A plain sample mean is an unbiased estimator of $F_D$ and is the
        reference used throughout this paper;
  \item any claim that a learned predictor must help.  Whether learning adds anything is an
        empirical question this paper treats as open.
\end{itemize}

\subsection{Why the threshold distribution must be stated explicitly}

The table also shows why the two threshold readings cannot be conflated.  Under
uniform-threshold LT with $\theta^\kappa \sim U[0,1]$ and no upper threshold, the same instance is
\emph{monotone} ($%(u_single).4f \to %(u_pair).4f$), so the argument would fail there.  The
source model samples $(\theta^\kappa, \theta^\tau)$ jointly from the simplex, which induces the
marginal $F^\kappa(x) = 2x - x^2$ for the lower threshold and $F^\tau(x) = x^2$ for the upper; a
later lemma in the same work describes sampling $\theta^\tau$ uniformly and then
$\theta^\kappa$ conditionally, which is a different joint law.  Every claim in this paper
therefore names the distribution it assumes, and the degenerate no-overexposure setting is
treated as its own configuration rather than as standard uniform-threshold LT.

The practical consequence for the rest of the paper is that this objective must be estimated by
simulating the process.  Sampling-based coverage machinery is unavailable, so the question turns
into how to spend a simulation budget well -- which is the subject of \S\ref{sec:experiments}.
"""


def main() -> int:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    simplex = payload["simplex_values"]
    uniform = payload["uniform_lt_values"]
    edges = payload["instance"]["edges"]

    text = TEMPLATE % {
        "omega": edges["a->t"],
        "gformula": r"2\delta(1-\delta)",
        "d_empty": 0.0,
        "d_single": edges["a->t"],
        "d_pair": edges["a->t"] + edges["b->t"],
        "f_empty": simplex["S = {}"],
        "f_single": simplex["S = {a}"],
        "f_pair": simplex["S = {a,b}"],
        "u_empty": uniform["S = {}"],
        "u_single": uniform["S = {a}"],
        "u_pair": uniform["S = {a,b}"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"  non-monotone under the model's distribution: "
          f"{payload['non_monotone_under_simplex']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
