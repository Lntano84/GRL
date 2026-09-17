"""Gate-3c: ablation -- do the threshold features add anything beyond the analytic score?

Gate 3b found that neither ridge nor an MLP beat ``delta2`` on held-out contexts (regret
0.62 and 0.97 against 0.50).  Two explanations are possible and they call for opposite
responses:

1. **Not enough data.**  480 records over 12 contexts is small; more data might close the gap.
2. **No incremental signal.**  The analytic score already summarises the local exposure
   structure, and the threshold features are collinear with it.

Pooled correlations cannot separate these, because the dataset mixes regimes whose label
scales differ by three orders of magnitude (about +200 when unsaturated, about 0 when
saturated), so *every* informative feature correlates with the label at |r| ~ 0.9 simply by
identifying the regime.

This script runs the ablation that does separate them.  For each feature subset and each
held-out context it fits the model on the other contexts and scores the held-out one, so the
number is always a **within-regime** generalisation score:

``analytic_only``     the single analytic_score column.  This is delta2 as a regressor.
``analytic+threshold`` analytic plus the threshold-derived columns.
``threshold_only``    threshold columns with the analytic score removed.
``delta_only``        neighbourhood delta columns, no thresholds, no analytic score.
``all``               everything (what Gate 3b used).

If ``analytic+threshold`` does not improve on ``analytic_only``, the threshold features carry
no incremental information at this sample size.  If ``threshold_only`` is far worse, the
analytic score is doing the real work.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts" / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from evaluate_overexposure_pool_ranking import spearman  # noqa: E402
from train_overexposure_ranker import (  # noqa: E402
    regret,
    ridge_fit,
    ridge_predict,
    standardise,
    top_k_recall,
    train_mlp,
)

THRESHOLD_FEATURES = [
    "frac_overexposed",
    "frac_untouched",
    "frac_in_band",
    "mean_margin_to_tau",
    "mean_margin_to_kappa",
]
DELTA_FEATURES = [
    "delta_self",
    "nb_delta_mean",
    "nb_delta_std",
    "nb_delta_min",
    "nb_delta_max",
]
STRUCTURAL_FEATURES = ["out_degree", "in_degree", "out_weight_sum"]

SUBSETS = {
    "analytic_only": ["analytic_score"],
    "analytic+threshold": ["analytic_score", *THRESHOLD_FEATURES],
    "threshold_only": list(THRESHOLD_FEATURES),
    "delta_only": list(DELTA_FEATURES),
    "analytic+structural": ["analytic_score", *STRUCTURAL_FEATURES],
    "analytic+structural+threshold": ["analytic_score", *STRUCTURAL_FEATURES,
                                      *THRESHOLD_FEATURES],
    "structural_only": list(STRUCTURAL_FEATURES),
    "all": None,  # everything
}


def subset_indices(names: list[str], wanted: list[str] | None) -> list[int]:
    if wanted is None:
        return list(range(len(names)))
    return [names.index(w) for w in wanted]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    names: list[str] = payload["feature_names"]
    records: list[dict] = payload["records"]

    contexts: dict[tuple[float, int], list[dict]] = {}
    for record in records:
        contexts.setdefault((record["seed_fraction"], record["context"]), []).append(record)
    keys = sorted(contexts)

    print(f"dataset={args.dataset.name}  records={len(records)}  contexts={len(keys)}")
    print(f"top-{args.top_k}, leave-one-context-out")
    print()

    summary: dict[str, dict[str, list[float]]] = {}
    for label, wanted in SUBSETS.items():
        columns = subset_indices(names, wanted)
        stat: dict[str, list[float]] = {"spearman": [], "recall": [], "regret": []}
        for held_out in keys:
            train = [r for k, rows_ in contexts.items() if k != held_out for r in rows_]
            test = contexts[held_out]
            if len(test) < 3 or not train:
                continue
            x_train = np.array([[r["features"][i] for i in columns] for r in train], dtype=float)
            y_train = np.array([r["label"] for r in train], dtype=float)
            x_test = np.array([[r["features"][i] for i in columns] for r in test], dtype=float)
            y_test = np.array([r["label"] for r in test], dtype=float)
            x_train_s, x_test_s = standardise(x_train, x_test)

            scores = ridge_predict(ridge_fit(x_train_s, y_train, args.alpha), x_test_s)
            stat["spearman"].append(spearman(list(scores), list(y_test)))
            stat["recall"].append(top_k_recall(y_test, scores, args.top_k))
            stat["regret"].append(regret(y_test, scores, args.top_k))
        summary[label] = stat

    print("=" * 96)
    print("ABLATION (ridge regression, held-out contexts)")
    print("=" * 96)
    print(f"  {'feature subset':<24}{'#feat':>7}{'spearman':>11}{'recall':>9}{'regret':>9}"
          f"{'vs analytic_only':>18}")
    base_regret = None
    for label, wanted in SUBSETS.items():
        n = len(names) if wanted is None else len(wanted)
        block = summary[label]
        rho = statistics.fmean(block["spearman"])
        rec = statistics.fmean(block["recall"])
        reg = statistics.fmean([r for r in block["regret"] if r == r])
        if label == "analytic_only":
            base_regret = reg
            delta = "-"
        else:
            delta = f"{reg - base_regret:+.4f}"
        print(f"  {label:<24}{n:>7}{rho:>+11.4f}{rec:>9.4f}{reg:>9.4f}{delta:>18}")

    print()
    print("VERDICT")
    print("-" * 96)
    an = summary["analytic_only"]["regret"]
    for label in ("analytic+threshold", "threshold_only", "delta_only",
                  "analytic+structural", "analytic+structural+threshold",
                  "structural_only", "all"):
        block = summary[label]["regret"]
        wins = sum(1 for a, b in zip(block, an) if a < b)
        delta = statistics.fmean(block) - statistics.fmean(an)
        n_contexts = len(block)
        # A subset only counts as an improvement if it wins on a clear majority of the
        # held-out contexts, not merely on the mean -- with 12 contexts a mean shift driven
        # by one or two outliers is not evidence.
        flag = "IMPROVES" if (delta < 0 and wins > n_contexts * 0.6) else ""
        print(f"  {label:<30} regret {statistics.fmean(block):.4f}  "
              f"({delta:+.4f})   wins {wins}/{n_contexts}  {flag}")
    print()

    add = statistics.fmean(summary["analytic+threshold"]["regret"]) - statistics.fmean(an)
    if add < -0.01:
        print("  -> threshold features DO add incremental signal beyond the analytic score.")
    else:
        print("  -> threshold features add NO measurable incremental signal beyond the")
        print("     analytic score.  The pooled correlations in Gate 3b were regime")
        print("     identification, not incremental information.")

    best = None
    for label in ("analytic+structural", "analytic+structural+threshold"):
        block = summary[label]["regret"]
        wins = sum(1 for a, b in zip(block, an) if a < b)
        if statistics.fmean(block) < statistics.fmean(an) and wins > len(block) * 0.6:
            best = label
    if best:
        print(f"  -> '{best}' improves on the analytic score alone AND wins on a majority")
        print("     of held-out contexts.  That is the one candidate worth pursuing further.")
    else:
        print("  -> no feature subset improves on the analytic score on a majority of")
        print("     held-out contexts.")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "dataset": str(args.dataset),
            "top_k": args.top_k,
            "summary": {
                label: {
                    "n_features": len(names) if wanted is None else len(wanted),
                    "spearman": statistics.fmean(block["spearman"]),
                    "recall": statistics.fmean(block["recall"]),
                    "regret": statistics.fmean([r for r in block["regret"] if r == r]),
                }
                for label, wanted in SUBSETS.items()
                for block in [summary[label]]
            },
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
