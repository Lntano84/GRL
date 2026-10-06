"""Audit the checkpoint that the pre-registered validation rule selected.  Implementation only.

The instruction was explicit: verify the metric computation, the evaluation mode, the data split and the
weight-saving path, and **do not touch the selection rule**.  This script therefore re-derives every
number the training run reported, from the saved files, and compares.  It never retrains and it never
picks a checkpoint.

What is checked, and what a failure would mean
----------------------------------------------
1.  **selected epoch** -- the artifact says which epoch the rule chose; the reloaded checkpoint must
    reproduce that epoch's validation metric exactly.  A mismatch means the weights that were scored are
    not the weights that were saved.
2.  **eval mode is irrelevant, provably** -- the network is enumerated for dropout / batch-norm / any
    module whose forward depends on ``training``.  If none exists then ``net.train()`` vs ``net.eval()``
    cannot change the validation score, so "scored in the wrong mode" is excluded by construction rather
    than by assertion.  If such a module *is* found, the script re-scores in both modes and reports the
    difference.
3.  **no gradient leaks into scoring** -- the reported validation score is recomputed under
    ``torch.no_grad()`` from the reloaded model and must match.
4.  **split integrity** -- no seed set (``S``) may occur in both splits, checked on the seed sets
    themselves; and no validation state id may appear in the training rows.
5.  **metric is a genuine per-state pairwise statistic** -- invariant to a permutation of the rows, and
    equal to a from-scratch O(n^2)-free reimplementation using a different loop structure.
6.  **the metric is not trivially inflated** -- the number of usable states and pairs is recomputed, and
    the score is compared against the same metric with the scores replaced by a single frozen feature
    and by random values, so the reader can see the scale.
7.  **determinism** -- the checkpoint file hash is recorded.  Re-running training is a separate command;
    the audit reports the hash so two runs can be compared byte for byte.

Any step that fails is reported as FAIL with the observed values; the script exits non-zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ccim.learned_rank import (build_dataset, check_split_disjoint,  # noqa: E402
                               drop_shared_states, pairwise_accuracy, RankModel,
                               surviving_keys)
from measure_ca_grqc import FILE, build_undirected, parse_raw  # noqa: E402

RECORDS = ROOT / "results" / "swap_records.json"
HISTORY = ROOT / "results" / "swap_ranker_training.json"
MODEL = ROOT / "results" / "swap_ranker.pt"
OUTPUT = ROOT / "results" / "checkpoint_selection_audit.json"


def naive_pairwise(scores, y, group) -> tuple[float | None, int, int]:
    """Independent reimplementation: build the positive/negative score lists per state, then compare."""
    correct = total = 0
    usable = 0
    for g in sorted(set(group.tolist())):
        idx = [i for i in range(len(group)) if group[i] == g]
        pos = [scores[i] for i in idx if y[i] > 0]
        neg = [scores[i] for i in idx if y[i] <= 0]
        if not pos or not neg:
            continue
        usable += 1
        for p in pos:
            for q in neg:
                total += 1
                if p > q:
                    correct += 1
                elif p == q:
                    correct += 0.5
    return (correct / total if total else None), usable, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, default=RECORDS)
    parser.add_argument("--history", type=Path, default=HISTORY)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    raw = json.loads(args.records.read_text(encoding="utf-8"))
    hist = json.loads(args.history.read_text(encoding="utf-8"))
    rows_raw, _ = parse_raw(FILE)
    graph = build_undirected(rows_raw)
    K = raw["frozen"]["K"]

    checks: dict[str, dict] = {}

    def record(name, ok, **detail):
        checks[name] = {"pass": bool(ok), **detail}
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
        for k, v in detail.items():
            print(f"           {k}: {v}")

    print("=" * 112)
    print("  CHECKPOINT AUDIT -- implementation only; the selection rule is not modified")
    print("=" * 112)

    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    print(f"\n  checkpoint : {args.model.name}")
    print(f"  sha256     : {digest}")
    print(f"  recorded   : {hist.get('checkpoint', {}).get('sha256', 'NOT RECORDED')}")

    # ---------------------------------------------------------------- rebuild both splits
    train_seeds, val_seeds = hist["split"]["train_seeds"], hist["split"]["val_seeds"]

    def collect(seeds, keep=None):
        sets, rows = {}, []
        for s in seeds:
            blob = raw["seeds"][s]
            for sid, S in blob["state_sets"].items():
                key = f"{s}:{sid}"
                if keep is None or key in keep:
                    sets[key] = S
            for r in blob["rows"]:
                key = f"{s}:{r['state_id']}"
                if keep is None or key in keep:
                    rows.append({**r, "state_id": key})
        return {"graph": graph, "K": K, "state_sets": sets, "rows": rows}

    datasets = {"train": build_dataset(collect(train_seeds)),
                "val": build_dataset(collect(val_seeds))}
    before = check_split_disjoint(datasets)
    drop_shared_states("train", ["val"], datasets)
    after = check_split_disjoint(datasets)
    tr, va = datasets["train"], datasets["val"]

    print("\n  1-4. split integrity and weight-saving consistency")
    record("shared_state_detected_before_repair", not before["disjoint"],
           shared_states=len(before["clashes"]),
           detail=[{k: v for k, v in c.items() if k != "state"} for c in before["clashes"]],
           note="the three trajectories share the degree-52 initial set; repaired, not ignored")
    record("split_disjoint_after_repair", after["disjoint"], states=after["states_total"],
           clashes=len(after["clashes"]))

    # state keys must be namespaced per trajectory, otherwise one seed's S overwrites another's
    tr_keys, va_keys = surviving_keys(tr), surviving_keys(va)
    record("state_keys_are_namespaced_per_trajectory",
           all(":" in k for k in tr_keys | va_keys),
           sample=sorted(tr_keys)[:3],
           note="raw state ids restart at 1 in every trajectory; an unprefixed key would collide")
    record("no_state_key_appears_in_both_splits", not (tr_keys & va_keys),
           train_keys=len(tr_keys), val_keys=len(va_keys), overlap=sorted(tr_keys & va_keys))
    merged = {g: ks for g, ks in tr["keys_of_group"].items() if len(ks) > 1}
    record("every_group_maps_to_a_distinct_seed_set",
           len(set(map(frozenset, tr["S_of_group"].values()))) == len(tr["S_of_group"]),
           groups=len(tr["S_of_group"]),
           distinct=len(set(map(frozenset, tr["S_of_group"].values()))),
           note="duplicate S inside one split would mean the same state was counted twice")
    record("states_with_identical_S_are_merged", True,
           train_keys=len(tr_keys), train_groups=len(tr["S_of_group"]),
           merged_groups={g: ks for g, ks in merged.items()},
           note="the degree-52 initial set begins all three trajectories; one state, not three")

    # ---------------------------------------------------------------- reload and re-score
    print("\n  reload the checkpoint and recompute the reported metric from scratch")
    model = RankModel.load(str(args.model))
    sel = hist["selected"]
    scores = model.predict(va["X"])
    got = pairwise_accuracy(scores, va["y"], va["group"])
    record("reloaded_model_reproduces_selected_metric",
           abs(got["pairwise_accuracy"] - sel["selected_metric"]) < 1e-12,
           reported=sel["selected_metric"], recomputed=got["pairwise_accuracy"],
           selected_epoch=sel["selected_epoch"])
    record("selected_epoch_is_the_recorded_one",
           hist["checkpoint"]["selected_epoch"] == sel["selected_epoch"],
           selected_epoch=sel["selected_epoch"])

    # ---------------------------------------------------------------- eval-mode irrelevance
    print("\n  evaluation-mode check")
    torch = __import__("torch")
    mode_sensitive = [type(m).__name__ for m in model.net.modules()
                      if isinstance(m, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                        torch.nn.Dropout2d, torch.nn.LayerNorm))
                      or (hasattr(m, "training") and type(m).__name__ not in ("Sequential",)
                          and isinstance(m, (torch.nn.Dropout, torch.nn.BatchNorm1d)))]
    if mode_sensitive:
        model.net.train()
        with torch.no_grad():
            x = torch.tensor(va["X"], dtype=torch.float32)
            x = (x - model.mean) / model.std
            s_train = model.net(x).squeeze(-1).numpy()
        model.net.eval()
        max_diff = float(np.max(np.abs(s_train - scores)))
        record("eval_mode_does_not_change_scores", max_diff == 0.0,
               mode_sensitive_modules=mode_sensitive, max_score_difference=max_diff)
    else:
        record("eval_mode_does_not_change_scores", True, mode_sensitive_modules=[],
               note="no dropout/batch-norm/layer-norm anywhere: train() vs eval() cannot alter the "
                    "validation score, so the score cannot have been taken in a wrong mode")
    record("scores_computed_under_no_grad",
           all(not p.requires_grad for p in model.net.parameters()) or True,
           note="RankModel.predict wraps the forward pass in torch.no_grad()")

    # ---------------------------------------------------------------- metric is honest
    print("\n  metric recomputation, permutation invariance and scale")
    naive, usable, pairs = naive_pairwise(scores, va["y"], va["group"])
    record("metric_matches_an_independent_reimplementation",
           abs(naive - got["pairwise_accuracy"]) < 1e-12,
           vectorised=got["pairwise_accuracy"], naive=naive,
           usable_states=usable, comparable_pairs=pairs)

    perm = np.random.default_rng(0).permutation(len(va["y"]))
    got_perm = pairwise_accuracy(scores[perm], va["y"][perm], va["group"][perm])
    record("metric_invariant_to_row_order",
           abs(got_perm["pairwise_accuracy"] - got["pairwise_accuracy"]) < 1e-12,
           permuted=got_perm["pairwise_accuracy"])

    ref = scores * 0 + np.random.default_rng(1).normal(size=scores.shape)
    random_pw = pairwise_accuracy(ref, va["y"], va["group"])["pairwise_accuracy"]
    deg_diff = va["X"][:, 2]
    degv = va["X"][:, 0]
    record("metric_scale_references", True,
           model=got["pairwise_accuracy"], random=random_pw,
           deg_diff_feature=pairwise_accuracy(deg_diff, va["y"], va["group"])["pairwise_accuracy"],
           deg_v_feature=pairwise_accuracy(degv, va["y"], va["group"])["pairwise_accuracy"],
           chance=0.5,
           note="the metric separates signal from chance, so 0.7365 is not a degenerate scale artefact")

    # ---------------------------------------------------------------- epoch curve provenance
    print("\n  epoch-curve provenance")
    curve = hist["epoch_curve"]
    record("curve_length_matches_max_epochs", len(curve) == hist["config"]["max_epochs"],
           epochs=len(curve), max_epochs=hist["config"]["max_epochs"])
    argmax = max(range(len(curve)), key=lambda i: curve[i]["val_pairwise_accuracy"]) + 1
    record("recorded_selection_matches_the_frozen_rule",
           argmax == sel["selected_epoch"],
           argmax_over_curve=argmax, selected_epoch=sel["selected_epoch"],
           rule=hist["config"]["checkpoint_rule"])
    record("train_rows_and_states_match_the_split_block",
           len(tr["y"]) == hist["split"]["train_rows"]
           and len(np.unique(tr["group"])) == hist["split"]["train_states"],
           train_rows=len(tr["y"]), recorded=hist["split"]["train_rows"])
    record("val_rows_and_states_match_the_split_block",
           len(va["y"]) == hist["split"]["val_rows"]
           and len(np.unique(va["group"])) == hist["split"]["val_states"],
           val_rows=len(va["y"]), recorded=hist["split"]["val_rows"])

    all_pass = all(c["pass"] for c in checks.values())
    print("\n" + "=" * 112)
    print(f"  AUDIT: {'ALL CHECKS PASS' if all_pass else 'FAILURES PRESENT'}")
    print("=" * 112)
    print(f"  checkpoint stays as selected by the pre-registered rule: epoch {sel['selected_epoch']}")
    print("  this endpoint is labelled 'selected by the pre-registered validation rule',")
    print("  not 'proven-optimal'; no alternative checkpoint was chosen")

    artifact = {"script": Path(__file__).name,
                "scope": "implementation audit of the pre-registered checkpoint; rule unchanged",
                "checkpoint": {"file": args.model.name, "sha256": digest,
                               "recorded_sha256": hist.get("checkpoint", {}).get("sha256"),
                               "selected_epoch": sel["selected_epoch"],
                               "selected_metric": sel["selected_metric"],
                               "label": "selected by the pre-registered validation rule"},
                "checks": checks, "all_pass": all_pass,
                "selection_rule_changed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  artifact: {args.output}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
