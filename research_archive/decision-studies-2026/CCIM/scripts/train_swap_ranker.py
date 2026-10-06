"""Train the one frozen swap ranker.  2 trajectories train, 1 validate, no sweep, no online result seen.

Order of operations, and why it matters
---------------------------------------
This script is run and its output written to disk **before** any online three-arm comparison exists.  The
checkpoint is chosen on the validation trajectory only, by the metric frozen in
:class:`ccim.learned_rank.FrozenConfig`, and the full epoch curve is stored so the choice can be audited.
No online number feeds back into the model, the features, or the epoch.

The split, and the clash that has to be repaired
------------------------------------------------
Train = replayed seeds 0 and 1; validate = replayed seed 2.  All three trajectories begin from the same
degree-52 initial set, so that state occurs on both sides.  A state may not cross the split, so it is
removed from the validation side and the removal is reported with its exact row and positive counts --
the split is repaired, not merely asserted to be clean.

Reference points, reported at the same time
-------------------------------------------
The offline section prints the within-state pairwise accuracy of the frozen model next to the two
single-feature references measured in ``results/swap_records_inspection.json``: ``deg_diff`` (the simple
structural baseline's own ordering) and ``f1_delta``.  ``deg_diff`` reaches 0.64-0.73 on its own, so the
honest question is not "does the model beat chance" but "does it beat the free structural ordering" --
and a model that merely re-learns ``deg_diff`` has added nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ccim.learned_rank import (CONFIG, build_dataset, check_split_disjoint,  # noqa: E402
                               drop_shared_states, pairwise_accuracy, save_history,
                               surviving_keys, train)
from ccim.swap_features import FEATURE_NAMES  # noqa: E402
from measure_ca_grqc import FILE, build_undirected, parse_raw  # noqa: E402

RECORDS = ROOT / "results" / "swap_records.json"
MODEL_OUT = ROOT / "results" / "swap_ranker.pt"
HISTORY_OUT = ROOT / "results" / "swap_ranker_training.json"


def collect(raw: dict, seeds: list[str], graph, keep: set | None = None) -> dict:
    """Raw records for ``seeds``, optionally restricted to a set of surviving namespaced state keys.

    State ids are namespaced ``"<seed>:<state_id>"`` because every trajectory numbers its states from 1;
    without the prefix, one seed's state sets silently overwrite another's and rows get scored against
    the wrong ``S``.
    """
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
    return {"graph": graph, "K": raw["frozen"]["K"], "state_sets": sets, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, default=RECORDS)
    parser.add_argument("--train-seeds", nargs="+", default=["0", "1"])
    parser.add_argument("--val-seeds", nargs="+", default=["2"])
    parser.add_argument("--model-out", type=Path, default=MODEL_OUT)
    parser.add_argument("--history-out", type=Path, default=HISTORY_OUT)
    args = parser.parse_args()

    raw = json.loads(args.records.read_text(encoding="utf-8"))
    rows_raw, _ = parse_raw(FILE)
    graph = build_undirected(rows_raw)

    print("=" * 112)
    print("  TRAINING THE FROZEN SWAP RANKER")
    print("=" * 112)
    print(f"    records        : {args.records.name}"
          f"   (graph sha256 {raw['frozen']['file_sha256'][:16]}...)")
    print(f"    train seeds    : {args.train_seeds}      validate seeds: {args.val_seeds}")
    print(f"    features       : {len(FEATURE_NAMES)}  {list(FEATURE_NAMES)}")
    print(f"    hidden / loss  : {CONFIG.hidden} / {CONFIG.loss}")
    print(f"    optimiser      : {CONFIG.optimizer} lr={CONFIG.learning_rate} "
          f"batch={CONFIG.batch_size} max_epochs={CONFIG.max_epochs}")
    print(f"    checkpoint     : max {CONFIG.checkpoint_metric} ({CONFIG.checkpoint_rule})")

    t0 = time.perf_counter()
    datasets = {"train": build_dataset(collect(raw, args.train_seeds, graph)),
                "val": build_dataset(collect(raw, args.val_seeds, graph))}
    build_seconds = time.perf_counter() - t0

    before = check_split_disjoint(datasets)
    print(f"\n    split check before repair : "
          f"{'disjoint' if before['disjoint'] else str(len(before['clashes'])) + ' shared state(s)'}")
    removed = drop_shared_states("train", ["val"], datasets)
    after = check_split_disjoint(datasets)
    for name, info in removed.items():
        if info["states_dropped"]:
            print(f"      removed from {name}: {info['states_dropped']} state(s), "
                  f"{info['rows_dropped']} rows, {info['positives_dropped']} improving row(s) "
                  f"(shared seed-set size(s) {info['shared_state_sizes']})")
    print(f"    split check after repair  : {'disjoint' if after['disjoint'] else 'STILL SHARED'}")
    if not after["disjoint"]:
        print("  refusing to train on a leaking split")
        return 1

    tr, va = datasets["train"], datasets["val"]
    print(f"    train          : {len(tr['y']):,} rows / {len(np.unique(tr['group']))} states"
          f"   positives {int((tr['y'] > 0).sum())} ({100 * (tr['y'] > 0).mean():.2f}%)")
    print(f"    validate       : {len(va['y']):,} rows / {len(np.unique(va['group']))} states"
          f"   positives {int((va['y'] > 0).sum())} ({100 * (va['y'] > 0).mean():.2f}%)")
    print(f"    dataset build  : {build_seconds:.2f} s")

    tr_keep, va_keep = surviving_keys(tr), surviving_keys(va)
    print("\n    training (validation is used only for the pre-agreed checkpoint choice)")
    t1 = time.perf_counter()
    model, history = train(collect(raw, args.train_seeds, graph, tr_keep),
                           collect(raw, args.val_seeds, graph, va_keep), verbose=True)
    train_seconds = time.perf_counter() - t1

    sel = history["selected"]
    print(f"    selected epoch : {sel['selected_epoch']} of {CONFIG.max_epochs}"
          f"   val pairwise accuracy {sel['selected_metric']:.4f}")
    print(f"    training time  : {train_seconds:.2f} s   (label generation is reported separately)")

    ref = json.loads((ROOT / "results" / "swap_records_inspection.json").read_text(encoding="utf-8"))
    val_ref = ref["per_seed"][args.val_seeds[0]]["feature_pairwise_accuracy_alone"]
    pw_model = pairwise_accuracy(model.predict(va["X"]), va["y"], va["group"])
    print(f"\n    offline, on the validation trajectory only ({len(va['y']):,} rows, "
          f"{pw_model['states_usable']} usable states):")
    print(f"      {'ordering source':<36} {'within-state pairwise':>22}")
    print(f"      {'frozen model (selected epoch)':<36} {pw_model['pairwise_accuracy']:>22.4f}")
    for name in ("deg_diff", "deg_v", "deg_u", "f1_delta", "one_round_gain_v", "one_round_loss_u"):
        print(f"      {name + ' (single feature)':<36} {val_ref[name]:>22.4f}")
    print(f"      {'random within batch':<36} {0.5:>22.4f}")

    artifact = {
        "script": Path(__file__).name,
        "scope": "one frozen configuration; trained and checkpointed before any online comparison exists",
        "records_file": args.records.name,
        "frozen": raw["frozen"],
        "split": {"train_seeds": args.train_seeds, "val_seeds": args.val_seeds,
                  "shared_before_repair": before, "removed": removed, "after_repair": after,
                  "train_rows": int(len(tr["y"])), "train_states": int(len(np.unique(tr["group"]))),
                  "val_rows": int(len(va["y"])), "val_states": int(len(np.unique(va["group"]))),
                  "train_positives": int((tr["y"] > 0).sum()),
                  "val_positives": int((va["y"] > 0).sum())},
        "config": history["config"],
        "selected": sel,
        "epoch_curve": history["history"],
        "offline_validation": {
            "model_pairwise_accuracy": pw_model["pairwise_accuracy"],
            "model_states_usable": pw_model["states_usable"],
            "single_feature_pairwise_accuracy": {k: val_ref[k] for k in
                                                 ("deg_diff", "deg_v", "deg_u", "f1_delta",
                                                  "one_round_gain_v", "one_round_loss_u",
                                                  "active_neighbours_v", "active_neighbours_u")},
            "random_reference": 0.5,
            "note": ("offline accuracy on the visited-candidate distribution is not evidence that online "
                     "search improves; the replayed rows stop at the first improvement, so they are a "
                     "biased sample of the pool"),
        },
        "cost": {"dataset_build_seconds": build_seconds, "training_seconds": train_seconds,
                 "label_generation_seconds": raw["label_generation"]["replay_seconds_total"],
                 "note": ("label generation, training and model loading are listed separately from the "
                          "online search timing and are never treated as free")},
    }
    model.save(str(args.model_out))
    digest = hashlib.sha256(Path(args.model_out).read_bytes()).hexdigest()
    artifact["checkpoint"] = {
        "file": args.model_out.name,
        "sha256": digest,
        "selected_epoch": sel["selected_epoch"],
        "label": ("selected by the pre-registered validation rule (max "
                  "val_pairwise_accuracy, ties earliest epoch) -- this is NOT a proven-optimal model; "
                  "selection optimism is possible and the online comparison is what tests its value"),
    }
    save_history(str(args.history_out), artifact)
    print(f"\n    checkpoint sha256 : {digest}")
    print(f"    label             : selected by the pre-registered validation rule (epoch "
          f"{sel['selected_epoch']}); not a proven-optimal model")
    print(f"    model   : {args.model_out}")
    print(f"    history : {args.history_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
