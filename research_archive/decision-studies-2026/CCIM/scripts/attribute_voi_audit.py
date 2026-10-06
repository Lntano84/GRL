"""Recompute and cross-check the completed attribute VOI experiment from saved rows."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import attribute_voi_probe as P  # noqa: E402

OUT = ROOT / "results" / "attribute_voi"


def rows(split, count):
    out = []
    for idx in range(count):
        entry = json.loads((OUT / split / f"state_{idx:03d}.json").read_text(encoding="utf-8"))
        assert entry["plan_sha256"] == P.digest(P.PLAN)
        out.append(entry["result"])
    return out


def main():
    graph, ids, classes, i, j, same, full, pair, info = P.load_graph()
    dev, confirm = rows("development", P.N_DEV), rows("confirmation", P.N_CONFIRM)
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert [r["state"] for r in dev] == list(range(20))
    assert [r["state"] for r in confirm] == list(range(1000, 1080))
    keys = set()
    for row in dev + confirm:
        queried = np.zeros(len(ids), dtype=bool)
        queried[row["query_order"]] = True
        assert len(row["query_order"]) == 9 and int(queried.sum()) == 9
        mask = P.known_pairs(queried, i, j)
        # Complete visible state consists of surveyed IDs and every confirmed edge.
        key = (tuple(sorted(row["query_order"])), tuple(np.flatnonzero(mask & full)))
        assert key not in keys
        keys.add(key)
        assert len(row["pool"]) == P.POOL
        assert len(set(row["pool"])) == P.POOL
        assert all(not queried[v] for v in row["pool"])
        assert set(row["choices"].values()) <= set(row["pool"])
        for arm1, node1 in row["choices"].items():
            for arm2, node2 in row["choices"].items():
                if node1 == node2:
                    assert row["values"][arm1] == row["values"][arm2]
                    assert row["chosen_seed_sets"][arm1] == row["chosen_seed_sets"][arm2]
    means = {r: float(np.mean([row["values"][r] for row in dev])) for r in P.RULES}
    best = max(P.RULES, key=lambda r: means[r])
    assert best == summary["baseline"] == "observed_degree"
    for arm in ("voi",) + P.RULES:
        mean = float(np.mean([r["values"][arm] for r in confirm]))
        assert abs(mean - summary["confirmation_means"][arm]) < 1e-12
    for arm in P.RULES:
        diff = [r["values"]["voi"] - r["values"][arm] for r in confirm]
        mean = statistics.mean(diff)
        margin = t.ppf(0.975, len(diff) - 1) * statistics.stdev(diff) / len(diff) ** 0.5
        old = summary["paired_contrasts"][arm]
        assert abs(mean - old["mean"]) < 1e-12
        assert abs(mean - margin - old["ci95"][0]) < 1e-12
        assert abs(mean + margin - old["ci95"][1]) < 1e-12
        assert (sum(d > 0 for d in diff), sum(d < 0 for d in diff),
                sum(d == 0 for d in diff)) == (old["wins"], old["losses"], old["ties"])
    assert summary["primary"]["ci95"][1] < summary["delta"]
    print("PASS: 100 distinct visible states, legal candidate pools, shared-action values, "
          "development baseline, all confirmation means and intervals, stopping rule")


if __name__ == "__main__":
    main()
