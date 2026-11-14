"""Build the submission evidence tables from the two published paired readouts."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOLS = HERE.parent
OUT = HERE.parents[5] / "outputs/grid337_native_last_batch"
ARMS = (
    ("GRID330", "GRID330_remaining19", "BOUNDED261_GUARD_DC", "FRAME_GUARD"),
    ("GRID330", "GRID330_remaining19", "BOUNDED261_GUARD_DC", "PARENT208_GUARD_DC"),
    ("GRID334", "GRID334_logeipc_closed_loop", "LOGEIPC_DC", "BOUNDED261_GUARD_DC"),
)
METRICS = ("steps", "complete", "illegal", "ambiguous", "exception_steps", "cost",
           "candidate_queries", "total_forecasts", "controller_s", "all_p95", "all_p99",
           "search_p95", "search_p99")


def load(path):
    return json.loads(path.read_bytes())


def main():
    rows = []
    for source, folder, arm, reference in ARMS:
        results = PROTOCOLS / folder / "results"
        path = results / "paired_results.json"
        data = path.read_bytes()
        expected = "826a34ca7444edbb0d75ec45d2dab45ee514cbe89966b4e15906bba8e5c12123" if source == "GRID330" else load(
            results / "delivery_hashes.json")["paired_results.json"]
        assert hashlib.sha256(data).hexdigest() == expected
        doc = json.loads(data)
        records = [*doc["records"]]
        if source == "GRID330":
            records.append(doc["historical_baseline"])
        index = {(r["week"], r["arm"]):r for r in records}
        assert len(index) == len(records)
        matching = [p for p in doc["pairs"] if p["arm"] == arm and p["reference"] == reference]
        assert len(matching) == 4
        for pair in matching:
            week = pair["week"]
            a, b = index[(week, arm)], index[(week, reference)]
            assert pair["same_horizon"] == (a["steps"] == b["steps"] and a["complete"] == b["complete"])
            assert pair["complete_both"] == (a["complete"] and b["complete"])
            fresh = pair.get("fresh_timing", True)
            row = dict(source=source, date=week[:10], arm=arm, reference=reference,
                       matched_horizon=pair["same_horizon"], same_trajectory=pair["same_trajectory"],
                       both_complete=pair["complete_both"], fresh_timing=fresh,
                       candidate_delta=pair["candidate_delta"], full_week_cost_delta_pct=pair["cost_delta_pct"],
                       controller_delta_pct=pair["controller_s_change_pct"],
                       first_difference_step=(pair.get("first_difference") or {}).get("step"))
            if not fresh:
                assert row["controller_delta_pct"] is None
            for label, rec in (("arm", a), ("reference", b)):
                for metric in METRICS:
                    row[f"{label}_{metric}"] = None if metric == "cost" and not rec["complete"] else rec[metric]
            rows.append(row)
    assert len(rows) == 12
    with (OUT / "paired_main_table.csv").open("x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

