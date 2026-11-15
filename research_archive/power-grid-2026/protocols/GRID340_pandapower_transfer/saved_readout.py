"""Summarize the GRID340 saved prefix without launching any AC solve."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from pilot import OUT, load_policy, summarize

def main():
    policy, _ = load_policy()
    groups = defaultdict(list)
    for line in (OUT / "observations.jsonl").read_text(encoding="utf8").splitlines():
        row = json.loads(line)
        groups[row["case"], row["scale"]].append(row)
    results = []
    for (case, scale), rows in groups.items():
        base, *feedback = rows
        assert base["pool_id"] is None
        assert [row["pool_id"] for row in feedback] == list(range(len(feedback)))
        library = json.loads((OUT / f"{case}_candidates.json").read_bytes())
        assert len(feedback) == len(library)
        arms = summarize(policy, library, feedback)
        finite = [row["rho"] for row in feedback if row["rho"] is not None and row["voltage_ok"]]
        delta = None if not finite else base["rho"] - min(finite)
        statuses = Counter(
            "nonconverged" if not r["converged"] else
            "voltage_failed" if not r["voltage_ok"] else
            "admissible" if r["strict_admissible"] else "other_converged"
            for r in feedback
        )
        results.append(dict(case=case, scale=scale, baseline=base, difference_from_best=delta,
                            statuses=statuses, arms=arms))
    assert [(r["case"], r["scale"]) for r in results] == [
        ("ieee118", 1.0), ("ieee118", 1.05), ("ieee300", 1.0),
    ]
    log = (OUT / "pilot.log").read_text(encoding="utf8")
    assert "AssertionError: ('ieee300', 1.05, {'converged': False" in log
    result = dict(completed=results,
                  failed_baseline=dict(case="ieee300", scale=1.05, converged=False,
                                       candidate_solves=0, evidence="pilot.log"),
                  saved_solve_rows=sum(map(len, groups.values())),
                  attempted_solves=sum(map(len, groups.values())) + 1,
                  new_solves_in_readout=0, grid2op_steps=0, new_fits=0)
    with (OUT / "saved_readout.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    for run in results:
        print(json.dumps({**run, "arms": {
            name: {k: v for k, v in r.items() if k != "visited"} for name, r in run["arms"].items()}}))
    print(f"solves={result['attempted_solves']}")

if __name__ == "__main__":
    main()
