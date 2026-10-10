"""Resume only missing episodes without altering the frozen experiment code.

The interrupted episode cannot restore continuous-controller state; archive it
and rerun that episode. Completed episodes and all original frozen files remain
unchanged. Downtime is excluded from active execution time.
"""
import gzip
import hashlib
import json
import time
from pathlib import Path

import run as original

ROOT, OUT = original.ROOT, original.OUT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partial_rows(path):
    rows = []
    error = None
    if path.exists():
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    rows.append(json.loads(line))
        except (EOFError, OSError, json.JSONDecodeError) as exc:
            error = str(exc)
    return rows, error


def main():
    assert not (OUT / "finished.json").exists()
    assert original.json.loads((ROOT / "outputs/grid21/audit.json").read_text())["passed"]
    assert original.json.loads((OUT / "preflight.json").read_text())["passed"]
    for name, expected in json.loads((OUT / "code_freeze.json").read_text()).items():
        assert sha(ROOT / name) == expected, name
    results = json.loads((OUT / "manifest.json").read_text())
    expected_order = original.D["execution_order"]
    preserved = {}
    for item, result in zip(expected_order, results):
        assert (item["week"], item["method"]) == (result["week"], result["method"])
        path = OUT / "runs" / (item["week"] + "__" + item["method"])
        assert json.loads((path / "summary.json").read_text()) == result
        assert (path / "vectors.npz").is_file()
        for p in path.rglob("*"):
            if p.is_file():
                preserved[p.relative_to(ROOT).as_posix()] = sha(p)
    assert len(results) == 14, "This recovery is for the documented single interruption."
    pending = expected_order[len(results):]
    interrupted = OUT / "runs" / (pending[0]["week"] + "__" + pending[0]["method"])
    assert interrupted.is_dir() and not (interrupted / "summary.json").exists()
    assert interrupted.resolve().parent == (OUT / "runs").resolve()
    archived = OUT / "interrupted_runs" / (interrupted.name + "__first_attempt")
    assert not archived.exists()
    archived.parent.mkdir(exist_ok=True)
    rows, read_error = partial_rows(interrupted / "steps.jsonl.gz")
    hashes = {p.name: sha(p) for p in interrupted.iterdir() if p.is_file()}
    interrupted.rename(archived)
    evidence = {
        "completed_episodes_preserved": len(results), "completed_file_hashes": preserved,
        "archive": archived.relative_to(ROOT).as_posix(), "archived_file_hashes": hashes,
        "partial_readable_rows": len(rows), "partial_gzip_read_error": read_error,
        "partial_logged_public_forecasts_lower_bound": sum(r["simulations"] for r in rows),
        "partial_logged_controller_s_lower_bound": sum(r["decision_s"] for r in rows),
        "interruption_cause": "Process absent at continuation; no failure.json. Cause not established.",
        "new_policy_or_fits": False,
        "timing_scope": "Each completed episode retains its own measured clock. Aggregate successful-episode time excludes downtime and aborted work. Exact aborted in-flight work is unavailable; not an end-to-end cost claim.",
        "active_wall_cap_reserve_for_interruption_s": original.D["caps"]["episode_s"],
        "physical_cap_reserve_for_interruption": 2017,
    }
    original.write_json(OUT / "resume_evidence.json", evidence)
    original.write_json(OUT / "resume_code_freeze.json", {
        "work/grid22/resume.py": sha(Path(__file__)),
        "outputs/grid22/resume_evidence.json": sha(OUT / "resume_evidence.json"),
    })
    old_wall = sum(s["wall_s"] for s in results)
    original.START = time.perf_counter() - old_wall - original.D["caps"]["episode_s"]
    original.counts.update({
        "physical_steps": sum(s["steps"] for s in results),
        "public_forecasts": sum(s["public_forecasts"] for s in results), "model_fits": 0,
    })
    for item in pending:
        results.append(original.run(item))
        original.write_json(OUT / "manifest.json", results)
        assert original.counts["physical_steps"] + 2017 <= original.D["caps"]["physical_steps"]
    for name, expected in preserved.items():
        assert sha(ROOT / name) == expected, name
    original.write_json(OUT / "finished.json", {
        "passed_engineering": True, "summaries": results, "counts": original.counts,
        "wall_s": sum(s["wall_s"] for s in results),
        "resumed": True, "interrupted_work_separately_archived": True,
        "wall_scope": "Sum of successful episode walltimes; excludes downtime and interrupted work.",
    })
    print(json.dumps({"complete": True, "runs": len(results), "counts": original.counts}), flush=True)


if __name__ == "__main__":
    main()
