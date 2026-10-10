"""Two independent quality-only workers, one atomic checkpoint writer.

Frozen scientific settings unchanged. Concurrent wall times cannot certify
controller speed. Each rollout process has its own environment and seed.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid24"


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    d = json.loads((OUT / "design.json").read_text())
    assert json.loads((OUT / "pilot_audit.json").read_text())["passed"]
    assert json.loads((OUT / "pilot_review.json").read_text())["pilot_gate_passed"]
    for freeze in ["code_freeze.json", "parallel_code_freeze.json"]:
        for name, sha in json.loads((OUT / freeze).read_text()).items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    if args.worker is not None:
        item = d["execution_order"][args.worker]
        assert item["phase"] == "remaining"
        sys.path.insert(0, str(ROOT / "work/grid24"))
        import run as runner
        summary = runner.run(item)
        print(json.dumps({"worker_completed": summary["method"], "week": summary["week"]}), flush=True)
        return

    started = time.perf_counter()
    results = json.loads((OUT / "manifest.json").read_text())
    completed = {(s["week"], s["method"]) for s in results}
    pending = [(i, x) for i, x in enumerate(d["execution_order"])
               if x["phase"] == "remaining" and (x["week"], x["method"]) not in completed]
    active = []
    save(OUT / "parallel_execution.json", {"workers": 2, "timing_scope": "Concurrent quality screening only. No independent speed claim.",
         "scientific_design_unchanged": True, "planned_remaining": len(pending)})
    (OUT / "RUN_STATE.md").write_text("RUNNING. Pilot passed; two independent quality-only workers. No training or reserved-family evaluation.\n", encoding="utf-8")
    try:
        while pending or active:
            assert time.perf_counter() - started < d["caps"]["wall_s"]
            while pending and len(active) < 2:
                index, item = pending.pop(0)
                log = (OUT / f'worker_{index:02d}.log').open("w", encoding="utf-8")
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(index)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                active.append((process, log, index, item))
            done = []
            for entry in active:
                process, log, index, item = entry
                code = process.poll()
                if code is None: continue
                log.close()
                assert code == 0, (index, item, code)
                path = OUT / "runs" / (item["week"] + "__" + item["method"]) / "summary.json"
                s = json.loads(path.read_text())
                assert s["method"] == item["method"] and s["week"] == item["week"]
                results.append(s)
                save(OUT / "manifest.json", results)
                totals = {"physical_steps": sum(r["steps"] for r in results),
                          "public_forecasts": sum(r["public_forecasts"] for r in results), "model_fits": 0}
                assert totals["physical_steps"] <= d["caps"]["physical_steps"]
                assert totals["public_forecasts"] <= d["caps"]["public_forecasts"]
                print(json.dumps({"completed_runs": len(results), "planned_runs": len(d["execution_order"]),
                    "week": s["week"], "method": s["method"], "complete": s["complete"], "cost": s["cost"]}), flush=True)
                done.append(entry)
            active = [entry for entry in active if entry not in done]
            if pending or active: time.sleep(1)
        assert len(results) == len(d["execution_order"])
        save(OUT / "finished.json", {"passed_engineering": True, "summaries": results, "counts": totals,
            "wall_s_this_process": time.perf_counter()-started, "wall_s_sum_runs": sum(s["wall_s"] for s in results),
            "concurrent_remaining_workers": 2, "pilot_workers": 1})
        print(json.dumps({"complete": True, "runs": len(results), "counts": totals}), flush=True)
    finally:
        for process, log, _, _ in active:
            if process.poll() is None: process.terminate(); process.wait()
            log.close()


if __name__ == "__main__": main()
