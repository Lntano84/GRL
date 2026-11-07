"""Seal one development-only closed-loop pilot and exercise synthetic control flow."""
import argparse
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import runtime as rt


def freeze():
    assert not rt.O.exists()
    head = rt.git("rev-parse", "HEAD").decode().strip()
    old = rt.PREVIOUS
    for path in (old / "design.json", old / "runtime.py", old / "source_freeze.json",
                 old / "results/finished.json", old / "results/paired_results.json",
                 old / "results/close_grid312.py"):
        rt.tracked(path, head)
    old_design = rt.read(old / "design.json")
    old_freeze = rt.read(old / "source_freeze.json")
    assert old_design["caps"] == rt.CAPS and len(old_design["execution_order"]) == rt.REPLAYS
    model_file = rt.BASE / "model_references.json"
    assert rt.sha(model_file) == old_freeze[model_file.relative_to(rt.R).as_posix()]
    model, = [r for r in rt.read(model_file) if r["pool"] == "f0_s0"]
    bindings = rt.read(rt.R / "outputs/grid297_paired_collection_preparation/published_model_bindings.json")
    for rel, digest in model["hashes"].items():
        entry = bindings[rel]
        assert rt.sha(rt.R / rel) == digest == old_freeze[rel] == entry["sha256"]
        assert hashlib.sha256(rt.git("show", head + ":" + entry["archive_path"])).hexdigest() == digest
    history = rt.read(old / "results/finished.json")
    assert history["accepted"] == rt.REPLAYS
    items = []
    for row in old_design["execution_order"]:
        learned = row["arm"] != "FRAME_GUARD"
        items.append(dict(row, load_guard=learned, method=row["method"] + ("__guard09" if learned else "")))
    def key(row):
        return row["week"], row["arm"], row["pool"]
    assert [key(row) for row in items[12:]] == [key(row) for row in items[:12][::-1]]
    assert len({(row["week"], row["arm"], row["repeat"]) for row in items}) == rt.REPLAYS
    design = dict(old_design, id=rt.SCOPE, execution_order=items,
                  load_guard_threshold=rt.CUTOFF,
                  interpretation=(
                      "Same-run FRAME vs guarded208 vs guarded261. Compare complete cost only on complete horizons, "
                      "latency and queries on equal horizons, safety and completion separately. "
                      "Archive paid suffix per trigger and first delivered-action divergence. "
                      "Historical GRID312 unguarded arms provide descriptive cost/query context only, not fresh timing."
                  ),
                  boundaries=(
                      "Four already-exposed development dates, one frozen pool, two exact reverse serial repeats. "
                      "Zero training, new dates or final eight. One fixed window. No rerun of complete or partial rows."
                  ),
                  historical_runtime={**history, "resource_passed": False})
    rt.O.mkdir()
    rt.save(rt.A / "design.json", design)
    (rt.O / "design.json").write_bytes((rt.A / "design.json").read_bytes())
    rt.save(rt.O / "model_references.json", [model])
    dependencies = (*rt.A.glob("*.py"), rt.A / "design.json", old / "source_freeze.json",
                    old / "results/paired_results.json", old / "results/close_grid312.py",
                    rt.O / "design.json", rt.O / "model_references.json",
                    rt.R / "work/grid02/ljn_nn/LJNAgent.py")
    freeze = {**old_freeze, **{p.relative_to(rt.R).as_posix(): rt.sha(p) for p in dependencies}}
    assert all(old_freeze.get(k, v) == v for k, v in freeze.items())
    rt.save(rt.O / "code_freeze.json", freeze)
    rt.save(rt.A / "source_freeze.json", {**freeze,
        (rt.O / "code_freeze.json").relative_to(rt.R).as_posix(): rt.sha(rt.O / "code_freeze.json")})


def synthetic():
    globals_policy = {}
    original = {}
    exec(compile(rt.policy_source(), "guarded", "exec"), globals_policy)
    exec(compile((rt.OLD / "policy.py").read_text(encoding="utf8"), "original", "exec"), original)
    g = rt.runner.module("_grid320_synthetic_global", rt.R / "work/grid91_mechanism/policies.py")

    class Missing(ValueError):
        pass

    api = SimpleNamespace(G=g, best=lambda rows: max((r for r in rows if r["strict_admissible"]), key=g.key, default=None),
                          reward=lambda row: g.key(row)[0], observe=lambda *a: SimpleNamespace(tolist=lambda: [0.0]*41),
                          InputUnavailable=Missing)
    provider = SimpleNamespace(context=lambda *a: {})
    results = []
    for arm in ("PARENT208", "BOUNDED261"):
        for name, rho, native, extend, expected in (
            ("below_threshold_stop", .88, [False], False, (320, 320)),
            ("equal_threshold_stop", .90, [False], False, (320, 320)),
            ("above_threshold_stop", .95, [False], False, (320, 600)),
            ("previous_local_prefix", .95, [True, False], False, (384, 600)),
            ("paid_extension_unchanged", .95, [False], True, (384, 384)),
            ("no_pending_bypass", .50, [False], False, (256, 256)),
        ):
            observations = []
            for ns, paid in zip((original, globals_policy), expected, strict=True):
                seen = []
                sequence = iter(native)
                model = SimpleNamespace(native=lambda *a: {"continue_": next(sequence), "log": {}, "raw83": None},
                                        extension=lambda *a: {"continue_": extend, "log": {}})
                def reveal(i, phase):
                    assert i not in seen
                    seen.append(i)
                    local = name != "no_pending_bypass"
                    score = 2. if local and i==128 else 3. if local and i==530 else 1.
                    return dict(pool_id=i, phase=phase, reward=score, rho=rho, strict_admissible=True)
                policy = ns["CompletionPolicy"](arm, model, api)
                result = policy.run(list(range(600)), reveal, [0]*600, {i:i for i in range(600)},
                                    limit=1.2, provider=provider)
                assert seen == list(range(paid))
                assert result["chosen_pool_id"] == (0 if paid==256 else 530 if paid==600 else 128)
                observations.append(result)
            changed = expected[1]-expected[0]
            assert rt.load_guard_receipt(observations[1]) == changed
            assert observations[0]["learned_decisions"] == observations[1]["learned_decisions"]
            if not changed:
                fields = set(observations[0]) - {"local_feature_s", "local_inference_s"}
                assert all(observations[0][k] == observations[1][k] for k in fields)
            results.append(dict(arm=arm, case=name, original_paid=expected[0], guarded_paid=expected[1]))
    return results


def qualify():
    head = rt.git("rev-parse", "HEAD").decode().strip()
    for path in (*rt.A.glob("*.py"), rt.A / "design.json", rt.A / "source_freeze.json"):
        rt.tracked(path, head)
    generated = {"run": rt.assemble(), "adapter": rt.adapter_source(),
                 "worker": rt.runner.instrument_worker((rt.R / "work/grid55/worker.py").read_text(encoding="utf8"))}
    for name, code in generated.items():
        compile(code, "grid320_" + name, "exec")
    assert "grid320_load_guard_closed_loop" in generated["run"]
    assert "N.CompletionPolicy('FRAME_GUARD')" in generated["adapter"]
    cases = synthetic()
    assert not any(k in sys.modules for k in ("torch", "grid2op", "common"))
    rejected = 0
    for arguments in (["run"], ["worker", "0"]):
        p = subprocess.run([sys.executable, "-B", "-X", "utf8", str(rt.ENTRY), *arguments],
                           cwd=rt.R, capture_output=True, text=True, encoding="utf8", timeout=15)
        assert p.returncode == 2 and "authorization absent before live imports" in p.stderr
        rejected += 1
    assert not (rt.O / "runs").exists() and not (rt.A / "authorization.json").exists()
    rt.save(rt.A / "offline_review.json", {"passed": True, "source_commit": head, "compiled_sources": 3,
        "synthetic_policy_cases": cases, "missing_scope_entries_rejected": rejected,
        "new_model_loads": 0, "new_predictions": 0, "new_physics": 0, "new_fits": 0,
        "resource_passed": False, "live_authorization": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "qualify"))
    {"freeze": freeze, "qualify": qualify}[parser.parse_args().phase]()
