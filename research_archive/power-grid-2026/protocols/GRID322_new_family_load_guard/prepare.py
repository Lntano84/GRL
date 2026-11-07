"""Bind old scenario bytes and check the new four-arm program offline."""
import argparse
import hashlib
import subprocess
import sys

import family_review as plan
import runtime as rt


def freeze():
    head = plan.git("rev-parse", "HEAD").decode().strip()
    for path in (plan.A / "design.json", plan.A / "family_review.py", plan.PREV / "source_freeze.json"):
        plan.tracked(path, head)
    assert plan.read(plan.O / "family_review.json")["completed_week_has_trigger"]
    assert not (plan.O / "runs").exists()
    design = plan.read(plan.A / "design.json")
    inherited = plan.read(plan.PREV / "source_freeze.json")
    old_env = plan.read(plan.PREV / "design.json")["environment"] + "/"
    # The new trial uses only the four GRID309 chronics. Drop the old scenario
    # bytes and outcome reference files from the inherited dependency list.
    old_refs = set(plan.read(plan.PREV / "design.json")["full_references"].values())
    files = {k: v for k, v in inherited.items() if not k.startswith(old_env) and k not in old_refs}
    env_prefix = design["environment"] + "/"
    root = {p: h for p, h in plan.read(plan.SOURCE / "source_freeze.json").items() if p.startswith(env_prefix)}
    chronics = plan.read(plan.SOURCE / "generation_public/generated_freeze.json")
    assert len(root) == 18 and len(chronics) == 52
    assert all(p.startswith(env_prefix + "chronics/") for p in chronics)
    assert {p[len(env_prefix + "chronics/"):].split("/")[0] for p in chronics} == {x["scenario"] for x in design["calendar"]}
    files.update(root)
    files.update(chronics)
    model, = plan.read(plan.OLD / "model_references.json")
    assert model["pool"] == "f0_s0"
    for rel, record in plan.read(plan.SOURCE / "published_model_bindings.json").items():
        assert model["hashes"][rel] == record["sha256"] == files[rel]
        assert hashlib.sha256(plan.git("show", head + ":" + record["archive_path"])).hexdigest() == record["sha256"]
    (plan.O / "design.json").write_bytes((plan.A / "design.json").read_bytes())
    plan.save(plan.O / "model_references.json", [model])
    paths = [*plan.A.glob("*.py"), plan.A / "design.json", plan.O / "design.json",
             plan.O / "model_references.json", plan.O / "family_review.json", plan.O / "saved_stops.json",
             plan.PREV / "source_freeze.json"]
    files.update({p.relative_to(plan.R).as_posix(): plan.sha(p) for p in paths})
    files.update(design["source_hashes"])
    for member in design["saved_unprotected261"]:
        files.update(member["files"])
    for rel, expected in files.items():
        assert plan.sha(plan.R / rel) == expected, rel
    plan.save(plan.O / "code_freeze.json", files)
    plan.save(plan.A / "source_freeze.json", {**files,
        (plan.O / "code_freeze.json").relative_to(plan.R).as_posix(): plan.sha(plan.O / "code_freeze.json")})


def qualify():
    head = plan.git("rev-parse", "HEAD").decode().strip()
    for path in (*plan.A.glob("*.py"), plan.A / "design.json", plan.A / "source_freeze.json"):
        plan.tracked(path, head)
    assert not (plan.O / "runs").exists() and not (plan.A / "authorization.json").exists()
    code = rt.assemble()
    worker = rt.runner.instrument_worker((plan.R / "work/grid55/worker.py").read_text(encoding="utf8"))
    compile(code, "GRID322 generated run", "exec")
    compile(worker, "GRID322 generated worker", "exec")
    d = plan.read(plan.A / "design.json")
    assert "grid322_new_family_load_guard" in code
    assert "grid320_load_guard_closed_loop" not in code
    compiled = {}
    for row in d["execution_order"]:
        name = row["evaluation_arm"]
        adapter = rt.adapter_for(row)()
        if name not in compiled:
            compile(adapter, "GRID322 adapter " + name, "exec")
            assert "N.CompletionPolicy('FRAME_GUARD')" in adapter
            assert ("result['load_guard']" in adapter) == row["load_guard"]
            compiled[name] = row["load_guard"]
    assert len(compiled) == 4
    assert sum(x["load_guard"] for x in d["execution_order"]) == 8
    assert not any(name in sys.modules for name in ("torch", "grid2op", "common"))
    for args in (["run"], ["worker", "0"]):
        p = subprocess.run([sys.executable, "-B", "-X", "utf8", str(rt.ENTRY), *args],
                           cwd=plan.R, capture_output=True, text=True, encoding="utf8", timeout=15)
        assert p.returncode == 2 and "authorization absent before live imports" in p.stderr
    plan.save(plan.A / "offline_review.json", {"passed": True, "source_commit": head,
        "four_arm_routing": compiled, "compiled_sources": 6, "missing_scope_entries_rejected": 2,
        "source_hashes_verified": len(plan.read(plan.A / "source_freeze.json")),
        "generated_scenarios_reused": 4, "new_generation": 0, "new_fits": 0, "new_physics": 0,
        "new_predictions": 0, "resource_passed": False, "live_authorization": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "qualify"))
    {"freeze": freeze, "qualify": qualify}[parser.parse_args().phase]()
