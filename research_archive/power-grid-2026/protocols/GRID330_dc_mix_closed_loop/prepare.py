"""Freeze inputs and qualify GRID330 without constructing an environment."""
import argparse
import ast
import hashlib
import runpy
import subprocess
import sys
from types import SimpleNamespace

import runtime as rt

PREV = rt.A.parent / "GRID322_new_family_load_guard"
SOURCE = rt.A.parent / "GRID309_action_opportunity_preparation"
DC_OLD = rt.A.parent / "GRID325_dc_global_priority"
DC_FAST = rt.A.parent / "GRID328_dc_fast_solver"
DC_PREP = rt.A.parent / "GRID329_prepared_dc_plans"
DC_MIX = rt.A.parent / "GRID326_fixed_dc_mix"
TEMPLATES = rt.R / "outputs/grid322_new_family_load_guard"


def refs(*paths):
    return {p.relative_to(rt.R).as_posix(): rt.sha(p) for p in paths}


def freeze():
    head = rt.git("rev-parse", "HEAD").decode().strip()
    previous = rt.read(PREV / "design.json")
    parent_paths = [PREV / "design.json", PREV / "source_freeze.json",
                    PREV / "results/closed_loop/finished.json", SOURCE / "published_model_bindings.json"]
    for path in parent_paths:
        rt.tracked(path, head)
    calendar = previous["calendar"]
    assert len(calendar) == 4 and {c["environment_seed"] for c in calendar} == {0}
    assert {c["model_pool"] for c in calendar} == {"f0_s0"}
    assert rt.read(PREV / "results/closed_loop/finished.json")["accepted"] == 16
    roles = tuple(rt.ARMS)
    templates = {r["week"]: r for r in previous["execution_order"] if r["evaluation_arm"] == "PARENT208_GUARD"}
    items = []
    for j, day in enumerate(calendar):
        template = templates[day["scenario"]]
        for role in roles[j:] + roles[:j]:
            arm, guarded, mixed = rt.ARMS[role]
            learned = arm != "FRAME_GUARD"
            items.append(dict(template, index=len(items), evaluation_arm=role, arm=arm,
                              load_guard=guarded, dc_mix=mixed, completion_mode=arm,
                              method=role+("__f0_s0" if learned else ""), pool="f0_s0" if learned else None,
                              model_seed=0 if learned else None, order_mode="STATIC64_DC64" if mixed else "STATIC_ID"))
    assert len(items) == rt.REPLAYS == 20
    assert all(r["within_pair"] == 0 and r["repeat"] == 0 and r["budget"] == 128 for r in items)
    design = {
        "id": rt.SCOPE, "environment": previous["environment"], "environment_seed": 0,
        "pool": "f0_s0", "load_guard_threshold": 0.9, "caps": rt.CAPS, "calendar": calendar,
        "execution_order": items, "learned_models_trained": 0,
        "comparisons": {
            "replacement": ["BOUNDED261_GUARD_DC", "FRAME_GUARD"],
            "learning_increment": ["BOUNDED261_GUARD_DC", "PARENT208_GUARD_DC"],
            "ordering_with_208": ["PARENT208_GUARD_DC", "PARENT208_GUARD"],
            "ordering_with_261": ["BOUNDED261_GUARD_DC", "BOUNDED261_GUARD"],
            "learning_reference_static": ["BOUNDED261_GUARD", "PARENT208_GUARD"]},
        "procedure": "Fresh serial one pass, arm start rotated by date, existing f0_s0 only. Freeze initial state, seed and scenario bytes. Mixed arms preserve the original controller and ordered local logic with a 64+64 shortlist and original-order remainder. Calls with <=128 or unsupported base/canonical input use the original order and log the reason.",
        "readout": "Report every date, same-initial-state paired delivered trajectory and first divergence, completed-horizon costs, completion and legality/safety, query counts, whole-control time and search/all-step P95/P99 on equal horizons, component subtotals. For equal duration early termination compare descriptive timing with completion status stated. No pooling in old timing. One run per arm/date, descriptive and development evidence.",
        "resource": {"historical_grid322_seconds": 1336.754949099999, "historical_runs": 16,
                     "projected_seconds_before_new_overhead": 1336.754949099999 * 20 / 16,
                     "resource_passed": False, "no_retry": True},
    }
    rt.save(rt.A / "design.json", design)
    (rt.O / "design.json").write_bytes((rt.A / "design.json").read_bytes())
    model, = rt.read(TEMPLATES / "model_references.json")
    assert model["pool"] == "f0_s0"
    rt.save(rt.O / "model_references.json", [model])
    files = {p: h for p, h in rt.read(PREV / "source_freeze.json").items()
             if not p.startswith(TEMPLATES.relative_to(rt.R).as_posix() + "/")}
    for path, binding in rt.read(SOURCE / "published_model_bindings.json").items():
        assert files[path] == model["hashes"][path] == binding["sha256"]
        assert hashlib.sha256(rt.git("show", head + ":" + binding["archive_path"])).hexdigest() == binding["sha256"]
    for leaf in ("grid.json", "config.py"):
        old = rt.R / "work/grid100_data_foundation/trainval_env" / leaf
        new = rt.R / design["environment"] / leaf
        assert rt.sha(old) == rt.sha(new) == files[new.relative_to(rt.R).as_posix()]
        files.update(refs(old))
    dll = rt.R / "outputs/grid328_dc_fast_solver/solver.dll"
    assert rt.sha(dll) == rt.read(DC_FAST / "fixed_scope.json")["binary_sha256"]
    prereq = [DC_FAST / "run.py", DC_FAST / "fixed_scope.json", DC_PREP / "readout.py",
              DC_PREP / "fixed_scope.json", DC_OLD / "readout.py", DC_OLD / "fixed_scope.json",
              DC_OLD / "public_ranks.json", DC_MIX / "readout.py", DC_MIX / "public_mix.json"]
    for path in prereq:
        rt.tracked(path, head)
    files.update(refs(*parent_paths, *prereq, dll, *rt.A.glob("*.py"), rt.A / "design.json",
                      rt.O / "design.json", rt.O / "model_references.json"))
    for rel, expected in files.items():
        assert rt.sha(rt.R / rel) == expected, rel
    rt.save(rt.O / "code_freeze.json", files)
    rt.save(rt.A / "source_freeze.json", {**files, **refs(rt.O / "code_freeze.json")})


def qualify():
    head = rt.git("rev-parse", "HEAD").decode().strip()
    for path in (*rt.A.glob("*.py"), rt.A / "design.json", rt.A / "source_freeze.json"):
        rt.tracked(path, head)
    assert not (rt.A / "authorization.json").exists() and not (rt.O / "runs").exists()
    design = rt.read(rt.A / "design.json")
    generated = {"runner": rt.assemble(), "worker": rt.instrument_worker(
        (rt.R / "work/grid55/worker.py").read_text(encoding="utf8"))}
    for row in design["execution_order"]:
        name = row["evaluation_arm"]
        if name not in generated:
            text = rt.adapter_source(row)
            ast_text = [n.value for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Constant)]
            assert (rt.parent.policy_source() in ast_text) == row["load_guard"]
            assert ("LiveRank()" in text) == row["dc_mix"]
            assert ("dc_mix_receipt" in text) == row["dc_mix"]
            generated[name] = text
    assert "grid330_dc_mix_closed_loop" in generated["runner"] and len(generated) == 7
    for name, text in generated.items():
        compile(text, "GRID330 " + name, "exec")
    for args in (["run"], ["worker", "0"]):
        p = subprocess.run([sys.executable, "-B", "-X", "utf8", str(rt.ENTRY), *args],
                           cwd=rt.R, capture_output=True, text=True, encoding="utf8", timeout=15)
        assert p.returncode == 2 and "authorization absent before live imports" in p.stderr
    assert not any(name in sys.modules for name in ("torch", "grid2op", "common"))

    from ordering import LiveRank
    ranker = LiveRank()
    sys.path.insert(0, str(rt.R / ranker.prior["DECODER"]))
    from public import Decoder
    import numpy as np

    decoder = Decoder()
    search = SimpleNamespace(env=decoder.ac, library=decoder.library, subids=decoder.subs)
    scope = rt.read(DC_OLD / "fixed_scope.json")
    old_rank = {r["call_id"]: r for r in rt.read(DC_OLD / "public_ranks.json")["rows"]}
    mix = {r["call_id"]: r for r in rt.read(DC_MIX / "public_mix.json")["rows"]}
    cases = []
    for date in scope["dates"]["validation"]:
        # Always the first published member for each date, regardless of its score or outcome.
        member = next(m for m in scope["members"] if m["date"] == date)
        cid = member["call_id"]
        for field in ("source", "state"):
            rel = member[field]
            assert rt.sha(rt.R / rel) == scope["source_hashes"][rel]
        with np.load(rt.R / member["state"], allow_pickle=False) as z:
            obs = decoder.oc.from_vect(z["observation"])
            base = decoder.ac.from_vect(z["base_action"])
        _, reps, _, _, _, first = ranker.prepare(search, obs, base)
        order = sorted(reps)
        result, record = ranker.apply(search, obs, base, order, first)
        assert record["status"] == "ranked" and result[:128] == mix[cid]["mix128"]
        assert record["dc128"] == old_rank[cid]["dc128"] and record["fallback"] == old_rank[cid]["fallback"]
        assert record["scored"] == old_rank[cid]["scored"] and set(result) == set(order)
        assert result[128:] == [i for i in order if i not in set(result[:128])]
        short, bypass = ranker.apply(search, obs, base, order[:128], None)
        assert short == order[:128] and bypass["reason"] == "pool_fits_first_batch"
        cases.append({"call_id": cid, "M": len(order), "scored": record["scored"],
                      "solver_counts": record["solver_counts"], "order_exact": True})
    assert any(row["scored"] > 0 for row in cases)
    assert "common" not in sys.modules and "torch" not in sys.modules
    assert not (rt.O / "runs").exists()
    rt.save(rt.A / "offline_review.json", {"passed": True, "source_commit": head,
        "compiled_sources": list(generated), "denied_without_receipt": 2, "saved_adapter_cases": cases,
        "model_loads": 0, "new_model_forward": 0, "new_grid2op_simulation": 0, "new_fits": 0,
        "resource_passed": False, "live_authorization": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "qualify"))
    {"freeze": freeze, "qualify": qualify}[parser.parse_args().stage]()

