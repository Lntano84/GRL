"""Freeze a quality-first, existing-threshold search safeguard experiment.

New files only. No old weights/results changed; no reserved weeks opened.
"""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "work/grid24"
OUT = ROOT / "outputs/grid24"


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:120]
    return text.replace(old, new)


def main():
    assert json.loads((ROOT / "outputs/grid23/audit.json").read_text())["passed"]
    assert not OUT.exists(), "Never overwrite a prepared experiment."
    OUT.mkdir()
    original = (ROOT / "work/grid23/canonical_adapter.py").read_text()
    source = original.replace('class Search:', 'class Search:')
    source = replace_once(source,
        '        fallback = not any(r["strict_admissible"] for r in outcomes) and len(first) < len(reps)',
        '''        shortlist_valid = [i for i in first if cache[i]["strict_admissible"]]
        shortlist_choice = max(shortlist_valid, key=lambda i: (float(np.float32(cache[i]["reward"])), -i)) if shortlist_valid else None
        guard_rho = float(self.ctl.base.rho_safe)
        assert guard_rho == .9  # Author default; no calibration or threshold search.
        guard_reason = ("no_admissible_shortlist" if shortlist_choice is None else
                        "shortlist_above_author_safe" if cache[shortlist_choice]["rho"] >= guard_rho else
                        "shortlist_below_author_safe")
        fallback = bool(self.rule != "FULL" and len(first) < len(reps) and
                        (shortlist_choice is None or cache[shortlist_choice]["rho"] >= guard_rho))''')
    source = replace_once(source,
        '               "rho_limit": limit, "fallback": fallback, "chosen_pool_id": chosen,',
        '               "rho_limit": limit, "fallback": fallback, "chosen_pool_id": chosen,\n'
        '               "guard_rho": guard_rho, "guard_reason": guard_reason, "shortlist_choice": shortlist_choice,')
    (SRC / "guard_adapter.py").write_text(source, encoding="utf-8")
    (OUT / "adapter_delta.patch").write_text(''.join(difflib.unified_diff(
        original.splitlines(True), source.splitlines(True), fromfile="GRID23", tofile="GRID24")), encoding="utf-8")

    run = (ROOT / "work/grid23/run.py").read_text()
    run = run.replace('"work/grid23"', '"work/grid24"').replace('"outputs/grid23"', '"outputs/grid24"')
    run = replace_once(run, 'from canonical_adapter import Search', 'from guard_adapter import Search')
    run = replace_once(run, 'def main():', '''def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pilot", "remaining", "all"], default="pilot")
    phase = parser.parse_args().phase
    selected = [x for x in D["execution_order"] if phase == "all" or x["phase"] == phase]''')
    run = replace_once(run, '    assert not (OUT / "finished.json").exists()\n    results = []\n    for item in D["execution_order"]:',
        '''    results = json.loads((OUT / "manifest.json").read_text()) if (OUT / "manifest.json").exists() else []
    completed = {(s["week"], s["method"]) for s in results}
    for item in selected:
        if (item["week"], item["method"]) in completed:
            print(json.dumps({"preserved_completed": item["week"] + "__" + item["method"]}), flush=True)
            continue''')
    run = replace_once(run,
        '    write_json(OUT / "finished.json", {"passed_engineering": True, "summaries": results, "counts": counts, "wall_s": time.perf_counter()-START})',
        '''    aggregate = {"physical_steps": sum(s["steps"] for s in results),
                 "public_forecasts": sum(s["public_forecasts"] for s in results), "model_fits": 0}
    write_json(OUT / ("finished.json" if len(results) == len(D["execution_order"]) else phase + "_finished.json"),
        {"passed_engineering": True, "summaries": results, "counts": aggregate, "phase": phase,
         "wall_s_this_process": time.perf_counter()-START, "wall_s_sum_runs": sum(s["wall_s"] for s in results)})''')
    (SRC / "run.py").write_text(run, encoding="utf-8")

    audit = (ROOT / "work/grid23/audit.py").read_text()
    audit = audit.replace('"outputs/grid23"', '"outputs/grid24"')
    audit = replace_once(audit,
        'D = json.loads((OUT / "design.json").read_text()); F = json.loads((OUT / "finished.json").read_text())',
        '''import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=["pilot", "all"], default="pilot")
phase = parser.parse_args().phase
D = json.loads((OUT / "design.json").read_text())
F = json.loads((OUT / ("finished.json" if phase == "all" else "pilot_finished.json")).read_text())
if phase == "pilot": D["execution_order"] = [x for x in D["execution_order"] if x["phase"] == "pilot"]''')
    audit = replace_once(audit,
        '            fallback = not any(delivered[i]["strict_admissible"] for i in shortlist) and len(shortlist)<len(reps)',
        '''            short_valid = [i for i in shortlist if delivered[i]["strict_admissible"]]
            short_choice = max(short_valid, key=lambda i: (float(np.float32(delivered[i]["reward"])), -i)) if short_valid else None
            assert c["shortlist_choice"] == short_choice and c["guard_rho"] == .9
            reason = ("no_admissible_shortlist" if short_choice is None else
                      "shortlist_above_author_safe" if delivered[short_choice]["rho"] >= .9 else
                      "shortlist_below_author_safe")
            assert c["guard_reason"] == reason
            fallback = bool(rule != "FULL" and len(shortlist) < len(reps) and
                            (short_choice is None or delivered[short_choice]["rho"] >= .9))''')
    audit = audit.replace('OUT / "audit.json"', 'OUT / ("audit.json" if phase == "all" else "pilot_audit.json")')
    (SRC / "audit.py").write_text(audit, encoding="utf-8")

    previous = json.loads((ROOT / "outputs/grid23/design.json").read_text())
    methods = [m.copy() for m in previous["methods"] if m["rule"] in ["GNN_seed0", "GNN_seed1", "OLDNN"]]
    for m in methods: m["method"] += "_GUARD"
    order = [{"week": previous["weeks"][0], "phase": "pilot", **m} for m in methods if m["rule"].startswith("GNN")]
    for wi, week in enumerate(previous["weeks"]):
        seq = methods[wi % len(methods):] + methods[:wi % len(methods)]
        order.extend({"week": week, "phase": "remaining", **m} for m in seq
                     if not (wi == 0 and m["rule"].startswith("GNN")))
    d = {"stage": "GRID24_AUTHOR_SAFE_THRESHOLD_CACHED_FULL_EXPANSION", "weeks": previous["weeks"],
         "methods": methods, "execution_order": order, "new_model_fits": 0, "environment_seed": 0,
         "pilot": "Two frozen GNN seeds on the exposed March week with worst K128 cost regret.",
         "guard": "Evaluate K128, choose using unchanged exact native-reward tie. If no admissible candidate or its forecast rho >= author's existing 0.9 threshold, query the uncached complement and choose over all candidates. No duplicate forecast. No late-time switch.",
         "unchanged": "All V1 weights, native objective, candidate library/masks/aliases, legality, continuous control, restoration cadence6, physical environment and seed. No new labels, fits or RL.",
         "pilot_gate": "Both GNN seeds complete the FULL-complete March week, cost regret <=1%, and fewer N1 queries than existing FULL. Otherwise diagnose before running the remaining matrix.",
         "development_quality_gate": "No lost completed week or earlier failure versus sealed GRID23 FULL. Common-complete mean cost regret <=0.5%, worst <=1%. These are engineering investment criteria, not publication requirements.",
         "efficiency": "Report N1 query reduction plus component and total time. Query reduction >=25% is a screen. Existing timed reference is descriptive; no speed certification from one instrumented run.",
         "reference": "Sealed GRID23 FULL trajectories, environment seed0, same controller. No new FULL baseline claimed. Unprotected same-K arms remain as immutable ablations.",
         "split": "Exposed development weeks only; all 8 reserved date families remain unopened.",
         "caps": {"wall_s": 7200, "episode_s": 600, "physical_steps": 26000, "public_forecasts": 1000000},
         "sources": ["https://arxiv.org/html/2407.19865v1", "https://arxiv.org/html/2106.15200v1"],
         "notes": "Existing verified hybrid fallback, not a novel algorithm or production safety guarantee. Guard available identically to GNN and original PPO ranking. MLP/bias extensions deferred until quality evidence warrants them."}
    (OUT / "design.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    fixed = [ROOT / "outputs/grid23/design.json", ROOT / "outputs/grid23/audit.json", ROOT / "outputs/grid23/manifest.json"]
    fixed += [ROOT / m["checkpoint"] for m in methods if m["checkpoint"]]
    fixed += [ROOT / "work/grid23/canonical_adapter.py", ROOT / "work/grid23/run.py"]
    for week in d["weeks"]:
        fixed.extend(sorted((ROOT / "outputs/grid23/runs" / (week + "__FULL")).glob("*")))
    preservation = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in fixed if p.is_file()}
    (OUT / "preservation_hashes.json").write_text(json.dumps(preservation, indent=2), encoding="utf-8")
    (OUT / "RUN_STATE.md").write_text("PREPARED. Frozen 12 development runs; pilot first. No new training or reserved-family evaluation.\n", encoding="utf-8")
    print(json.dumps({"prepared": True, "runs": len(order), "pilot_runs": 2, "old_files_sealed": len(preservation)}))


if __name__ == "__main__": main()
