"""Prepare a narrow development revision; no model fit or environment rollout.

Run only after GRID22 is complete and its audit passes. Shared native-reward
tie control and a previously evaluated candidate budget are not novel modules.
"""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "work/grid23"
OUT = ROOT / "outputs/grid23"


def main():
    assert json.loads((ROOT / "outputs/grid22/audit.json").read_text())["passed"]
    assert (ROOT / "outputs/grid22/delivery_manifest.json").is_file()
    for name, sha in json.loads((ROOT / "outputs/grid22/delivery_manifest.json").read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    OUT.mkdir(parents=True, exist_ok=False)
    adapter = (ROOT / "work/grid22/adapter.py").read_text(encoding="utf-8")
    old = "-considered.index(i)"
    assert adapter.count(old) == 1
    canonical = adapter.replace(old, "-i")
    (SRC / "canonical_adapter.py").write_text(canonical, encoding="utf-8")
    run = (ROOT / "work/grid22/run.py").read_text(encoding="utf-8")
    assert run.count('from adapter import Search') == 1
    run = run.replace('from adapter import Search', 'from canonical_adapter import Search')
    run = run.replace('str(ROOT / "work/grid22")', 'str(ROOT / "work/grid23")')
    run = run.replace('OUT = ROOT / "outputs/grid22"', 'OUT = ROOT / "outputs/grid23"')
    run = run.replace('json.loads((ROOT / "outputs/grid21/audit.json").read_text())["passed"]',
                      'json.loads((ROOT / "outputs/grid22/audit.json").read_text())["passed"]')
    (SRC / "run.py").write_text(run, encoding="utf-8")
    preflight = (ROOT / "work/grid22/preflight.py").read_text(encoding="utf-8")
    preflight = preflight.replace('str(ROOT / "work/grid22")', 'str(ROOT / "work/grid23")')
    preflight = preflight.replace('from adapter import Search', 'from canonical_adapter import Search')
    preflight = preflight.replace('OUT = ROOT / "outputs/grid22"', 'OUT = ROOT / "outputs/grid23"')
    # These occurrences are budget/index operations; ranks still compare the
    # entire old frozen order, not a newly selected teacher target.
    import re
    preflight = re.sub(r'\b32\b', '128', preflight)
    (SRC / "preflight.py").write_text(preflight, encoding="utf-8")
    audit = (ROOT / "work/grid22/audit.py").read_text(encoding="utf-8")
    audit = audit.replace('OUT = ROOT / "outputs/grid22"', 'OUT = ROOT / "outputs/grid23"')
    assert audit.count('-considered.index(i)') == 1
    audit = audit.replace('-considered.index(i)', '-i')
    (SRC / "audit.py").write_text(audit, encoding="utf-8")
    closeout = (ROOT / "work/grid22/closeout.py").read_text(encoding="utf-8")
    closeout = closeout.replace('OUT = ROOT / "outputs/grid22"', 'OUT = ROOT / "outputs/grid23"')
    closeout = closeout.replace('["OLDNN32", "OLDNN128", "MLP0_32", "BIAS0_32"]',
                                '["OLDNN128", "MLP0_128", "BIAS0_128"]')
    closeout = closeout.replace('"GNN0_32"', '"GNN0_128"').replace('"GNN1_32"', '"GNN1_128"')
    closeout = closeout.replace(' if g["reference"] != "OLDNN128"', '')
    closeout = closeout.replace('ROOT / "work/grid22"', 'ROOT / "work/grid23"')
    closeout = closeout.replace('GRID22_report.md', 'GRID23_report.md')
    closeout = closeout.replace('GRID22：PPO先验残差GNN的整周闭环开发比较', 'GRID23：共同精确破并列与128候选的闭环修订')
    closeout = closeout.replace('冻结28条完整轨迹', '冻结24条完整轨迹')
    closeout = closeout.replace('COMPLETE.28 closed-loop developmental runs audited and sealed. No pending jobs, new fits or RL in stage22.',
                                'COMPLETE.24 canonical-tie K128 development runs audited and sealed. No pending jobs, new fits or RL in stage23.')
    closeout = closeout.replace('对同预算PPO、MLP及固定偏置的图增量筛查', '对同为128候选的PPO、MLP及固定偏置的图增量筛查')
    marker = '(OUT / "GRID23_report.md").write_text'
    assert closeout.count(marker) == 1
    closeout = closeout.replace(marker, 'lines += ["", "此次同时更换共同的精确破并列规则与候选预算；这是修订后配置的成绩，不是两个因素的独立因果消融。原GRID22与其模型保持封存。"]\n' + marker)
    (SRC / "closeout.py").write_text(closeout, encoding="utf-8")
    delta = "".join(difflib.unified_diff(adapter.splitlines(True), canonical.splitlines(True),
                                        fromfile="GRID22/adapter.py", tofile="GRID23/canonical_adapter.py"))
    (OUT / "adapter_delta.patch").write_text(delta, encoding="utf-8")
    base = json.loads((ROOT / "outputs/grid22/design.json").read_text())
    methods = [m.copy() for m in base["methods"] if m["method"] != "OLDNN32"]
    for m in methods:
        if m["method"] != "FULL":
            m["budget"] = 128
            m["method"] = m["method"].replace("_32", "_128")
    execution = []
    for i, week in enumerate(base["weeks"]):
        alternatives = methods[1:]
        sequence = [methods[0]] + alternatives[i % len(alternatives):] + alternatives[:i % len(alternatives)]
        execution += [{"week": week, **m} for m in sequence]
    design = {
        "stage": "GRID23_COMMON_NATIVE_TIE_AND_K128_DEVELOPMENT",
        "weeks": base["weeks"], "methods": methods, "execution_order": execution,
        "environment_seed": 0, "new_model_fits": 0,
        "caps": {"wall_s": 7200, "episode_s": 600, "physical_steps": 52000, "public_forecasts": 1000000},
        "motivation": "K32 closed-loop differs from teacher-state ranking; exact native-reward ties can change chosen actions despite best candidates being queried. Test coverage at previously specified K128 without training or changing targets.",
        "change": "All methods choose smallest canonical action ID on EXACT equal native float32 reward. No near-tie tolerance. GNN, MLP, bias and original PPO all use K128 with identical masked candidates, admission and cached full fallback.",
        "unchanged": "V1 checkpoints, training labels, original PPO, other control modules, continuous optimizer, restoration cadence6, actual endpoint mask and action equivalence, four exposed development weeks and environment seed0.",
        "comparison": "Primary both GNN seeds versus FULL on complete-week parity, per-week cost regret<=1%, mean controller reduction>=10%, mean search forecast reduction>=30%. Graph increment against same-K original PPO, MLP and fixed bias requires survival parity and mean two-GNN-seed cost gain>=0.5% for each. No new significance or final-test claim.",
        "notes": "Two factors change together. This stage tests the final revised configuration, not a causal decomposition of tie handling and budget. Compare exact tie diagnostic separately. No models selected from these results; all specified checkpoints remain archived.",
        "reservation": "All variants of outputs/grid_final_holdout/reserved_date_families.json remain unopened; original GRID03 test partition was partly consumed and cannot support final-test claims.",
    }
    (OUT / "design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    (OUT / "RUN_STATE.md").write_text("PREPARED. No preflight or rollout yet, no fits or RL.\n", encoding="utf-8")
    print(json.dumps({"prepared": True, "runs": len(execution), "methods": [m["method"] for m in methods], "new_fits": 0}))


if __name__ == "__main__":
    main()
