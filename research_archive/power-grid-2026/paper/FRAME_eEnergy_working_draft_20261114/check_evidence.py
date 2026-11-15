"""Verify Table 1 and manuscript aggregates against published saved evidence."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from render_table import render

HERE = Path(__file__).resolve().parent
PROTOCOLS = HERE.parents[1] / "protocols"
ARTIFACTS = HERE.parents[5] / "outputs"
GRID330 = PROTOCOLS / "GRID330_remaining19/results/paired_results.json"
GRID334 = PROTOCOLS / "GRID334_logeipc_closed_loop/results/paired_results.json"
GRID335 = PROTOCOLS / "GRID335_march_divergence/summary.json"
GRID61 = PROTOCOLS / "GRID61_complete/outputs/grid61/runs"

def read(path):
    return json.loads(path.read_bytes())

def check():
    published = PROTOCOLS / "GRID337_native_last_batch/paper_table.py"
    spec = importlib.util.spec_from_file_location("grid337_published_table", published)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with TemporaryDirectory(dir=ARTIFACTS) as tmp:
        module.OUT = Path(tmp)
        module.main()
        regenerated = (module.OUT / "paired_main_table.csv").read_bytes()
    current = HERE / "paired_results.csv"
    assert regenerated == current.read_bytes()
    with current.open(encoding="utf8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    draft = (HERE / "manuscript_en.md").read_text(encoding="utf8")
    table = [line for line in draft.splitlines() if line.startswith("| ")]
    assert render(rows).splitlines() == table
    assert len(rows) == 12
    groups = {name: [r for r in rows if r["reference"] == name] for name in
              ("FRAME_GUARD", "PARENT208_GUARD_DC", "BOUNDED261_GUARD_DC")}
    assert all(len(group) == 4 for group in groups.values())
    for group in groups.values():
        assert sum(r["both_complete"] == "True" for r in group) == 3
        sep, = [r for r in group if r["date"] == "2039-09-12"]
        assert sep["arm_steps"] == sep["reference_steps"] == "451"
        assert sep["full_week_cost_delta_pct"] == ""
        assert sep["arm_exception_steps"] == sep["reference_exception_steps"] == "1"
        assert all(r[f"{arm}_{flag}"] == "0" for r in group for arm in ("arm", "reference")
                   for flag in ("illegal", "ambiguous"))

    frame, parent, bayes = groups.values()
    total = lambda group, key: sum(int(r[key]) for r in group)
    frame_delta = total(frame, "candidate_delta")
    frame_pct = -100 * frame_delta / total(frame, "reference_candidate_queries")
    assert frame_delta == -1615 and f"{frame_pct:.2f}" == "20.95"
    assert all(r["same_trajectory"] == "True" and r["candidate_delta"] == "0"
               and r["arm_total_forecasts"] == r["reference_total_forecasts"] for r in parent)
    previous = read(GRID330)
    ordering = [r for r in previous["pairs"] if r["arm"] == "BOUNDED261_GUARD_DC"
                and r["reference"] == "BOUNDED261_GUARD"]
    assert len(ordering) == 4 and sum(r["candidate_delta"] for r in ordering) == -1528
    protected = [r for r in previous["records"] if r["arm"] == "BOUNDED261_GUARD_DC"]
    assert len(protected) == 4 and all(r["residual_cancellations"] == 0 for r in protected)
    assert sum(r["controller_delta_pct"] == "" for r in frame) == 1
    assert frame[0]["date"] == "2039-03-14" and frame[0]["controller_delta_pct"] == ""
    bayes_delta = total(bayes, "candidate_delta")
    bayes_pct = -100 * bayes_delta / total(bayes, "reference_candidate_queries")
    forecast_delta = total(bayes, "arm_total_forecasts") - total(bayes, "reference_total_forecasts")
    assert (bayes_delta, f"{bayes_pct:.2f}", forecast_delta) == (-133, "2.18", -72)
    full_bayes = [r for r in bayes if r["both_complete"] == "True"]
    seconds = [sum(float(r[f"{side}_controller_s"]) for r in full_bayes) for side in ("arm", "reference")]
    assert [f"{x:.2f}" for x in seconds] == ["227.16", "229.82"]
    assert f"{100 * (1 - seconds[0] / seconds[1]):.2f}" == "1.16"
    march, = [r for r in bayes if r["date"] == "2039-03-14"]
    assert march["first_difference_step"] == "520" and f'{float(march["full_week_cost_delta_pct"]):.4f}' == "0.0861"
    split = read(GRID335)
    assert split["first_split"]["step"] == 520 and split["first_split"]["original_native_continue"]
    assert split["first_split"]["original_local_guard"] is False
    *intervals, whole_week = split["bands"]
    assert (whole_week["start"], whole_week["end"]) == (1, 2017)
    deltas = {key: sum(r["delta"][key] for r in intervals)
              for key in ("global_candidates", "local_candidates", "candidates")}
    assert all(whole_week["delta"][k] == v for k, v in deltas.items())
    assert deltas == dict(global_candidates=256, local_candidates=-105, candidates=151)
    march_ablation = []
    for repeat in (0, 1):
        runs = {arm: read(GRID61 / f"r{repeat}__2035-03-05_0__{arm}/summary.json")
                for arm in ("FULL", "NO_LOCAL", "FRAME")}
        assert all(r["complete"] and r["steps"] == 2017 for r in runs.values())
        delta = [100 * (runs[arm]["cost"] / runs["FULL"]["cost"] - 1)
                 for arm in ("NO_LOCAL", "FRAME")]
        assert [f"{x:.3f}" for x in delta] == ["4.810", "0.372"]
        march_ablation.append(delta)
    evidence = [current, HERE / "manuscript_en.md", published, GRID330, GRID334, GRID335,
                *(GRID61 / f"r{repeat}__2035-03-05_0__{arm}/summary.json"
                  for repeat in (0,1) for arm in ("FULL","NO_LOCAL","FRAME"))]
    print(json.dumps(dict(passed=True, table_rows=len(rows), evidence_hashes={
        path.relative_to(HERE.parents[1]).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in evidence}, frame_candidate_delta=frame_delta, frame_pct=frame_pct,
        logeipc_candidate_delta=bayes_delta, logeipc_forecast_delta=forecast_delta,
        logeipc_complete_seconds=seconds, march_2035_cost_deltas=march_ablation,
        march_2039_candidate_deltas=deltas, new_simulations=0, new_fits=0), indent=2))

if __name__ == "__main__":
    check()
