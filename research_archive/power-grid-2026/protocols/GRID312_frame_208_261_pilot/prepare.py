"""Freeze original inputs and inspect the assembled three-arm runtime offline."""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import runtime as rt

P = rt.A.parent
MODEL = rt.BASE/"model_references.json"
DESIGN = rt.BASE/"design.json"
BINDINGS = rt.R/"outputs/grid297_paired_collection_preparation/published_model_bindings.json"
WEEKS = ("2037-03-02_100","2037-05-04_100","2037-08-03_100","2037-10-19_100")


def tracked(path,head):
    assert rt.git("show",head+":"+path.relative_to(rt.I).as_posix())==path.read_bytes()


def freeze():
    assert not rt.O.exists()
    old = rt.read(DESIGN)
    assert old["environment_seed"]==0 and tuple(x["scenario"] for x in old["selected"])==WEEKS
    assert old["quality_threshold"]==.005
    original = {x["week"]:x for x in old["execution_order"]
                if x["repeat"]==0 and x["arm"]=="PARENT208" and x["pool"]=="f0_s0"}
    assert set(original)==set(WEEKS)
    first = []
    for j,week in enumerate(WEEKS):
        arms = rt.ARMS[j%3:]+rt.ARMS[:j%3]
        for arm in arms:
            first.append({**original[week],"arm":arm,"completion_mode":arm,
                          "method":arm if arm=="FRAME_GUARD" else arm+"__f0_s0",
                          "pool":None if arm=="FRAME_GUARD" else "f0_s0",
                          "model_seed":None if arm=="FRAME_GUARD" else 0,"within_pair":0})
    execution = [{**x,"repeat":0 if j<12 else 1,"index":j}
                 for j,x in enumerate(first+first[::-1])]
    assert len({(r["week"],r["arm"],r["repeat"]) for r in execution})==24
    head = rt.git("rev-parse","HEAD").decode().strip()
    parent = P/"GRID291_control_preparation/source_launch_recovery02"
    for path in (DESIGN,MODEL):
        origin = parent/path.name
        tracked(origin,head)
        assert rt.sha(origin)==rt.sha(path)
    model, = [x for x in rt.read(MODEL) if x["pool"]=="f0_s0"]
    model_bindings = rt.read(BINDINGS)
    for rel,h in model["hashes"].items():
        binding = model_bindings[rel]
        assert rt.sha(rt.R/rel)==binding["sha256"]==h
        assert hashlib.sha256(rt.git("show",head+":"+binding["archive_path"])).hexdigest()==h
    env = old["environment"]+"/"
    frozen_data = {rel:h for rel,h in rt.read(rt.BASE/"data_freeze.json").items() if
                   rel.startswith(env) and (not rel.startswith(env+"chronics/") or
                     any(rel.startswith(env+"chronics/"+week+"/") for week in WEEKS))}
    assert len(frozen_data)==70
    for week,path in old["full_references"].items():
        assert week in WEEKS
        frozen_data[path] = rt.read(rt.BASE/"data_freeze.json")[path]
    numerical = rt.read(rt.BASE/"code_freeze.json")
    design = {"id":"GRID312_FRAME_208_261_PILOT","pool":"f0_s0","environment":old["environment"],
              "environment_seed":0,"execution_order":execution,"full_references":old["full_references"],
              "quality_threshold":old["quality_threshold"],"caps":rt.CAPS,
              "timing":"Original same ctl.propose instrumentation, rotated first schedule and exact reverse second schedule; fresh serial workers.",
              "interpretation":"Pilot on four previously exposed development dates, one frozen pool. Paired cost for complete weeks, relative FULL quality on complete reference; latency and query ratios only on equal horizons. Report repeat directions, candidate and total forecasts, complete cost, P95 and P99 separately.",
              "boundaries":"Zero training, no new dates, no final eight, no extra runs or automatic retry. FULL historical cost is reference only."}
    rt.O.mkdir()
    rt.save(rt.A/"design.json",design)
    (rt.O/"design.json").write_bytes((rt.A/"design.json").read_bytes())
    rt.save(rt.O/"model_references.json",[model])
    files = [DESIGN,MODEL,rt.BASE/"data_freeze.json",rt.BASE/"code_freeze.json",BINDINGS,
             *rt.A.glob("*.py"),rt.O/"design.json",rt.O/"model_references.json",
             rt.OLD/"models.py",rt.OLD/"adapter.py",rt.OLD/"policy.py",
             rt.R/"work/grid297_paired_collection_preparation/source_builder.py",
             rt.R/"work/grid301_remaining_collection/guard.py"]
    freeze = {**numerical,**frozen_data,**model["hashes"],**{p.relative_to(rt.R).as_posix():rt.sha(p) for p in files}}
    for rel,h in freeze.items():
        assert rt.sha(rt.R/rel)==h,rel
    rt.save(rt.O/"code_freeze.json",freeze)
    rt.save(rt.A/"source_freeze.json",{**freeze,(rt.O/"code_freeze.json").relative_to(rt.R).as_posix():rt.sha(rt.O/"code_freeze.json")})
    print("Frozen four dates, 24 runs, seven model payloads and matching controller/data.")


def qualify():
    head = rt.git("rev-parse","HEAD").decode().strip()
    for path in (*rt.A.glob("*.py"),rt.A/"design.json",rt.A/"source_freeze.json"):
        tracked(path,head)
    code = rt.assemble()
    compile(code,"grid312_run.py","exec")
    compile(rt.instrument_worker((rt.R/"work/grid55/worker.py").read_text(encoding="utf8")),"grid312_worker.py","exec")
    compile(rt.adapter_source(),"grid312_adapter.py","exec")
    assert "None if item['arm']=='FRAME_GUARD' else models297.FrozenModels" in code
    assert "N.CompletionPolicy('FRAME_GUARD')" in rt.adapter_source()
    assert not any(k in sys.modules for k in ("torch","grid2op","common"))
    for args in (["run"],["worker","0"]):
        result = subprocess.run([sys.executable,"-B","-X","utf8",str(rt.A/"runtime.py"),*args],
                                cwd=rt.R,capture_output=True,text=True,encoding="utf8",timeout=15)
        assert result.returncode==2 and "absent before live imports" in result.stderr
    rt.save(rt.A/"offline_review.json",{"passed":True,"source_commit":head,"compiled_sources":3,
        "missing_scope_entries_rejected":2,"new_model_predictions":0,"new_physics":0,"new_fits":0})


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("freeze","qualify"))
    {"freeze":freeze,"qualify":qualify}[parser.parse_args().phase]()
