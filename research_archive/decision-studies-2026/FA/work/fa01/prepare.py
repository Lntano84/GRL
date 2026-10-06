"""Freeze inputs and a minimally adapted copy of the audited FA00 runner."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
OLD=PROJECT/"outputs"/"fa00"
OUT=PROJECT/"outputs"/"fa01"
OUT.mkdir(parents=True,exist_ok=True)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")

def main():
    assert not(OUT/"FA01_freeze_manifest.json").exists(),"Already frozen"
    frozen=read(OLD/"FA00_freeze_manifest.json")
    for path,digest in frozen["files"].items():assert sha(PROJECT/path)==digest,path
    assert read(OLD/"FA00_audit.json")["hard_failures"]==0
    original=(ROOT.parent/"fa00"/"run.py").read_text(encoding="utf-8")
    base=original.replace('"FA00_protocol.json"','"FA01_protocol.json"')
    base=base.replace('OUT=ROOT.parents[1]/"outputs"/"fa00"',
                      'OUT=ROOT.parents[1]/"outputs"/"fa01"')
    base=base.replace('if arm=="FIXED-100" or cap>=T:',
                      'if arm in ("FIXED-100","RANDOM-100","CHEAP-100") or cap>=T:')
    # Keep only the execution helpers; the old 18-trajectory CLI is not reused.
    base=base.split('\ndef main():')[0]
    assert original.count('if arm=="FIXED-100" or cap>=T:')==1
    ast.parse(base)
    (ROOT/"runner_base.py").write_text(base,encoding="utf-8")
    p=copy.deepcopy(read(OLD/"FA00_protocol.json"))
    p.update(name="FA01",arms=["RANDOM-100","CHEAP-100"],
        control="FA00 FIXED-100, three unchanged existing trajectories",
        development_only=True,test_feedback="No new test evaluation until all six new trajectories complete; this split was already inspected in FA00",
        acquisition_only_change=True,candidate_engine="Identical FA00 candidates() computes eligibility, uncertainty and PredCost for both controls; unused scores do not enter batch choice",
        random_rule="Uniform without replacement over canonical (instance,pair) candidates; no hidden cost matching",
        cheap_rule="Lowest legal PredCost; exact equal-cost ties randomized; no uncertainty use",
        acquisition_rng="keyed_seed(seed,'acquire',round); canonical instance,pair order before permutation",
        execution_rng="FA00 keyed_seed(seed,'batch',round), shuffle selected set",
        fixed_cap_s=100,validation="none",new_trajectories=6,workers=6,
        wall_limit_s=7200,
        continuation="Pareto mean PAR10 >=5% lower than each control; at least 2/3 seeds in same direction per comparison",
        closure="Any cheap control mean PAR10 <=1.02*Pareto mean, including a better control",
        rule_priority="closure before continuation if ever overlapping, else undetermined",
        no_new_models=True,no_extra_seeds_or_budgets=True,
        timing_warning="Same candidate engine and regressors retained even for random sampling; no claim that baseline compute is optimized; concurrent local timings not speed comparisons")
    write(OUT/"FA01_protocol.json",p)
    inherited=[OLD/"FA00_protocol.json",OLD/"FA00_data.npz",OLD/"FA00_preprocessing.npz",OLD/"FA00_data_audit.json",OLD/"FA00_audit.json"]
    for seed in p["seeds"]:
        dest=OLD/"runs"/f"FIXED-100_s{seed}"
        assert read(dest/"result.json")["status"]=="complete"
        inherited.extend(dest/n for n in ["result.json","history.json","final_predictions.npy","actions.jsonl.gz"])
        inherited.extend(sorted(dest.glob("predictions_r*.npy")))
    inherited.extend(ROOT.parent/"fa00"/n for n in ["core.py","run.py","audit_analyse.py","source_manifest.json"])
    inherited.extend(ROOT.parent/"fa00"/"upstream"/e["path"] for e in read(ROOT.parent/"fa00"/"source_manifest.json")["files"])
    write(OUT/"FA01_inherited_manifest.json",{str(f.relative_to(PROJECT)):sha(f) for f in inherited})
    write(OUT/"RUN_STATE.json",dict(status="prepared_not_frozen",target=6))
    print("Prepared: six trajectories; inherited baseline and model hashes locked")

if __name__=="__main__":main()
