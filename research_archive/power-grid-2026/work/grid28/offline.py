import copy
import csv
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
for name in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]:os.environ[name]="1"
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT / "outputs/grid28"
spec=importlib.util.spec_from_file_location("semantic_complete",ROOT / "work/grid28/policy.py")
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
import torch
torch.set_num_threads(1)
D=json.loads((OUT / "design.json").read_text())
assert not (OUT / "offline.json").exists()
for p,h in json.loads((OUT / "schedule_freeze.json").read_text()).items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
checkpoint=ROOT / D["rankers"][0]["checkpoint"]
state=torch.load(checkpoint,map_location="cpu",weights_only=True)
subids=state["inner.subids"].tolist()  # Static public library metadata, shared by all rankers.
started=time.perf_counter();rows=[];hashes={checkpoint.relative_to(ROOT).as_posix():hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
def read(p):
    raw=p.read_bytes();hashes[p.relative_to(ROOT).as_posix()]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
for week in D["weeks"]:
    directory=ROOT / "outputs/grid26/runs" / f"r0__{week}__FULL"
    for path in sorted(directory.glob("call[0-9][0-9][0-9].json")):
        full=read(path);truth={r["pool_id"]:r for r in full["outcomes"]}
        for method in D["rankers"]:
            own=read(directory.parent / f"r0__{week}__{method['method']}" / path.name)
            assert own["state_hash"]==full["state_hash"]
            aliases={int(k):v for k,v in own["aliases"].items()}
            for policy in D["policies"]:
                result=P.local_complete(own["order"],lambda i,p:{**truth[i],"phase":p},policy,subids,aliases)
                seen={r["pool_id"] for r in result["outcomes"]};assert len(seen)==result["public_queries"]
                altered=copy.deepcopy(truth)
                for i in set(truth)-seen:altered[i].update(rho=.01,reward=99.,strict_admissible=True)
                again=P.local_complete(own["order"],lambda i,p:{**altered[i],"phase":p},policy,subids,aliases)
                assert result==again
                selected=truth.get(result["chosen_pool_id"]);winner=truth.get(full["chosen_pool_id"])
                assert selected is None or selected["strict_admissible"]
                rows.append({"week":week,"call":path.name,"step":own["step"],"ranker":method["method"],"policy":policy,
                    "queries":result["public_queries"],"local_queries":result["local_queries"],"protected_queries":own["public_queries"],
                    "choice":result["chosen_pool_id"],"full_choice":full["chosen_pool_id"],
                    "same_full_choice":result["chosen_pool_id"]==full["chosen_pool_id"],
                    "rho_regret":0. if winner is None else (None if selected is None else selected["rho"]-winner["rho"]),
                    "local_substation":result["local_substation"],"stop_reason":result["stop_reason"],"events":result["events"]})
assert len(rows)==156
summaries=[]
for method in D["rankers"]:
    for policy in D["policies"]:
        g=[r for r in rows if r["ranker"]==method["method"] and r["policy"]==policy]
        summaries.append({"ranker":method["method"],"policy":policy,"states":len(g),"queries":sum(r["queries"] for r in g),
            "query_reduction_to_protected":1-sum(r["queries"] for r in g)/sum(r["protected_queries"] for r in g),
            "same_full_choice":sum(r["same_full_choice"] for r in g),"changed_choice":sum(not r["same_full_choice"] for r in g),
            "max_rho_regret":max((r["rho_regret"] or 0.) for r in g),
            "march_critical_choice_changed":any(r["week"]==D["weeks"][0] and r["step"]==1733 and not r["same_full_choice"] for r in g)})
result={"passed_engineering":True,"summaries":summaries,"rows":rows,"hand_checks":P.hand_checks(),"permission_checks":156,
    "input_hashes":hashes,"public_subids":subids,"wall_s":time.perf_counter()-started,
    "new_forecasts":0,"new_fits":0,"reserved_evaluations":0,"warning":"Local completion is a shared non-learning rule, not a safe-stopping certificate or new graph contribution."}
(OUT / "offline.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
with (OUT / "offline_per_state.csv").open("w",newline="",encoding="utf-8-sig") as f:
    fields=[k for k in rows[0] if k!='events'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
(OUT / "RUN_STATE.md").write_text("OFFLINE COMPLETE. Review frozen pilot gate. No new forecasts/fits/holdout.\n")
print(json.dumps({"summaries":summaries,"permission_checks":156},indent=2))
