"""Read all saved GRID312 calls at their exact common control inputs."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import time
import traceback
import zipfile

R = Path("D:/Codex/GRL研究")
I = R/"work/GRL-results-integration"
A = Path(__file__).resolve().parent
P = A.parent/"GRID312_frame_208_261_pilot"
SOURCE = R/"outputs/grid312_frame_208_261_pilot"
OUT = R/"outputs/grid313_frame_learning_call_attribution"
REFS = P/"results/raw_records_manifest.json"
SCOPE = A/"fixed_scope.json"
LEARNED = ("PARENT208","BOUNDED261")
FEEDBACK = ("pool_id","rho","reward","done","illegal","ambiguous","exceptions","strict_admissible","phase","combined_hash")
PUBLIC = ("order","aliases","representatives","rho_limit","completion_candidate_ids","local_substation","parameters")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git","--no-optional-locks","-C",str(I),*args])


def verify(path,head):
    assert git("show",head+":"+path.relative_to(I).as_posix())==path.read_bytes(),path


def freeze():
    assert not OUT.exists() and not SCOPE.exists()
    head = git("rev-parse","HEAD").decode().strip()
    dependencies = (P/"design.json",P/"results/paired_results.json",REFS)
    for path in dependencies:
        verify(path,head)
    raw = read(REFS)
    assert raw["passed"] and raw["independent_object_readback"]
    refs = {r["path"]:r for r in raw["entries"]}
    index = {row["index"]:row for row in read(P/"results/paired_results.json")["records"]}
    members = []
    for item in read(P/"design.json")["execution_order"]:
        key = item["index"]
        assert key in index and index[key]["arm"]==item["arm"]
        folder = (SOURCE/"runs"/f"r{item['repeat']}__{item['week']}__{item['method']}").relative_to(R).as_posix()
        calls = sorted(path for path in refs if path.startswith(folder+"/call") and path.endswith(".json"))
        assert all(path[:-5]+"_state.npz" in refs for path in calls)
        paths = [folder+"/steps.jsonl.gz",folder+"/summary.json",*calls,
                 *[path[:-5]+"_state.npz" for path in calls]]
        members.append({"item":item,"calls":calls,"folder":folder,"files":{path:refs[path] for path in paths}})
    assert len(members)==24 and sum(len(x["calls"]) for x in members)==184
    save(SCOPE,{"members":members,"raw_archive":raw["local_archive"],
         "input_sha256":{p.relative_to(R).as_posix():digest(p.read_bytes()) for p in dependencies},
         "pair_identity":"Same date, repeat, step, full observation and full base action; same order, aliases, station, eligibility limit, shared feedback and exact paid candidate prefix for stop attribution.",
         "mechanism":"Account exact skipped tail by last recorded decision, without a native-only counterfactual or summing across divergent paths as if matched.",
         "physical":"Audit first delivered action difference and same-state search evidence at that step. Cost at this step does not identify whole-week causality.",
         "wall_cap_s":600,"new_physics":0,"new_predictions":0,"new_fits":0})


def reward(row):
    return struct.unpack("f",struct.pack("f",row["reward"]))[0]


def load_run(member,archive):
    import numpy as np
    def raw(path):
        ref = member["files"][path]
        local = R/path
        data = local.read_bytes() if local.is_file() else archive.read(ref["object"])
        assert len(data)==ref["bytes"] and digest(data)==ref["sha256"],path
        return data
    root = member["folder"]
    steps = [json.loads(row) for row in gzip.decompress(raw(root+"/steps.jsonl.gz")).splitlines()]
    summary = json.loads(raw(root+"/summary.json"))
    item = member["item"]
    assert all(summary[k]==v for k,v in item.items())
    assert len(steps)==summary["steps"] and len(member["calls"])==summary["n1_calls"]
    calls = {}
    for path in member["calls"]:
        call = json.loads(raw(path))
        with np.load(io.BytesIO(raw(path[:-5]+"_state.npz")),allow_pickle=False) as z:
            obs = digest(np.ascontiguousarray(z["observation"]).tobytes())
            base = digest(np.ascontiguousarray(z["base_action"]).tobytes())
        assert obs==call["state_hash"] and call["learning_method"]==item["arm"]
        key = (call["step"],obs,base)
        assert key not in calls
        calls[key] = call
    assert sum(c["public_queries"] for c in calls.values())==summary["n1_forecasts"]
    return {"item":item,"steps":steps,"calls":calls,"summary":summary}


def attributed(frame,model):
    a,b = frame["outcomes"],model["outcomes"]
    fx,lx = [x["pool_id"] for x in a],[x["pool_id"] for x in b]
    if any(frame[k]!=model[k] for k in PUBLIC):
        return "public_setup_differs",None
    if len(lx)>len(fx) or fx[:len(lx)]!=lx:
        return "paid_prefix_differs",None
    if any(any(x[k]!=y[k] for k in FEEDBACK) for x,y in zip(a,b)):
        return "shared_feedback_differs",None
    diff = len(a)-len(b)
    assert len(a)==frame["public_queries"] and len(b)==model["public_queries"]
    cause = "same_queries"
    if diff:
        assert all(x["phase"]=="local_completion" for x in a[len(b):])
        d = model["learned_decisions"][-1]
        assert d["optional_extension"] and not model["engineering_fallback"]
        assert d["paid_queries"]+(len(d["next_ids"]) if d["action"] else 0)==len(b)
        if d["action"]:
            assert d["parent_CONT"] is True
            cause = "one_batch_limit_after_parent_CONT"
        elif not d["parent_CONT"]:
            cause = "parent_STOP_after_native_STOP"
        else:
            assert model["learning_method"]=="BOUNDED261" and d["residual_logit"]<0
            cause = "residual_STOP"
    full = next((r for r in a if r["pool_id"]==frame["chosen_pool_id"]),None)
    short = next((r for r in b if r["pool_id"]==model["chosen_pool_id"]),None)
    return cause,{"saved_queries":diff,"same_candidate":frame["chosen_pool_id"]==model["chosen_pool_id"],
        "frame_candidate":frame["chosen_pool_id"],"learned_candidate":model["chosen_pool_id"],
        "frame_winner_reveal_position":None if full is None else fx.index(full["pool_id"])+1,
        "learned_queries":len(b),"frame_queries":len(a),
        "frame_winner_in_skipped_tail":full is not None and full["pool_id"] in fx[len(b):],
        "instant_reward_loss":reward(full)-reward(short) if full is not None and short is not None else None,
        "last_native_parent_cont":model["learned_decisions"][-1].get("parent_CONT") if model["learned_decisions"] else None}


def compare(frame,model):
    fc,lc = frame["calls"],model["calls"]
    exact = fc.keys() & lc.keys()
    known = []
    other = Counter()
    for key in sorted(exact):
        status,row = attributed(fc[key],lc[key])
        if row is None:
            other[status]+=1
        else:
            known.append({"step":key[0],"state_hash":key[1],"base_hash":key[2],"stop_mechanism":status,**row})
    counts = Counter()
    for row in known:
        counts[row["stop_mechanism"]]+=row["saved_queries"]
    total = frame["summary"]["n1_forecasts"]-model["summary"]["n1_forecasts"]
    unmatched_frame = sum(fc[k]["public_queries"] for k in fc.keys()-lc.keys())
    unmatched_model = sum(lc[k]["public_queries"] for k in lc.keys()-fc.keys())
    setup_diff_delta = sum(fc[k]["public_queries"]-lc[k]["public_queries"] for k in exact
                           if not any(x["step"]==k[0] and x["state_hash"]==k[1] and x["base_hash"]==k[2] for x in known))
    assert total==sum(counts.values())+unmatched_frame-unmatched_model+setup_diff_delta
    first = None
    for f,m in zip(frame["steps"],model["steps"]):
        if f["action_hash"]!=m["action_hash"]:
            assert f["step"]==m["step"]
            first = {"step":f["step"],"same_before_state":f["before_hash"]==m["before_hash"],
                     "frame_cost_at_step":f["raw_cost"],"learned_cost_at_step":m["raw_cost"],
                     "matched_calls_at_step":[r for r in known if r["step"]==f["step"]]}
            break
    return {"week":model["item"]["week"],"arm":model["item"]["arm"],"repeat":model["item"]["repeat"],
        "frame_calls":len(fc),"learned_calls":len(lc),"identical_input_calls":len(exact),"qualified_calls":len(known),
        "input_match_exclusions":dict(other),"attributed_queries":dict(counts),"total_query_difference":total,
        "unmatched_frame_queries":unmatched_frame,"unmatched_learned_queries":unmatched_model,
        "noncomparable_matched_input_delta":setup_diff_delta,"qualified":known,"first_delivered_difference":first}


def audit():
    assert not OUT.exists() and not git("status","--porcelain").strip()
    head = git("rev-parse","HEAD").decode().strip()
    assert git("ls-remote","lntano-src","refs/heads/main").decode().split()[0]==head
    for path in (SCOPE,Path(__file__)):
        verify(path,head)
    scope = read(SCOPE)
    start = time.perf_counter()
    save(OUT/"started.json",{"source_commit":head,"wall_cap_s":600})
    for path,sha in scope["input_sha256"].items():
        assert digest((R/path).read_bytes())==sha
    batches = defaultdict(dict)
    with zipfile.ZipFile(Path(scope["raw_archive"])/"raw_objects.zip") as archive:
        for member in scope["members"]:
            assert time.perf_counter()-start<600
            item = member["item"]
            batches[(item["week"],item["repeat"])][item["arm"]]=load_run(member,archive)
    rows = []
    for _,runs in sorted(batches.items()):
        for arm in LEARNED:
            rows.append(compare(runs["FRAME_GUARD"],runs[arm]))
    assert len(rows)==16
    save(OUT/"paired_call_readout.json",rows)
    result = {"runs":24,"calls":184,"arm_week_repeat_pairs":len(rows),"new_physics":0,
              "new_predictions":0,"new_fits":0,"wall_s":time.perf_counter()-start}
    assert result["wall_s"]<600
    save(OUT/"summary.json",result)
    for row in rows:
        first=row["first_delivered_difference"]
        print(row["week"],row["repeat"],row["arm"],row["qualified_calls"],row["total_query_difference"],
              row["attributed_queries"],row["unmatched_frame_queries"]-row["unmatched_learned_queries"],
              None if first is None else (first["step"],first["same_before_state"]))
    

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("freeze","audit"))
    args=parser.parse_args()
    try:
        {"freeze":freeze,"audit":audit}[args.phase]()
    except Exception:
        if (OUT/"started.json").exists() and not (OUT/"summary.json").exists():
            save(OUT/"readout_error.json",{"traceback":traceback.format_exc()})
        raise
