"""Check one visible-prefix statistic against fixed GRID313 matched tail cases."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
import zipfile

R = Path("D:/Codex/GRL研究")
I = R/"work/GRL-results-integration"
A = Path(__file__).resolve().parent
P = A.parent/"GRID313_frame_learning_call_attribution"
O = R/"outputs/grid314_visible_stagnation_probe"
M = A/"fixed_scope.json"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as file:
        json.dump(value,file,ensure_ascii=False,indent=2,allow_nan=False)
        file.write("\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git","--no-optional-locks","-C",str(I),*args])


def verify(path,head):
    assert git("show",head+":"+path.relative_to(I).as_posix())==path.read_bytes(),path


def freeze():
    assert not M.exists() and not O.exists()
    head = git("rev-parse","HEAD").decode().strip()
    prior = P/"fixed_scope.json"
    matched = P/"results/paired_call_readout.json"
    for path in (prior,matched):
        verify(path,head)
    source = read(prior)
    runs = {}
    for m in source["members"]:
        if m["item"]["repeat"]==0 and m["item"]["arm"]=="PARENT208":
            runs[m["item"]["week"]] = {path:m["files"][path] for path in m["calls"]}
    cases = [{"week":p["week"],"step":x["step"],"state_hash":x["state_hash"]}
        for p in read(matched) if p["repeat"]==0 and p["arm"]=="PARENT208"
        for x in p["qualified"] if x["stop_mechanism"]=="parent_STOP_after_native_STOP" and x["saved_queries"]>0]
    assert len(cases)==9 and len({(x["week"],x["step"],x["state_hash"]) for x in cases})==9
    save(M,{"cases":cases,"calls_by_week":runs,"archive":source["raw_archive"],
        "inputs":{p.relative_to(R).as_posix():digest(p.read_bytes()) for p in (prior,matched)},
        "hypothesis":"An older last strict-admissible float32 reward improvement indicates a useless tail.",
        "feature":"Number of already paid candidates after the first attainment of the final visible maximum strict-admissible FP32 reward.",
        "fixed_reference":"Age >=128 candidates (two 64-query batches); diagnostic only, no threshold fitting.",
        "diagnostics":"Compare each different-outcome pair overall and separately at equal paid-prefix length; report original parent margin as context, do not use it to rank or select.",
        "selection":"Use repeat0, PARENT208, every exact matched parent-STOP with a skipped tail from GRID313; keep development failures. These labels are already exposed.",
        "wall_cap_s":600,"new_physics":0,"new_predictions":0,"new_fits":0})


def f32(value):
    return struct.unpack("f",struct.pack("f",value))[0]


def public(call):
    d = call["learned_decisions"][-1]
    assert d["optional_extension"] and d["action"]==0 and d["parent_CONT"] is False
    paid = d["paid_queries"]
    feedback = call["outcomes"][:paid]
    assert paid==len(call["outcomes"])==call["public_queries"]
    eligible = [(j,f32(x["reward"]),x["pool_id"],x["rho"]) for j,x in enumerate(feedback,1)
                if x["strict_admissible"]]
    first = max(eligible,key=lambda x:(x[1],-x[0]))
    chosen = max(eligible,key=lambda x:(x[1],-x[2]))
    assert chosen[2]==call["chosen_pool_id"]
    return {"paid":paid,"age":paid-first[0],"first_position_with_best_reward":first[0],
            "winner_first_position":chosen[0],"winner":chosen[2],
            "parent_b":d["parent_b"],"winner_rho":chosen[3],"remaining_before_stop":d["remaining"],
            "original_next_count":len(d["next_ids"])}


def compare(rows):
    out = Counter()
    for no in rows:
        if no["missed_better_winner"]:
            continue
        for yes in rows:
            if not yes["missed_better_winner"]:
                continue
            relation = "supports" if no["age"]>yes["age"] else "opposes" if no["age"]<yes["age"] else "ties"
            out[relation]+=1
    return {k:out[k] for k in ("supports","opposes","ties")}


def audit():
    assert not O.exists()
    head = git("rev-parse","HEAD").decode().strip()
    assert not git("status","--porcelain").strip()
    assert git("ls-remote","lntano-src","refs/heads/main").decode().split()[0]==head
    for p in (M,Path(__file__)):
        verify(p,head)
    d = read(M)
    start = time.perf_counter()
    save(O/"started.json",{"source_commit":head,"wall_cap_s":600})
    for p,h in d["inputs"].items():
        assert digest((R/p).read_bytes())==h
    keys = {(x["week"],x["step"],x["state_hash"]) for x in d["cases"]}
    parsed = {}
    with zipfile.ZipFile(Path(d["archive"])/"raw_objects.zip") as z:
        for week,files in d["calls_by_week"].items():
            for p,ref in files.items():
                assert time.perf_counter()-start<600
                local = R/p
                data = local.read_bytes() if local.is_file() else z.read(ref["object"])
                assert len(data)==ref["bytes"] and digest(data)==ref["sha256"]
                c = json.loads(data)
                k = week,c["step"],c["state_hash"]
                if k in keys:
                    assert k not in parsed and c["learning_method"]=="PARENT208"
                    parsed[k] = public(c)
    assert set(parsed)==keys
    save(O/"public_only.json",[{"week":k[0],"step":k[1],"state_hash":k[2],**p} for k,p in sorted(parsed.items())])
    labels = {(p["week"],x["step"],x["state_hash"]):x
        for p in read(P/"results/paired_call_readout.json") if p["repeat"]==0 and p["arm"]=="PARENT208"
        for x in p["qualified"] if x["stop_mechanism"]=="parent_STOP_after_native_STOP" and x["saved_queries"]>0}
    assert set(labels)==keys
    rows = []
    for k,v in sorted(parsed.items()):
        y = labels[k]
        assert y["instant_reward_loss"] is not None and y["instant_reward_loss"]>=0
        assert y["learned_queries"]==v["paid"] and y["learned_candidate"]==v["winner"]
        assert v["remaining_before_stop"]==y["saved_queries"]
        rows.append({"week":k[0],"step":k[1],"state_hash":k[2],**v,
            "missed_better_winner":y["instant_reward_loss"]>0,"instant_reward_loss":y["instant_reward_loss"],
            "skipped_queries":y["saved_queries"],"same_candidate":y["same_candidate"]})
    selected = [r for r in rows if r["age"]>=128]
    result = {"rows":len(rows),"winners_missed":sum(r["missed_better_winner"] for r in rows),
        "rank_overall":compare(rows),"rank_equal_paid":{str(n):compare([r for r in rows if r["paid"]==n])
                                                       for n in sorted({r["paid"] for r in rows})},
        "fixed_128":{"matches":len(selected),"missed_winners":sum(r["missed_better_winner"] for r in selected)},
        "new_fits":0,"new_predictions":0,"new_physics":0,"wall_s":time.perf_counter()-start}
    assert result["wall_s"]<600
    save(O/"joined_rows.json",rows)
    save(O/"summary.json",result)
    for r in rows:
        print(r["week"],r["step"],r["paid"],r["age"],r["remaining_before_stop"],r["missed_better_winner"])
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("freeze","audit"))
    {"freeze":freeze,"audit":audit}[parser.parse_args().phase]()
