"""One fixed paid-batch probe on the same GRID313 matched tails."""
import argparse
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
O = R/"outputs/grid315_one_batch_probe"
S = A/"fixed_scope.json"
PAYLOAD = P/"results/paired_call_readout.json"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as file:
        json.dump(value,file,ensure_ascii=False,indent=2,allow_nan=False)
        file.write("\n")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git","--no-optional-locks","-C",str(I),*args])


def verify(path,head):
    assert git("show",head+":"+path.relative_to(I).as_posix())==path.read_bytes(),path


def freeze():
    assert not S.exists() and not O.exists()
    head = git("rev-parse","HEAD").decode().strip()
    fixed = P/"fixed_scope.json"
    for path in (fixed,PAYLOAD):
        verify(path,head)
    source = read(fixed)
    cases = [{"week":r["week"],**x} for r in read(PAYLOAD)
        if r["repeat"]==0 and r["arm"]=="PARENT208"
        for x in r["qualified"] if x["stop_mechanism"]=="parent_STOP_after_native_STOP" and x["saved_queries"]>0]
    assert len(cases)==9 and len({(x["week"],x["step"],x["state_hash"]) for x in cases})==9
    weeks = {x["week"] for x in cases}
    calls = {m["item"]["week"]:{p:m["files"][p] for p in m["calls"]}
        for m in source["members"] if m["item"]["arm"]=="FRAME_GUARD" and m["item"]["repeat"]==0
        and m["item"]["week"] in weeks}
    assert set(calls)==weeks
    save(S,{"rule":"At each frozen parent STOP, buy the original first min(64,n) candidates; if maximum strict-admissible float32 reward is strictly better, buy the rest. Otherwise return the new best among paid candidates. Same order and original ID tie-break.",
        "qualification":"Reject as fixed replacement if any original FRAME winner is still lost or if the observed remaining query savings are negligible. Previously exposed calls, no policy validation.",
        "cases":cases,"calls":calls,"raw_archive":source["raw_archive"],
        "inputs":{p.relative_to(R).as_posix():sha(p.read_bytes()) for p in (fixed,PAYLOAD)},
        "new_physics":0,"new_predictions":0,"new_fits":0,"wall_cap_s":600})


def reward(row):
    return struct.unpack("f",struct.pack("f",row["reward"]))[0]


def best(rows):
    return max((r for r in rows if r["strict_admissible"]),key=lambda r:(reward(r),-r["pool_id"]))


def evaluate(case,call):
    outcomes = call["outcomes"]
    paid = case["learned_queries"]
    tail = case["saved_queries"]
    assert call["public_queries"]==len(outcomes)==paid+tail==case["frame_queries"]
    assert all(row["phase"]=="local_completion" for row in outcomes[paid:])
    q = min(64,tail)
    incumbent = best(outcomes[:paid])
    first = best(outcomes[:paid+q])
    full = best(outcomes)
    assert (incumbent["pool_id"],full["pool_id"])==(case["learned_candidate"],case["frame_candidate"])
    assert reward(full)-reward(incumbent)==case["instant_reward_loss"]
    opens = reward(first)>reward(incumbent)
    chosen = full if opens else first
    used = tail if opens else q
    return {k:case[k] for k in ("week","step","state_hash")}|{
        "original_saved_queries":tail,"first_batch_queries":q,"continue_tail":opens,
        "extra_queries_vs_learned_stop":used,"remaining_savings_vs_frame":tail-used,
        "learned_candidate":incumbent["pool_id"],"first_batch_candidate":first["pool_id"],
        "rule_candidate":chosen["pool_id"],"frame_candidate":full["pool_id"],
        "original_learner_lost_winner":incumbent["pool_id"]!=full["pool_id"],
        "probe_lost_winner":chosen["pool_id"]!=full["pool_id"],
        "first_batch_reward_gain":reward(first)-reward(incumbent),
        "remaining_reward_loss":reward(full)-reward(chosen)}


def audit():
    assert not O.exists() and not git("status","--porcelain").strip()
    head = git("rev-parse","HEAD").decode().strip()
    assert git("ls-remote","lntano-src","refs/heads/main").decode().split()[0]==head
    for p in (S,Path(__file__)):
        verify(p,head)
    start = time.perf_counter()
    save(O/"started.json",{"source_commit":head,"wall_cap_s":600})
    scope = read(S)
    for p,h in scope["inputs"].items():
        assert sha((R/p).read_bytes())==h
    cases = {(r["week"],r["step"],r["state_hash"]):r for r in scope["cases"]}
    result = {}
    with zipfile.ZipFile(Path(scope["raw_archive"])/"raw_objects.zip") as archive:
        for week,refs in scope["calls"].items():
            for name,ref in refs.items():
                assert time.perf_counter()-start<600
                path = R/name
                raw = path.read_bytes() if path.is_file() else archive.read(ref["object"])
                assert len(raw)==ref["bytes"] and sha(raw)==ref["sha256"]
                call = json.loads(raw)
                key = (week,call["step"],call["state_hash"])
                if key in cases:
                    assert key not in result and call["learning_method"]=="FRAME_GUARD"
                    result[key]=evaluate(cases[key],call)
    assert set(result)==set(cases)
    rows = [result[k] for k in sorted(result)]
    summary = {"cases":len(rows)}
    for field in ("original_saved_queries","first_batch_queries","continue_tail","extra_queries_vs_learned_stop",
                  "remaining_savings_vs_frame","original_learner_lost_winner","probe_lost_winner"):
        summary[field] = sum(r[field] for r in rows)
    assert summary["original_saved_queries"]==summary["extra_queries_vs_learned_stop"]+summary["remaining_savings_vs_frame"]
    summary.update(new_physics=0,new_predictions=0,new_fits=0,wall_s=time.perf_counter()-start)
    save(O/"rows.json",rows)
    save(O/"summary.json",summary)
    for row in rows:
        print(row["week"],row["step"],row["original_saved_queries"],row["extra_queries_vs_learned_stop"],
              row["first_batch_reward_gain"],row["probe_lost_winner"])
    print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("freeze","audit"))
    {"freeze":freeze,"audit":audit}[parser.parse_args().phase]()
