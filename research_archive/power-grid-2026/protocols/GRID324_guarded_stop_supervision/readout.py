"""One paid-data supervision audit for action-preserving stop after a guarded batch."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import time

R = Path("D:/Codex/GRL研究")
I = R / "work/GRL-results-integration"
A = Path(__file__).resolve().parent
O = R / "outputs/grid324_guarded_stop_supervision"
TEACHER = A.parent / "GRID139_global_feasibility/publication"
REAL = A.parent / "GRID323_guarded_tail_headroom"
SUBS = R / "outputs/grid28/offline.json"
BATCH, THRESHOLD, WALL_CAP = 64, .9, 600
PUBLIC = ("order","aliases","eligible","representatives","masked_ids","rho_limit","parameters","order_mode")
FEEDBACK = ("pool_id","rho","reward","strict_admissible","done","illegal","ambiguous","exceptions","combined_hash")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path, data):
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ref(path):
    return path.relative_to(R).as_posix()


def tracked(path, head):
    data = subprocess.check_output(["git","-C",str(I),"show",head+":"+path.relative_to(I).as_posix()])
    assert data == path.read_bytes(),path


def freeze():
    assert not O.exists()
    head = subprocess.check_output(["git","-C",str(I),"rev-parse","HEAD"]).decode().strip()
    deps = (TEACHER/"call_ledger.json",TEACHER/"source_hashes.json",REAL/"fixed_scope.json",
            REAL/"results/unique_states.json",REAL/"results/rows.json")
    for path in deps:
        tracked(path,head)
    old_hashes = read(TEACHER/"source_hashes.json")
    teacher = []
    for row in read(TEACHER/"call_ledger.json"):
        p = row["source"]
        assert sha((R/p).read_bytes()) == old_hashes[p]
        state = p[:-5]+"_state.npz"
        teacher.append({**{k:row[k] for k in ("call_id","scenario","date","step","partition","source",
                                              "M","global_n","global_choice","local_queries","frame_choice")},
                        "source_sha":old_hashes[p],"state":state,"state_sha":sha((R/state).read_bytes())})
    assert len(teacher)==305 and sum(r["partition"]=="train" for r in teacher)==243
    records = {x[0]:x[1:] for run in read(REAL/"fixed_scope.json")["runs"] for x in run["calls"]}
    rows = read(REAL/"results/rows.json")
    match = {(r["week"],r["step"],r["observation_hash"],r["base_hash"]):r["source"] for r in rows}
    real = []
    for row in read(REAL/"results/unique_states.json"):
        p = match[row["week"],row["step"],row["observation_hash"],row["base_hash"]]
        digest,state,state_sha = records[p]
        real.append({**row,"source":p,"source_sha":digest,"state":state,"state_sha":state_sha})
    assert len(real)==16
    save(A/"fixed_scope.json",{
        "teacher_members":teacher,"real_members":real,
        "source_hashes":{ref(p):sha(p.read_bytes()) for p in (*deps,SUBS)},
        "rule":"Use original 64-query boundaries after at least one purchased batch and strictly before tail exhaustion. CAN_STOP iff current strict-admissible best candidate ID equals the final guard winner; otherwise MUST_CONTINUE. Report exact candidate tie-break and reward gap separately.",
        "teacher_proxy":"Read original GRID139 FRAME global incumbent and lock station. Use only above .9, tail >64 as a proxy stratum. Original learned STOP and paid local prefix are not presumed; even an attractive proxy date remains separate.",
        "join":"Real guard to FULL by scenario, exact physical step, observation and base hashes. Verify public setup and all paid guard candidate feedback fields, compare by ID since FULL order is global. Matched labels remain the same physical states.",
        "gate":"Confirmed actual checkpoint labels must cover each action class on at least two dates to allocate distinct training and evaluation dates. Teacher proxy labels never satisfy this gate. No fitting, inference or physical simulation.",
        "limits":{"wall_s":WALL_CAP,"new_fits":0,"new_predictions":0,"new_physics":0}})


def verified(path,digest):
    raw = (R/path).read_bytes()
    assert sha(raw)==digest,path
    return raw


def hashes(member):
    import numpy as np
    with np.load(io.BytesIO(verified(member["state"],member["state_sha"])),allow_pickle=False) as z:
        return sha(z["observation"].tobytes()),sha(z["base_action"].tobytes())


def key(row):
    return struct.unpack("f",struct.pack("f",row["reward"]))[0],-row["pool_id"]


def best(rows):
    return max((r for r in rows if r["strict_admissible"]),key=key,default=None)


def labels(prefix,tail):
    full = best(prefix+tail)
    if not tail:
        return []
    assert full is not None
    rows = []
    for n in range(BATCH,len(tail),BATCH):
        visible = best(prefix+tail[:n])
        assert visible is not None
        rows.append(dict(paid_in_guard=n,remaining=len(tail)-n,visible_winner=visible["pool_id"],
                         final_winner=full["pool_id"],can_stop=visible["pool_id"]==full["pool_id"],
                         strict_reward_gap=key(full)[0]-key(visible)[0]))
    return rows


def count(rows):
    day = defaultdict(lambda:Counter(can_stop=0,must_continue=0))
    for r in rows:
        for x in r["checkpoints"]:
            day[r["date"]]["can_stop" if x["can_stop"] else "must_continue"] += 1
    return dict(states=len(rows),active_states=sum(bool(r["checkpoints"]) for r in rows),
                can_stop=sum(d["can_stop"] for d in day.values()),
                must_continue=sum(d["must_continue"] for d in day.values()),
                can_stop_dates=sum(d["can_stop"]>0 for d in day.values()),
                must_continue_dates=sum(d["must_continue"]>0 for d in day.values()),
                by_date=dict(sorted(day.items())))


def audit():
    t = time.monotonic()
    s = read(A/"fixed_scope.json")
    for path,digest in s["source_hashes"].items():
        verified(path,digest)
    subs = read(SUBS)["public_subids"]
    actual = {}
    for r in s["real_members"]:
        c = json.loads(verified(r["source"],r["source_sha"]))
        ob,base = hashes(r)
        assert (ob,base)==(r["observation_hash"],r["base_hash"]) and c["state_hash"]==ob
        g = c["load_guard"]
        paid = g["paid_before"]
        prefix,tail = c["outcomes"][:paid],c["outcomes"][paid:]
        assert g["threshold"]==THRESHOLD and g["rho"]>THRESHOLD
        assert [x["pool_id"] for x in tail]==g["extra_ids"] and len(tail)==r["pending"]
        assert best(prefix)["pool_id"]==r["incumbent"] and best(prefix+tail)["pool_id"]==r["full_winner"]
        identity = r["week"],r["step"],ob,base
        assert identity not in actual
        actual[identity]=(c,dict(week=r["week"],date=r["week"].split("_")[0],step=r["step"],
             state_hash=ob,base_hash=base,source=r["source"],pending=len(tail),
             checkpoints=labels(prefix,tail),joins=[]))
    real_steps = {(week,step) for week,step,_,_ in actual}
    teacher_rows = []
    mismatches = Counter()
    for m in s["teacher_members"]:
        assert time.monotonic()-t<WALL_CAP
        c = json.loads(verified(m["source"],m["source_sha"]))
        paid = c["outcomes"]
        by_id = {x["pool_id"]:x for x in paid}
        assert list(by_id)==c["order"] and len(paid)==m["M"] and m["step"]==c["step"]
        n = m["global_n"]
        prefix = paid[:n]
        inc = best(prefix)
        assert (None if inc is None else inc["pool_id"])==m["global_choice"]
        tail = []
        if m["local_queries"]:
            station = subs[inc["pool_id"]]
            representatives = {rep for alias,rep in c["aliases"].items() if subs[int(alias)]==station}
            tail = [x for x in paid[n:] if x["pool_id"] in representatives]
            assert len(tail)==m["local_queries"] and best(prefix+tail)["pool_id"]==m["frame_choice"]
        proxy = bool(inc and inc["rho"]>THRESHOLD and len(tail)>BATCH)
        row = {**{k:m[k] for k in ("call_id","scenario","date","step","partition","source")},
               "proxy":proxy,"pending":len(tail),"incumbent_rho":None if inc is None else inc["rho"],
               "checkpoints":labels(prefix,tail) if proxy else []}
        teacher_rows.append(row)
        if (m["scenario"],m["step"]) not in real_steps:
            continue
        ob,base = hashes(m)
        assert ob==c["state_hash"]
        identity = m["scenario"],m["step"],ob,base
        if identity not in actual:
            mismatches["same_step_different_state_or_base"]+=1
            continue
        real,record = actual[identity]
        if any(c[k]!=real[k] for k in PUBLIC):
            status="public_setup_differs"
        elif any(x["pool_id"] not in by_id or any(x[k]!=by_id[x["pool_id"]][k] for k in FEEDBACK)
                 for x in real["outcomes"]):
            status="paid_feedback_differs"
        else:
            status="exact_paid_match"
        record["joins"].append(dict(call_id=m["call_id"],status=status,partition=m["partition"]))
    actual_rows = [v for _,v in actual.values()]
    joined = [r for r in actual_rows if any(x["status"]=="exact_paid_match" for x in r["joins"])]
    by_partition = {p:count([r for r in teacher_rows if r["partition"]==p and r["proxy"]])
                    for p in ("train","validation")}
    real_count = count(actual_rows)
    enough = min(real_count["can_stop_dates"],real_count["must_continue_dates"])>=2
    O.mkdir(exist_ok=False)
    save(O/"actual_states.json",actual_rows)
    save(O/"teacher_states.json",teacher_rows)
    save(O/"summary.json",dict(actual=real_count,exact_full_matches=count(joined),
        other_full_joins=dict(mismatches),teacher_proxy=by_partition,teacher_calls=len(teacher_rows),
        passed_necessary_gate=enough,new_fits=0,new_predictions=0,new_physics=0,elapsed_s=time.monotonic()-t))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("freeze","audit"))
    {"freeze":freeze,"audit":audit}[parser.parse_args().stage]()
