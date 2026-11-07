"""Already-paid 0.9 guard tails: measure action-preserving room at 64-query checkpoints."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import subprocess
import time

R = Path("D:/Codex/GRL研究")
I = R / "work/GRL-results-integration"
A = Path(__file__).resolve().parent
O = R / "outputs/grid323_guarded_tail_headroom"
NAMES = {"GRID320": ("GRID320_load_guard_closed_loop", "grid320_load_guard_closed_loop"),
         "GRID322": ("GRID322_new_family_load_guard", "grid322_new_family_load_guard")}
BATCH = 64


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path, value):
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(I), *args])


def freeze():
    assert not O.exists()
    head = git("rev-parse", "HEAD").decode().strip()
    sources, runs = {}, []
    for stage, (protocol, output) in NAMES.items():
        p = A.parent / protocol
        folder = p / "results/closed_loop"
        files = [p / "design.json", folder / "paired_results.json", folder / "raw_records_manifest.json",
                 folder / "finished.json"]
        for f in files:
            data = f.read_bytes()
            assert git("show", head + ":" + f.relative_to(I).as_posix()) == data
            sources[f.relative_to(R).as_posix()] = sha(data)
        report = read(folder / "paired_results.json")
        design = read(p / "design.json")
        assert design["load_guard_threshold"] == .9
        refs = {x["path"]: x["sha256"] for x in read(folder / "raw_records_manifest.json")["entries"]}
        grouped = defaultdict(list)
        for rel, digest in sorted(refs.items()):
            if re.fullmatch(r"call[0-9]{3,}\.json", Path(rel).name):
                state = rel[:-5] + "_state.npz"
                grouped[Path(rel).parent.as_posix()].append((rel, digest, state, refs[state]))
        for row in report["records"]:
            item = design["execution_order"][row["index"]]
            if not item["load_guard"]:
                continue
            rel = f"outputs/{output}/runs/r{item['repeat']}__{item['week']}__{item['method']}"
            runs.append({"stage":stage, **{k:row[k] for k in ("week","arm","repeat","pool","candidate_queries",
                                                           "guard_extra_queries","guard_events")},
                         "calls":grouped[rel]})
    assert len(runs) == 24 and sum(len(r["guard_events"]) for r in runs) == 48
    save(A / "fixed_scope.json", {
        "source_hashes":sources,"runs":runs,"batch":BATCH,
        "comparison":"Paid initial incumbent and unchanged original pending order. Report the full-guard winner, fixed first min(64,n) winner, exact hindsight earliest winning prefix, and first original 64-boundary preserving the same winner. Zero prefix means original incumbent already won; report a minimum-one-batch oracle separately.",
        "identity":"Count every event and de-duplicate exact date/step/observation/base. Assert duplicate paid context and candidate feedback agree; retain every date.",
        "interpretation":"Hindsight earliest winning prefix is an upper bound on savings at these inputs, not an available stopping rule. The fixed-64 arm is only a descriptive diagnostic on saved data. No model or alternate threshold.",
        "limits":{"wall_s":600,"new_physics":0,"new_fits":0,"new_predictions":0},
    })


def verified(path, digest):
    data = (R / path).read_bytes()
    assert sha(data) == digest, path
    return data


def reward(row):
    return struct.unpack("f", struct.pack("f",row["reward"]))[0]


def winner(rows):
    return max((r for r in rows if r["strict_admissible"]),key=lambda r:(reward(r),-r["pool_id"]))


def evaluate(call,event):
    paid = event["paid_before"]
    outcomes = call["outcomes"]
    assert call["load_guard"]["extra_ids"] == event["extra_ids"]
    assert event["threshold"] == .9 and event["rho"] > .9
    assert all(r["phase"] == "local_completion" for r in outcomes[paid:])
    pending = [r["pool_id"] for r in outcomes[paid:]]
    assert pending == event["extra_ids"] and pending
    assert len(set(r["pool_id"] for r in outcomes)) == len(outcomes) == call["public_queries"]
    incumbent = winner(outcomes[:paid])
    fixed = winner(outcomes[:paid+BATCH])
    full = winner(outcomes)
    selected = full["pool_id"]
    assert incumbent["pool_id"] == event["incumbent"] and incumbent["rho"] == event["rho"]
    assert call["chosen_pool_id"] == selected
    earliest = 0 if selected == incumbent["pool_id"] else pending.index(selected)+1
    boundary = min(len(pending), ((earliest+BATCH-1)//BATCH)*BATCH)
    assert winner(outcomes[:paid+boundary])["pool_id"] == selected
    return dict(incumbent=incumbent["pool_id"], full_winner=selected, fixed64_winner=fixed["pool_id"],
                pending=len(pending), earliest=earliest, boundary=boundary, rho=incumbent["rho"],
                full_reward_gain=reward(full)-reward(incumbent),
                fixed64_reward_loss=reward(full)-reward(fixed),
                fixed64_saved=max(0,len(pending)-BATCH),
                oracle_candidate_saved=len(pending)-earliest,
                oracle_boundary_saved=len(pending)-boundary,
                oracle_after_one_batch_saved=len(pending)-max(boundary,min(BATCH,len(pending))))


def summarize(rows):
    return dict(events=len(rows), pending=sum(r["pending"] for r in rows),
        changed_winners=sum(r["full_winner"]!=r["incumbent"] for r in rows),
        fixed64_misses=sum(r["full_winner"]!=r["fixed64_winner"] for r in rows),
        fixed64_strict_losses=sum(r["fixed64_reward_loss"]>0 for r in rows),
        **{k:sum(r[k] for r in rows) for k in ("fixed64_saved","oracle_candidate_saved",
               "oracle_boundary_saved","oracle_after_one_batch_saved")})


def audit():
    import numpy as np
    assert not O.exists()
    t = time.monotonic()
    s = read(A / "fixed_scope.json")
    assert s["batch"] == BATCH
    for path, digest in s["source_hashes"].items():
        verified(path,digest)
    rows, distinct = [], {}
    for run in s["runs"]:
        assert time.monotonic()-t < s["limits"]["wall_s"]
        events = {(e["step"],e["state_hash"]):e for e in run["guard_events"]}
        paid, extra = 0,0
        for rel,h,state,sh in run["calls"]:
            call = json.loads(verified(rel,h))
            paid += call["public_queries"]
            e = events.pop((call["step"],call["state_hash"]),None)
            assert bool(e) == ("load_guard" in call)
            if not e:
                continue
            with np.load(io.BytesIO(verified(state,sh)),allow_pickle=False) as z:
                assert sha(z["observation"].tobytes()) == e["state_hash"]
                base = sha(z["base_action"].tobytes())
            r = evaluate(call,e)
            extra += r["pending"]
            key = (run["week"],e["step"],e["state_hash"],base)
            payload = sha(json.dumps([[row[k] for k in ("pool_id","rho","reward","strict_admissible","phase","combined_hash")]
                                      for row in call["outcomes"]],separators=(",",":")).encode())
            out = {"week":key[0],"step":key[1],"observation_hash":key[2],"base_hash":base,**r}
            if key in distinct:
                assert distinct[key]==(out,payload),key
            else:
                distinct[key]=(out,payload)
            rows.append({**out,**{k:run[k] for k in ("stage","arm","repeat","pool")},"source":rel})
        assert not events and paid==run["candidate_queries"] and extra==run["guard_extra_queries"]
    assert len(rows)==48
    clean = [r for r,_ in distinct.values()]
    canonical = [r for r in s["runs"] if r["repeat"]==0 and r["arm"] in ("BOUNDED261","BOUNDED261_GUARD")]
    assert len(canonical)==8
    assert sum(r["guard_extra_queries"] for r in canonical)==sum(r["pending"] for r in clean)
    O.mkdir(exist_ok=False)
    save(O/"rows.json",rows)
    save(O/"unique_states.json",sorted(clean,key=lambda r:(r["week"],r["step"])))
    save(O/"summary.json",dict(all_records=summarize(rows),unique_states=summarize(clean),
        by_date={w:summarize([r for r in clean if r["week"]==w]) for w in sorted({r["week"] for r in canonical})},
        reference_eight_week_candidate_queries=sum(r["candidate_queries"] for r in canonical),
        elapsed_s=time.monotonic()-t, new_physics=0,new_predictions=0,new_fits=0))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("freeze","audit"))
    {"freeze":freeze,"audit":audit}[parser.parse_args().stage]()
