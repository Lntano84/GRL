"""One bounded readout of already-paid optional GRID261 batches."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import time
import traceback
import zipfile

R = Path("D:/Codex/GRL研究")
I = R/"work/GRL-results-integration"
A = Path(__file__).resolve().parent
P = A.parent
O = R/"outputs/grid311_paid_batch_prefix"
SCOPE = A/"fixed_scope.json"
RUN = R/"outputs/grid291_control_preparation"
ARCHIVE = R/"archives/GRID291_control_replay_20261101/raw_records"
ROWS = R/"outputs/grid292_activation_support/public_actual_rows.json"
AUDIT = RUN/"saved_partial_review.json"
MANIFEST = RUN/"raw_records_manifest.json"
DURATION = 600


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")


def ref_bytes(data):
    return {"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}


def ref(path):
    return ref_bytes(path.read_bytes())


def git(*args):
    return subprocess.check_output(["git","--no-optional-locks","-C",str(I),*args])


def verified_git(path,head):
    assert git("show",head+":"+path.relative_to(I).as_posix())==path.read_bytes(),path


def freeze():
    assert not O.exists() and not SCOPE.exists()
    head = git("rev-parse","HEAD").decode().strip()
    published = {
        ROWS:P/"GRID292_activation_support/public/public_actual_rows.json",
        AUDIT:P/"GRID291_control_preparation/budget_limited_results/saved_partial_review.json",
        MANIFEST:P/"GRID291_control_preparation/budget_limited_results/raw_records_manifest.json",
    }
    for path,tracked in published.items():
        verified_git(tracked,head)
        assert ref(path)==ref(tracked)
    receipt = read(AUDIT)
    bounded = {r["index"]:r for r in receipt["records"] if r["arm"]=="BOUNDED261"}
    assert receipt["passed_saved_data"] and len(bounded)==93
    all_rows = read(ROWS)
    assert len(all_rows)==329
    seen = read(MANIFEST)["files"]
    members = []
    excluded = Counter()
    keys = ("index","pool","repeat","week","step","state_hash","batch","next_ids","next_queries",
            "source_call","residual_KEEP","public_signature")
    for row in all_rows:
        assert row["arm"]=="BOUNDED261" and row["optional_extension"] is True and not row["partial"]
        known = bounded[row["index"]]
        assert all(row[key]==known[key] for key in ("pool","week","repeat"))
        assert row["next_queries"]==len(row["next_ids"])
        if not row["parent_CONT"]:
            excluded["parent_STOP"] += 1
        elif not row["action"]:
            excluded["residual_STOP"] += 1
        elif row["next_queries"]==1:
            excluded["single_paid_candidate"] += 1
        else:
            assert 1 < row["next_queries"] <= 64
            path = R/row["source_call"]
            archive_ref = seen[path.relative_to(RUN).as_posix()]
            members.append({**{k:row[k] for k in keys},"reference":archive_ref})
    assert len(members)==25 and sum(excluded.values())+len(members)==329
    assert len({(r["index"],r["source_call"],r["batch"]) for r in members})==len(members)
    dependencies = [*published,*published.values(),
        R/"work/grid101_gain_learning/features.py",R/"work/grid91_mechanism/policies.py",
        R/"work/grid291_control_preparation/policy.py"]
    save(SCOPE,{"members":members,"excluded":dict(excluded),
         "inputs":{p.relative_to(R).as_posix():ref(p) for p in dependencies},
         "baseline_accepted_runs":len(bounded),
         "baseline_candidate_queries":sum(r["candidate_queries"] for r in bounded.values()),
         "baseline_total_forecasts":sum(r["total_queries"] for r in bounded.values()),
         "scope":"All accepted GRID291 BOUNDED261 optional extension decisions with parent CONT, actual CONT and batch length greater than one. KEEP reported separately.",
         "selection_before_reading_candidate_feedback":True,
         "fixed_half":"ceil(n/2) additional candidates in actual reveal order. No reorder, no sweep.",
         "ranking":"strict_admissible, then original float32 reward, then smaller pool_id. Audit original full chosen ID first.",
         "duplicates":"Report all row uses and non-KEEP per (date, step, state_hash) outcomes. Count a physical state once per reported category and retain disagreements.",
         "interpretation":"Earliest full winner is hindsight. A followup online stopping hypothesis requires positive full gain and exact winner retained at half on at least two different dates. Any half losses remain explicit; no deployment from this result.",
         "wall_cap_s":DURATION})
    print(json.dumps({"paid_multi_candidate_rows":len(members),"excluded":dict(excluded)}))


def reward(row):
    return struct.unpack("f",struct.pack("f",row["reward"]))[0]


def best(rows):
    return max((r for r in rows if r["strict_admissible"]),
               key=lambda r:(reward(r),-r["pool_id"]),default=None)


def one(item,call):
    assert (call["step"],call["state_hash"],call["model_pool"],call["learning_method"])==(
        item["step"],item["state_hash"],item["pool"],"BOUNDED261")
    decision, = [row for row in call["learned_decisions"]
                 if row["batch"]==item["batch"] and row["optional_extension"]]
    assert decision["action"]==1 and decision["parent_CONT"] is True
    assert decision["residual_KEEP"]==item["residual_KEEP"] and decision["next_ids"]==item["next_ids"]
    assert call["engineering_fallback"] is None
    assert decision==call["learned_decisions"][-1]
    known,n = decision["paid_queries"],item["next_queries"]
    saved = call["outcomes"]
    assert len(saved)==known+n==call["public_queries"]
    paid_prefix,paid_batch = saved[:known],saved[known:]
    assert [x["pool_id"] for x in paid_batch]==item["next_ids"]
    assert all(x["phase"]=="local_completion" for x in paid_batch)
    assert len({x["pool_id"] for x in saved})==len(saved)
    assert all(math.isfinite(x["reward"]) and math.isfinite(x["forecast_s"]) and x["forecast_s"]>=0 for x in saved)
    initial,full = best(paid_prefix),best(saved)
    assert initial is not None and full is not None
    assert full["pool_id"]==call["chosen_pool_id"]
    half_n = (n+1)//2
    half = best(saved[:known+half_n])
    full_id = full["pool_id"]
    earliest = 0 if full_id==initial["pool_id"] else item["next_ids"].index(full_id)+1
    assert best(saved[:known+earliest])["pool_id"]==full_id
    assert earliest==0 or best(saved[:known+earliest-1])["pool_id"]!=full_id
    assert (half["pool_id"]==full_id)==(earliest<=half_n)
    initial_gain = reward(full)-reward(initial)
    half_gain = reward(half)-reward(initial)
    loss = reward(full)-reward(half)
    assert 0 <= half_gain <= initial_gain and loss>=0
    result = {k:item[k] for k in ("index","pool","repeat","week","step","state_hash","source_call","residual_KEEP")}
    return {**result,"original_batch":n,"half_queries":half_n,"half_queries_saved":n-half_n,
        "initial_candidate":initial["pool_id"],"full_candidate":full_id,"half_candidate":half["pool_id"],
        "full_candidate_first_position":earliest,"half_retains_exact_candidate":half["pool_id"]==full_id,
        "half_same_reward_different_id":loss==0 and half["pool_id"]!=full_id,
        "full_increment":initial_gain,"half_increment":half_gain,"half_loss":loss,
        "half_suffix_saved_forecast_s":math.fsum(x["forecast_s"] for x in paid_batch[half_n:]),
        "hindsight_queries_saved":n-earliest}


def tally(rows):
    return {"uses":len(rows),"full_positive_uses":sum(r["full_increment"]>0 for r in rows),
            "full_winner_already_incumbent_uses":sum(r["full_candidate_first_position"]==0 for r in rows),
            "half_exact_uses":sum(r["half_retains_exact_candidate"] for r in rows),
            "half_tie_other_id_uses":sum(r["half_same_reward_different_id"] for r in rows),
            "half_loss_uses":sum(r["half_loss"]>0 for r in rows),
            "half_positive_gain_exact_uses":sum(r["full_increment"]>0 and r["half_retains_exact_candidate"] for r in rows),
            "full_batch_queries":sum(r["original_batch"] for r in rows),
            "half_queries_saved":sum(r["half_queries_saved"] for r in rows),
            "half_exact_only_hindsight_queries_saved":sum(r["half_queries_saved"] for r in rows if r["half_retains_exact_candidate"]),
            "half_suffix_recorded_forecast_s":math.fsum(r["half_suffix_saved_forecast_s"] for r in rows),
            "full_increment_sum":math.fsum(r["full_increment"] for r in rows),
            "half_loss_sum":math.fsum(r["half_loss"] for r in rows),
            "half_loss_max":max((r["half_loss"] for r in rows),default=0.),
            "hindsight_queries_saved":sum(r["hindsight_queries_saved"] for r in rows)}


def audit():
    assert not O.exists()
    head = git("rev-parse","HEAD").decode().strip()
    assert not git("status","--porcelain").strip()
    assert git("ls-remote","lntano-src","refs/heads/main").decode().split()[0]==head
    for path in (SCOPE,Path(__file__)):
        verified_git(path,head)
    scope = read(SCOPE)
    start = time.perf_counter()
    save(O/"started.json",{"source_commit":head,"wall_cap_s":DURATION,"no_retry":True})
    for path,digest in scope["inputs"].items():
        assert ref(R/path)==digest
    rows = []
    with zipfile.ZipFile(ARCHIVE/"small_files.zip") as archive:
        for item in scope["members"]:
            assert time.perf_counter()-start<DURATION
            path = R/item["source_call"]
            expected = item["reference"]
            if path.exists():
                raw = path.read_bytes()
            else:
                location = expected["location"]
                assert location.startswith("small_files.zip!")
                raw = archive.read(location.split("!",1)[1])
            assert ref_bytes(raw)=={k:expected[k] for k in ("bytes","sha256")}
            row = one(item,json.loads(raw))
            rows.append(row)
            save(O/"receipts"/f"{len(rows)-1:03d}.json",row)
    active = [row for row in rows if not row["residual_KEEP"]]
    keep = [row for row in rows if row["residual_KEEP"]]
    groups = defaultdict(list)
    for row in active:
        groups[(row["week"],row["step"],row["state_hash"])].append(row)
    positive_half = {key for key,g in groups.items() if any(
        x["full_increment"]>0 and x["half_retains_exact_candidate"] for x in g)}
    stable_positive_half = {key for key,g in groups.items() if all(
        x["full_increment"]>0 and x["half_retains_exact_candidate"] for x in g)}
    losses = {key for key,g in groups.items() if any(x["half_loss"]>0 for x in g)}
    dates = sorted({key[0] for key in stable_positive_half})
    result = {"all":tally(rows),"active":tally(active),"KEEP":tally(keep),
              "active_unique_physical_states":len(groups),"positive_half_unique_states_any_use":len(positive_half),
              "positive_half_unique_states_all_uses":len(stable_positive_half),
              "positive_half_dates_all_uses":dates,"half_loss_unique_states":len(losses),
              "half_loss_dates":sorted({key[0] for key in losses}),
              "active_state_groups_with_half_exact_disagreement":sum(len({r["half_retains_exact_candidate"] for r in g})>1 for g in groups.values()),
              "by_week":{week:tally([r for r in active if r["week"]==week]) for week in sorted({r["week"] for r in active})},
              "baseline_candidate_queries":scope["baseline_candidate_queries"],
              "baseline_total_forecasts":scope["baseline_total_forecasts"],
              "decision":"REVIEW_ONLY_IF_ONLINE_STOP_RULE_EXISTS" if len(dates)>=2 else "NO_CROSS_DATE_SUPPORT_FOR_BATCH_STOP",
              "new_physics":0,"new_predictions":0,"new_fits":0,"wall_s":time.perf_counter()-start}
    assert result["wall_s"]<DURATION
    save(O/"rows.json",rows)
    save(O/"summary.json",result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("freeze","audit"))
    args=parser.parse_args()
    try:
        {"freeze":freeze,"audit":audit}[args.phase]()
    except Exception:
        if (O/"started.json").exists() and not (O/"summary.json").exists():
            save(O/"readout_error.json",{"traceback":traceback.format_exc()})
        raise
