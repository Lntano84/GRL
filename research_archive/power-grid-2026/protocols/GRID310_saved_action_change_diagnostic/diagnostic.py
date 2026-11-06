"""Read archived public STOP fields and existing pair receipts, with no live runtime."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import time
import traceback

import importlib.util

A = Path(__file__).resolve().parent
P = A.parent
spec = importlib.util.spec_from_file_location("grid308_frozen_score", P/"GRID308_action_opportunity_plan/plan.py")
PLAN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PLAN)
R, I, read, save, ref, git, digest, score = (PLAN.R, PLAN.I, PLAN.read, PLAN.save, PLAN.ref, PLAN.git,
                                           PLAN.digest, PLAN.score)
O = R/"outputs/grid310_saved_action_change_diagnostic"
Q303 = P/"GRID303_saved_suffix_targets"
Q307 = P/"GRID307_unstarted_calibration_cont"
Q308 = P/"GRID308_action_opportunity_plan"
SCOPE305 = P/"GRID305_saved_calibration_candidates/results/proposed_scope.json"
CALENDAR = R/"outputs/grid300_new_training_dates/calendar.json"
CATALOG = R/"outputs/grid301_remaining_collection/baseline_raw_manifest.json"
PUBLISHED_CATALOG = P/"GRID301_remaining_collection/baseline_public/baseline_raw_manifest.json"
RECEIPTS = {303:Q303/"results/pairs.json",307:Q307/"label_results/pairs.json"}
WALL_CAP_S = 600


def rel(path):
    return path.relative_to(R).as_posix()


def verified_git(path,head):
    assert git("show",head+":"+path.relative_to(I).as_posix())==path.read_bytes(),path


def freeze():
    assert not O.exists() and not (A/"fixed_scope.json").exists()
    head = git("rev-parse","HEAD").decode().strip()
    inputs = [Q303/"fixed_inputs.json",Q307/"execution_order.json",Q308/"fixed_plan.json",Q308/"plan.py",
              *RECEIPTS.values(),SCOPE305,PUBLISHED_CATALOG]
    for path in inputs:
        verified_git(path,head)
    assert CATALOG.read_bytes()==PUBLISHED_CATALOG.read_bytes()
    catalog = read(CATALOG)
    cal_only = read(SCOPE305)["summary"]["cal_only_dates"]
    family = {row["date"]:row["original_role_slot"] for row in read(CALENDAR)["dates"]}
    rows, excluded = [], Counter()

    def add(source,item,base,path,call_ref,decision_index=None):
        common = item["target"]["common"]
        ids = common["next_ids"]
        assert item["target"]["ordinary261_continue"] is False and item["arm"]=="CONT_NEXT"
        assert 1 <= len(ids) <= 64 and len(set(ids))==len(ids)
        year = item["date"][:4]
        rows.append({"source":source,"index":item["index"],"date":item["date"],
            "family":family[item["date"]] if year=="2038" else item["date"],
            "pool":item["pool"],"role":item["partition"],"KEEP":item["KEEP"],
            "strict_cal":item["date"] in cal_only[year],"baseline_folder":base,
            "call":path,"call_ref":call_ref,"decision_index":decision_index,
            "common":{k:common[k] for k in ("step","state_hash","batch","next_ids","native_q",
                                         "raw41_hash","public_NN_input_hash")}})

    all303 = read(Q303/"fixed_inputs.json")["members"]
    assert len(all303)==233
    for pair in all303:
        item = pair["opposite"]
        if "partial" in pair:
            excluded["GRID303_paid_partial"] += 1
            continue
        if item["target"]["ordinary261_continue"]:
            excluded["GRID303_original_CONT"] += 1
            continue
        base = pair["baseline"]["folder"]
        assert item["baseline_folder"]==base
        common = item["target"]["common"]
        path = base+f"/call{common['call_index']:03d}.json"
        add(303,item,base,path,{k:catalog[path][k] for k in ("bytes","sha256")})
    old307 = {row["index"]:row for row in read(Q308/"fixed_plan.json")["qualification_scope"]["old_saved_public_calls"]}
    for item in read(Q307/"execution_order.json"):
        saved = old307[item["index"]]
        add(307,item,item["baseline_folder"],saved["call"],saved["reference"],saved["decision_index"])
    excluded["GRID306_paid_partial"] = 1
    assert sum(row["source"]==303 for row in rows)+excluded["GRID303_paid_partial"]+excluded["GRID303_original_CONT"]==233
    assert sum(row["source"]==307 for row in rows)==16
    assert len({(row["source"],row["index"]) for row in rows})==len(rows)
    save(A/"fixed_scope.json",{"prior_head":head,"members":rows,"excluded":dict(excluded),
         "provenance":{rel(p):ref(p) for p in (*inputs,CALENDAR,CATALOG)},
         "score":"Exact GRID308 lexicographic unseen topology count, then maximum live-terminal pair distance.",
         "primary":"Active pools, one representative per (date,step,state_hash) by source, pool, index. Compare unequal scores within (date,batch_size).",
         "sensitivity":["same baseline and same batch_size","strict date-disjoint calibration only"],
         "outcome":"Archived delivered joint action changed at trigger, regardless of eventual completion. Cost signs only when both arms have complete 2017-step clean ledgers.",
         "decision_rule":"A reason to review new collection needs at least two net-positive date families and zero net-negative families in primary, at least two net-positive and zero net-negative families in same-baseline sensitivity, and at least one positive and zero negative family in strict calibration. Diagnostic only, no fit qualification.",
         "receipt_outcomes_used_for_freeze":False,"wall_cap_s":WALL_CAP_S})
    print(json.dumps({"members":len(rows),"by_source":dict(Counter(r["source"] for r in rows)),
                      "excluded":dict(excluded)}))


def field(row,cache):
    path = R/row["call"]
    if path not in cache:
        assert ref(path)==row["call_ref"]
        cache[path] = read(path),row["call_ref"]
    call, expected = cache[path]
    assert expected==row["call_ref"]
    common = row["common"]
    assert (call["step"],call["state_hash"],call["model_pool"])==(common["step"],common["state_hash"],row["pool"])
    decisions = call["learned_decisions"]
    if row["decision_index"] is not None:
        d = decisions[row["decision_index"]]
    else:
        d, = [d for d in decisions if d["optional_extension"] and d["batch"]==common["batch"]]
    assert d["optional_extension"] and type(d["action"]) is int and d["action"]==0
    assert d["batch"]==common["batch"] and d["next_ids"]==common["next_ids"]
    assert d["native_q"]==common["native_q"] and d["native_q"][1]<=d["native_q"][0]
    assert digest(d["raw_features"])==common["raw41_hash"]
    public = d["NN_input"]
    assert digest(public)==common["public_NN_input_hash"]
    assert len(public["next_action"])==len(common["next_ids"])
    return score(public)


def outcome(row,saved):
    common = row["common"]
    assert (saved["date"],saved["pool"],saved["trigger_step"],saved["state_hash"],saved["KEEP"])==(
        row["date"],row["pool"],common["step"],common["state_hash"],row["KEEP"])
    if row["source"]==303:
        assert saved["ordinary261_continue"] is False and saved["opposite_arm"]=="CONT_NEXT"
        assert saved["known_batch_queries"]==len(common["next_ids"]) and saved["role"]==row["role"]
        complete = saved["stop_complete"] and saved["cont_complete"]
    else:
        complete = saved["complete_both"]
    changed = saved["trigger_action_changed"]
    assert type(changed) is bool and type(saved["clean_both"]) is bool
    clean_complete = complete and saved["clean_both"] and saved["stop_steps"]==saved["cont_steps"]==2017
    value = saved["cost_cont_minus_stop"]
    if clean_complete:
        assert type(value) in (int,float)
    return changed,value if clean_complete else None


def compare(rows,scope):
    groups = defaultdict(list)
    for row in rows:
        key = (row["date"],row["batch_size"])
        if scope=="same_baseline":
            key += (row["baseline_folder"],)
        groups[key].append(row)
    totals = Counter(states=len(rows),date_batch_groups=len(groups))
    by_date = defaultdict(Counter)
    by_family = defaultdict(Counter)
    details = []
    for key,members in sorted(groups.items()):
        local = Counter()
        for x,y in combinations(members,2):
            if x["score"]==y["score"]:
                name = "score_tie"
            elif x["changed"]==y["changed"]:
                name = "same_outcome"
            else:
                name = "higher_changed" if (x["score"]>y["score"])==x["changed"] else "lower_changed"
            local[name] += 1
        if local:
            totals.update(local)
            totals["groups_with_multiple_states"] += 1
            by_date[key[0]].update(local)
            by_family[members[0]["family"]].update(local)
            details.append({"group":list(key),"states":len(members),"pairs":dict(local)})
    positive = sorted(k for k,c in by_family.items() if c["higher_changed"]>c["lower_changed"])
    negative = sorted(k for k,c in by_family.items() if c["lower_changed"]>c["higher_changed"])
    return {"totals":dict(totals),"by_date":{k:dict(v) for k,v in sorted(by_date.items())},
            "by_family":{k:dict(v) for k,v in sorted(by_family.items())},"positive_families":positive,
            "negative_families":negative,"groups":details}


def cost_table(rows):
    return {label:dict(Counter(
        "unavailable" if row["cost_cont_minus_stop"] is None else
        "lower" if row["cost_cont_minus_stop"]<0 else "higher" if row["cost_cont_minus_stop"]>0 else "same"
        for row in rows if row["changed"]==changed)) for label,changed in (("changed",True),("unchanged",False))}


def audit():
    assert not O.exists()
    assert not git("status","--porcelain").strip()
    head = git("rev-parse","HEAD").decode().strip()
    assert git("ls-remote","lntano-src","refs/heads/main").decode().split()[0]==head
    for path in (Path(__file__),A/"fixed_scope.json"):
        verified_git(path,head)
    scope = read(A/"fixed_scope.json")
    start = time.perf_counter()
    save(O/"started.json",{"source_commit":head,"wall_cap_s":WALL_CAP_S,"no_retry":True})
    for path,value in scope["provenance"].items():
        assert ref(R/path)==value
    cache, rows = {}, []
    for member in scope["members"]:
        assert time.perf_counter()-start<WALL_CAP_S
        value = field(member,cache)
        c = member["common"]
        rows.append({k:member[k] for k in ("source","index","date","family","pool","role","KEEP","strict_cal","baseline_folder")}|
                    {"step":c["step"],"state_hash":c["state_hash"],"batch_size":len(c["next_ids"]),"score":list(value)})
    save(O/"public_scores.json",rows)
    # Join precomputed outcomes only after the public features have been saved.
    outcomes = {n:{row["index"]:row for row in read(path)} for n,path in RECEIPTS.items()}
    for member,row in zip(scope["members"],rows):
        changed,value = outcome(member,outcomes[member["source"]][member["index"]])
        row.update(changed=changed,cost_cont_minus_stop=value)
    save(O/"joined_rows.json",rows)
    active = [row for row in rows if not row["KEEP"]]
    states = defaultdict(list)
    for row in sorted(active,key=lambda x:(x["source"],x["pool"],x["index"])):
        states[(row["date"],row["step"],row["state_hash"])].append(row)
    representatives = [group[0] for group in states.values()]
    primary = compare(representatives,"primary")
    same_base = compare(representatives,"same_baseline")
    calibration = [row for row in representatives if row["strict_cal"]]
    cal = compare(calibration,"strict_cal")
    useful = (len(primary["positive_families"])>=2 and not primary["negative_families"]
              and len(same_base["positive_families"])>=2 and not same_base["negative_families"]
              and bool(cal["positive_families"]) and not cal["negative_families"])
    result = {"rows":len(rows),"calls_read":len(cache),"active_rows":len(active),"keep_rows":len(rows)-len(active),
        "unique_active_states":len(representatives),"duplicate_active_records":len(active)-len(representatives),
        "duplicate_state_groups_with_action_disagreement":sum(len({x["changed"] for x in g})>1 for g in states.values()),
        "active_dates":len({r["date"] for r in representatives}),"strict_cal_states":len(calibration),
        "representative_costs":cost_table(representatives),"strict_cal_costs":cost_table(calibration),
        "source_rows":{str(n):sum(r["source"]==n for r in rows) for n in RECEIPTS},
        "unseen_count_differs_from_batch_size":sum(r["score"][0]!=r["batch_size"] for r in rows),
        "primary":primary,"same_baseline":same_base,"strict_cal":cal,
        "decision":"REVIEW_NEW_COLLECTION_ONLY" if useful else "CLOSE_TOPOLOGY_SCORE_SAMPLING_ROUTE",
        "new_physics":0,"new_predictions":0,"new_fits":0,"new_cost_labels":0,"wall_s":time.perf_counter()-start}
    assert result["wall_s"]<WALL_CAP_S
    save(O/"summary.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("primary","same_baseline","strict_cal")},ensure_ascii=False))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("freeze","audit"))
    args = parser.parse_args()
    try:
        {"freeze":freeze,"audit":audit}[args.phase]()
    except Exception:
        if (O/"started.json").exists() and not (O/"summary.json").exists():
            save(O/"readout_error.json",{"traceback":traceback.format_exc()})
        raise
