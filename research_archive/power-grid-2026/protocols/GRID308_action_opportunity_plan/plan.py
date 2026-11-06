"""Public-field feasibility check and frozen proposal. Stdlib only."""
import argparse
from collections import Counter
from datetime import date, timedelta
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import subprocess
import time

R = Path("D:/Codex/GRL研究")
I = R/"work/GRL-results-integration"
A = Path(__file__).resolve().parent
O = R/"outputs/grid308_action_opportunity_plan"
P = A.parent/"GRID307_unstarted_calibration_cont"
C = R/"outputs/grid300_new_training_dates"
SCOPE305 = P.parent/"GRID305_saved_calibration_candidates/results/proposed_scope.json"
ARCHIVE301 = R/"outputs/grid301_remaining_collection/baseline_raw_manifest.json"
REPORT300 = C/"GRID300_generation_stop_report.md"
POOL_REFS = R/"outputs/grid301_remaining_collection/model_references.json"
FOUR_MONTHS = (3,6,9,12)
SEED_TAG = "GRL308_ACTION_OPPORTUNITY_V1"
N_REPLAYS = 12
WINDOW = 7200
GEN_WINDOW = 2700


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def ref(path):
    data = path.read_bytes()
    return {"bytes":len(data), "sha256":sha256(data).hexdigest()}


def git(*args):
    return subprocess.check_output(["git","--no-optional-locks","-C",str(I),*args])


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def signature(vector):
    if len(vector) != 68:
        raise ValueError("Expected 4 planes of 17 planned devices")
    bits, delta, active, present = [tuple(vector[i:i+17]) for i in range(0,68,17)]
    if (not all(x in (0,1) for x in (*bits,*active,*present))
        or not all(x in (-1,0,1) for x in delta)
        or any((b and not a) or (a and not p) for b,a,p in zip(bits,active,present))):
        raise ValueError("Public topology signature is malformed")
    return bits,active,present


def pair_distance(plan, anchor):
    b,a,p = plan
    ib,ia,ip = anchor
    if p != ip:
        raise ValueError("Candidate and incumbent use different device layouts")
    # Count changes in same-bus membership for terminals live in both plans.
    common = [i for i in range(17) if a[i] and ia[i]]
    return sum((b[i] == b[j]) != (ib[i] == ib[j]) for i,j in combinations(common,2))


def score(public):
    anchor = signature(public["anchor_action"])
    visible = {anchor, *(signature(x) for x in public["context_action"])}
    next_plans = {signature(x) for x in public["next_action"]}
    unseen = next_plans-visible
    return len(unseen),max((pair_distance(p,anchor) for p in unseen),default=0)


def second_monday(year,month):
    first = date(year,month,1)
    return first+timedelta(days=(-first.weekday())%7+7)


def calendar():
    old38 = read(C/"calendar.json")["dates"]
    old37 = [date.fromisoformat(x["original_role_slot"]) for x in old38]
    known_slots = {(x["month"],x["monday_ordinal"]) for x in old38}
    assert all((month,2) not in known_slots for month in FOUR_MONTHS)
    available = []
    for month in FOUR_MONTHS:
        d = second_monday(2039,month)
        assert all(abs((d-old).days)>=7 for old in old37+[date.fromisoformat(x["date"]) for x in old38])
        iso = d.isoformat()
        keys = ("load","renewable","forecast")
        seeds = [int.from_bytes(sha256(f"{SEED_TAG}|{iso}|{key}".encode()).digest()[:4],"little")%(2**32-1)
                 for key in keys]
        available.append({"date":iso,"family":{"month":month,"monday_ordinal":2},"scenario_id":100,
                          "scenario":iso+"_100","environment_seed":0,"model_pool":"f0_s0",
                          "role":"prospective_calibration_pilot_only",
                          "generator_seeds":dict(zip(("load_seed","renew_seed","gen_p_forecast_seed"),seeds))})
    return available


def freeze():
    """Bind only the paid public calls already matched in GRID307."""
    assert not O.exists() and not (A/"fixed_plan.json").exists()
    prior = read(SCOPE305)["fixed_representatives"]
    catalog38 = read(ARCHIVE301)
    members = []
    for item in read(P/"execution_order.json"):
        j = item["index"]
        assert 1 <= j <= 16
        common = item["target"]["common"]
        if item["year"] == 2037:
            fixed = prior[j]
            path = fixed["source_call"]
            expected = item["baseline_input_refs"][path]
            decision_index = fixed["decision_index"]
        else:
            assert item["year"] == 2038
            path = item["baseline_folder"]+f"/call{common['call_index']:03d}.json"
            expected = {k:catalog38[path][k] for k in ("bytes","sha256")}
            decision_index = None
        assert ref(R/path)==expected
        members.append({"index":j,"year":item["year"],"date":item["date"],"pool":item["pool"],
                        "call":path,"reference":expected,"decision_index":decision_index,
                        "common":{k:common[k] for k in ("step","state_hash","batch","next_ids","public_NN_input_hash")}})
    assert [m["index"] for m in members] == list(range(1,17))
    model = next(m for m in read(POOL_REFS) if m["pool"]=="f0_s0")
    for path,value in model["hashes"].items():
        assert ref(R/path)["sha256"]==value
    fixed = {
        "status":"DESIGN_ONLY_NOT_IMPLEMENTED_NOT_APPROVED",
        "calendar":calendar(),"qualification_scope":{
            "old_saved_public_calls":members,"wall_cap_s":600,
            "new_physics":0,"new_predictions":0,"new_cost_labels":0,"new_fits":0},
        "baseline_rule":"One frozen original261 baseline per calendar row. Record all original261 STOP triggers with a nonempty next batch and no engineering fallback. Freeze target marker and whole joint prefix before any CONT.",
        "eligible_rule":"Only native STOP followed by ordinary261 STOP, 1..64 original next ids, valid archived model-input topology, and at least one unseen planned topology. Deduplicate exact physical states by (date,step,before_hash) using chronological order.",
        "score_rule":"Lexicographic maximum of unseen planned topology count and maximum change in live terminal pair same-bus membership versus incumbent. Exact integer score; earliest (step,call,batch) for ties. No observed suffix outcome enters this rule.",
        "reference_rule":"Earliest different eligible state in the same baseline with the same next batch length and a strictly lower score. Record absence; no substitution across dates or batches.",
        "minimum_coverage_before_CONT":"Primary points in at least three of four fixed date families, and matched references in at least two. Otherwise retain base records, no CONT.",
        "generation_scenarios":4,"baselines":4,"max_cont_replays":8,"max_total_replays":N_REPLAYS,
        "caps":{"wall_s":WINDOW,"generation_s":GEN_WINDOW,"per_generation_s":1800,"per_replay_s":1200,
                "new_physical_steps":2017*N_REPLAYS,"all_forecasts":180000},
        "historical_resource_estimate":{
            "generation_wrapper_s":5051.926463*4/16,
            "replay_wall_s":read(R/"outputs/grid307_unstarted_calibration_cont/closeout.json")["wall_s"]*N_REPLAYS/16,
            "estimated_replay_forecasts":110427*N_REPLAYS/16,
            "estimate_includes_publication_wait":False,"new_year_live_qualified":False},
        "result_interpretation":"Report each family, both selected strata, delivery action changes, both-arm complete/clean outcomes and original suffix raw_cost deltas. Stop at scope or first engineering/resource guard. No extra scenario, tuning, or training from this pilot.",
        "model_reference":model,
        "provenance":{p.relative_to(R).as_posix():ref(p) for p in
            [P/"execution_order.json",SCOPE305,C/"calendar.json",ARCHIVE301,REPORT300,POOL_REFS,
             R/"outputs/grid307_unstarted_calibration_cont/closeout.json"]},
    }
    save(A/"fixed_plan.json",fixed)


def audit():
    """Inspect only predecision fields. No label joins and no simulator."""
    assert not O.exists()
    plan = read(A/"fixed_plan.json")
    head = git("rev-parse","HEAD").decode().strip()
    assert not git("status","--porcelain").strip()
    assert git("ls-remote","lntano-src","refs/heads/main").decode().split()[0]==head
    for path in (A/"fixed_plan.json",Path(__file__)):
        assert git("show",head+":"+path.relative_to(I).as_posix())==path.read_bytes()
    start = time.perf_counter()
    save(O/"started.json",{"source_commit":head,"wall_cap_s":600})
    for path,expected in plan["provenance"].items():
        assert ref(R/path)==expected
    outputs = []
    for item in plan["qualification_scope"]["old_saved_public_calls"]:
        assert time.perf_counter()-start<600
        path = R/item["call"]
        raw = path.read_bytes()
        assert len(raw)==item["reference"]["bytes"] and sha256(raw).hexdigest()==item["reference"]["sha256"]
        call = json.loads(raw)
        common = item["common"]
        assert (call["step"],call["state_hash"],call["model_pool"])==(common["step"],common["state_hash"],item["pool"])
        decisions = call["learned_decisions"]
        if item["decision_index"] is not None:
            d = decisions[item["decision_index"]]
        else:
            d, = [d for d in decisions if d["optional_extension"] and d["batch"]==common["batch"]]
        assert d["optional_extension"] and d["action"]==0 and d["native_q"][1]<=d["native_q"][0]
        assert d["batch"]==common["batch"] and d["next_ids"]==common["next_ids"]
        public = d["NN_input"]
        assert digest(public)==common["public_NN_input_hash"]
        assert 1<=len(public["next_action"])==len(common["next_ids"])<=64
        opportunity = score(public)
        outputs.append({k:item[k] for k in ("index","date","pool")}|
                       {"batch_size":len(common["next_ids"]),"score":list(opportunity),"eligible":opportunity[0]>0})
    distinct_scores = {tuple(row["score"]) for row in outputs}
    same_size_varies = sorted(n for n in {r["batch_size"] for r in outputs} if len({
        tuple(r["score"]) for r in outputs if r["batch_size"]==n})>1)
    summary = {"source_commit":head,"archived_public_calls_verified":len(outputs),
               "eligible":sum(r["eligible"] for r in outputs),"distinct_scores":len(distinct_scores),
               "batch_sizes":dict(sorted(Counter(r["batch_size"] for r in outputs).items())),
               "batch_sizes_with_score_variation":same_size_varies,
               "all_score_counts_equal_batch_size":all(r["batch_size"]==r["score"][0] for r in outputs),
               "rows":outputs,"new_fits":0,"new_predictions":0,"new_physics":0,"new_cost_labels":0,
               "wall_s":time.perf_counter()-start}
    save(O/"public_field_review.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="rows"},ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("freeze","audit"))
    {"freeze":freeze,"audit":audit}[parser.parse_args().phase]()
