"""Derive suffix-cost targets once from GRID307 published saved ledgers."""
from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback

R = Path("D:/Codex/GRL研究")
I = R/"work/GRL-results-integration"
O = R/"outputs/grid307_unstarted_calibration_cont"
A = I/"research_archive/power-grid-2026/protocols/GRID307_unstarted_calibration_cont"
OUT = O/"saved_cost_readout"
PRIOR = R/"work/grid298_saved_suffix_targets/readout.py"
spec = importlib.util.spec_from_file_location("original_ledger_audit",PRIOR)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
read, save, git = audit.load, audit.save, audit.git
FILES = ("summary.json","metadata.json","intervention.json","steps.jsonl.gz")
FIXED = A/"label_source/fixed_inputs.json"


def rel(path):
    return path.relative_to(R).as_posix()


def reference(path):
    data = path.read_bytes()
    return {"bytes":len(data),"sha256":audit.sha(data)}


def archive_catalog(path):
    record = read(path)
    assert record["passed"] and (record.get("independent_object_readback") or record.get("objects_independently_readback"))
    return {row["path"]:{key:row[key] for key in ("sha256","bytes")} for row in record["entries"]}


def prepare():
    """Bind immutable raw manifests and membership without reading cost ledgers."""
    assert not FIXED.exists()
    manifest = O/"manifest.json"
    close = read(O/"closeout.json")
    records = {x["index"]:x for x in read(manifest)["records"]} if manifest.exists() else {}
    items = read(A/"execution_order.json")
    assert list(records) == list(range(1,1+close["accepted_new"]))
    assert [item["index"] for item in items] == list(range(1,17))
    catalog37 = R/"archives/GRID297_paired_collection_20261102/raw_records/raw_records_manifest.json"
    catalog38 = R/"outputs/grid301_remaining_collection/baseline_raw_manifest.json"
    refs = {2037:archive_catalog(catalog37),2038:read(catalog38)}
    new_catalog = O/"raw_records_manifest.json"
    new_refs = archive_catalog(new_catalog)
    scope = R/"outputs/grid305_saved_calibration_candidates/proposed_scope.json"
    prior = read(scope)
    pairs, inputs = [], {}
    for item in items:
        index = item["index"]
        fixed = prior["fixed_representatives"][index]
        assert item["date"] in prior["summary"]["cal_only_dates"][str(item["year"])]
        assert item["partition"] == "own_training_cal4"
        pair = {"item":item,"family":fixed["family"],"accepted":index in records}
        if pair["accepted"]:
            assert records[index]["summary"]["index"] == index
            baseline = item["baseline_folder"]
            cont = rel(O/"runs"/f"r{item['repeat']}__{item['week']}__{item['method']}")
            for folder, catalog in ((baseline,refs[item["year"]]),(cont,new_refs)):
                for name in FILES:
                    path = folder+"/"+name
                    inputs[path] = {k:catalog[path][k] for k in ("bytes","sha256")}
                    if path in item["baseline_input_refs"]:
                        assert inputs[path] == item["baseline_input_refs"][path]
            pair["cont_folder"] = cont
        pairs.append(pair)
    paths = [A/"execution_order.json",O/"closeout.json",scope,catalog37,catalog38,new_catalog,PRIOR]
    if manifest.exists():
        paths.append(manifest)
    save(FIXED,{"pairs":pairs,"files":inputs,"metadata":{rel(p):reference(p) for p in paths},
                "paid_partial_index0_excluded":True})


def check_pair(pair, base, cont):
    bs,bm,be,br,ba = base
    cs,cm,ce,cr,ca = cont
    item = pair["item"]
    marker = item["target"]
    expected = marker["common"]
    got = ce["event"]
    assert marker["ordinary261_continue"] is False
    assert got["ordinary261_continue"] is False and got["forced_continue"] is True
    assert ce["prefix_matched"] and got["arm"] == cs["arm"] == "CONT_NEXT"
    assert all(got["common"].get(k) == value for k,value in expected.items())
    if item["year"] == 2038:
        assert got["common"] == expected and marker in be["candidates"]
    elif be["event"]:
        old = be["event"]
        assert old["forced_continue"] == old["ordinary261"]["continue_"]
    for key in ("pool","date","week","partition","KEEP"):
        assert bs[key] == cs[key] == item[key]
    for key in ("library_hashes","checkpoint_sha256","cost_per_MW","dt_hours","cadence"):
        assert bm[key] == cm[key],(item["index"],key)
    t = expected["step"]
    assert len(br) == 2017 and bs["complete"] and 1 <= t <= len(cr)
    assert all(all(x[k] == y[k] for k in audit.KEYS) for x,y in zip(br[:t-1],cr[:t-1]))
    prefix = [{k:row[k] for k in audit.KEYS} for row in br[:t-1]]
    assert audit.digest(prefix) == expected["prefix_signature"]
    assert sum(row["raw_cost"] for row in prefix) == expected["prefix_cost"]
    assert br[t-1]["before_hash"] == cr[t-1]["before_hash"] == expected["state_hash"]
    assert got["delivered"] == {k:cr[t-1][k] for k in (*audit.KEYS,"restore")}
    clean = not any(x[k] for x in (ba,ca) for k in ("illegal","ambiguous","exception_steps","missing_ledger_rows"))
    complete = len(cr) == 2017 and cs["complete"]
    row = {"status":"INCOMPLETE_OR_UNCLEAN_NO_COST_TARGET","stop_steps":len(br),"cont_steps":len(cr),
           "complete_both":complete,"clean_both":clean,"trigger_action_changed":br[t-1]["action_hash"]!=cr[t-1]["action_hash"],
           "cost_cont_minus_stop":None}
    if complete and clean:
        stop = math.fsum(x["raw_cost"] for x in br[t-1:])
        go = math.fsum(x["raw_cost"] for x in cr[t-1:])
        delta = go-stop
        assert abs(delta-(cs["cost"]-bs["cost"])) <= audit.tolerance(br)+audit.tolerance(cr)
        row.update(status="COMPLETE_MATCHED_COST_TARGET",stop_suffix_cost=stop,cont_suffix_cost=go,
                   cost_cont_minus_stop=delta)
    return row


def run():
    assert not OUT.exists()
    assert not git("status","--porcelain").strip()
    head = git("rev-parse","HEAD").decode().strip()
    assert git("ls-remote","lntano-src","refs/heads/main").decode().split()[0] == head
    for path in (FIXED, Path(__file__), A/"results/raw_records_manifest.json", A/"results/closeout.json"):
        assert git("show",head+":"+path.relative_to(I).as_posix()) == path.read_bytes()
    data = read(FIXED)
    audit.START = time.perf_counter()
    audit.REFS = data["files"]
    save(OUT/"started.json",{"source_commit":head,"wall_cap_s":600,"no_retry":True})
    for path,ref in data["metadata"].items():
        assert reference(R/path) == ref
    labels, bases = [], {}
    for pair in data["pairs"]:
        audit.checktime()
        item = pair["item"]
        j = item["index"]
        common = item["target"]["common"]
        row = {"index":j,"family":pair["family"],"date":item["date"],"pool":item["pool"],
               "KEEP":item["KEEP"],"trigger_step":common["step"],"state_hash":common["state_hash"],
               "status":"NO_ACCEPTED_REPLAY","cost_cont_minus_stop":None}
        if pair["accepted"]:
            base = item["baseline_folder"]
            if base not in bases:
                audit.O = OUT/"baseline_checks"
                bases[base] = audit.audit_episode({"folder":base})
            audit.O = OUT/"continue_checks"
            cont = audit.audit_episode({"folder":pair["cont_folder"],"index":j})
            row.update(check_pair(pair,bases[base],cont))
        labels.append(row)
        save(OUT/"pair_receipts"/f"{j:03d}.json",row)

    def counts(rows):
        delta = [x["cost_cont_minus_stop"] for x in rows if x["cost_cont_minus_stop"] is not None]
        return {"members":len(rows),"valid":len(delta),"lower":sum(x<0 for x in delta),
                "higher":sum(x>0 for x in delta),"same":sum(x==0 for x in delta)}
    groups = defaultdict(list)
    for row in labels:
        groups[row["family"]].append(row)
    out = {"all":counts(labels),"active":counts([r for r in labels if not r["KEEP"]]),
           "KEEP":counts([r for r in labels if r["KEEP"]]),
           "by_family":{k:counts(v) for k,v in sorted(groups.items())},
           "statuses":dict(Counter(r["status"] for r in labels)),
           "new_fits":0,"new_predictions":0,"new_physics":0,"wall_s":time.perf_counter()-audit.START}
    audit.checktime()
    save(OUT/"pairs.json",labels)
    save(OUT/"summary.json",out)
    print(json.dumps(out,ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1:] == ["--prepare"]:
        prepare()
    else:
        assert not sys.argv[1:]
        try:
            run()
        except Exception:
            if (OUT/"started.json").exists() and not (OUT/"summary.json").exists():
                save(OUT/"readout_error.json",{"traceback":traceback.format_exc()})
            raise
