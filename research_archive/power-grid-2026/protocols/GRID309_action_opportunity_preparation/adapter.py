"""Pure GRID308 scoring and immutable baseline/CONT selection."""
import gzip
import io
import json
from pathlib import Path
import sys
from runtime import LEGACY, OUT, PLAN, R, STATE, module, read, checked_bytes


def position(row):
    marker = row["marker"]["common"]
    return marker["step"], marker["call_index"], marker["batch"]


def order_candidates(candidates):
    unique = {}
    for row in sorted(candidates,key=position):
        common = row["marker"]["common"]
        if row["score"][0] <= 0:
            continue
        unique.setdefault((common["step"],common["state_hash"]),row)
    return list(unique.values())


def select(records):
    selected, dates = [], []
    for record in records:
        candidates = order_candidates(record["opportunities"])
        entry = {"date":record["date"],"baseline_index":record["index"],
                 "eligible_states":len(candidates),"primary":None,"reference":None}
        if candidates:
            best = min(candidates,key=lambda x:(-x["score"][0],-x["score"][1],position(x)))
            entry["primary"] = best
            length = len(best["marker"]["common"]["next_ids"])
            entry["reference"] = next((row for row in candidates
                if row["score"] < best["score"] and len(row["marker"]["common"]["next_ids"]) == length),None)
        dates.append(entry)
    primary = sum(d["primary"] is not None for d in dates)
    reference = sum(d["reference"] is not None for d in dates)
    opened = primary >= 3 and reference >= 2
    if opened:
        for d in dates:
            for role in ("primary","reference"):
                if d[role] is not None:
                    selected.append((d["baseline_index"],role,d[role]))
    return {"coverage_passed":opened,"primary_families":primary,"reference_families":reference,
            "dates":dates},selected


class PilotState(STATE.State):
    def __init__(self,item,prefix=None):
        super().__init__(item,prefix)
        self.opportunities = []

    def force(self,raw,native,ordinary):
        result = super().force(raw,native,ordinary)
        if self.target is None and ordinary["continue_"] is False:
            value = PLAN.score(native["NN_input"])
            self.opportunities.append({"marker":self.candidates[-1],"score":list(value)})
        return result

    def record(self):
        return {**super().record(),"opportunities":self.opportunities}


def install(Search,item,models,expected=None):
    assert expected is None
    prefix = None
    if item["target"] is not None:
        rel = item["baseline_folder"]+"/steps.jsonl.gz"
        data = checked_bytes(R/rel,read(OUT/"baseline_raw_manifest.json")[rel])
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
            prefix = []
            for _ in range(item["target"]["common"]["step"]-1):
                row = json.loads(stream.readline())
                prefix.append({k:row[k] for k in STATE.KEYS})
        assert STATE.digest(prefix) == item["target"]["common"]["prefix_signature"]
        assert sum(row["raw_cost"] for row in prefix) == item["target"]["common"]["prefix_cost"]
    state = PilotState(item,prefix)
    sys.path.insert(0,str(LEGACY))
    prior = module("grid309_original_adapter",LEGACY/"adapter.py")
    prior.EpisodeIntervention = lambda arm,unused:state
    assert prior.install(Search,item,models,None) is state
    return state
