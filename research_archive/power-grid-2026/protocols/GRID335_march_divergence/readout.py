"""Read two audited GRID334 March records and reconcile changes in query and cost."""
import gzip
import hashlib
import json
import math
from pathlib import Path

A = Path(__file__).resolve().parent
R = A.parents[5]
P = A.parent / "GRID334_logeipc_closed_loop/results"
O = R / "outputs/grid335_march_divergence"
REGIONS = ((1, 519), (520, 520), (521, 547), (548, 2017))
COST_KEYS = ("losses_mwh", "redispatch_mwh", "curtailment_delta_mwh", "storage_throughput_mwh")
FIELDS = ("step", "before_hash", "after_hash", "action_hash", "raw_cost", "done", "complete")


def load(path):
    return json.loads(path.read_bytes())


def verified(path, hashes):
    data = (R / path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == hashes[path], path
    return data


def read_arm(record, hashes):
    arm = record["arm"]
    suffix = arm + ("__f0_s0" if arm.startswith("BOUNDED261") else "")
    base = f"outputs/grid334_logeipc_closed_loop/runs/r0__{record['week']}__{suffix}/"
    steps = [json.loads(line) for line in gzip.decompress(verified(base + "steps.jsonl.gz", hashes)).splitlines()]
    calls = [json.loads(verified(path, hashes)) for path in sorted(hashes)
             if path.startswith(base + "call") and path.endswith(".json")]
    assert len(steps) == record["steps"] == 2017
    assert len({c["step"] for c in calls}) == len(calls)
    assert math.fsum(row["raw_cost"] for row in steps) == record["cost"]
    assert sum(c["public_queries"] for c in calls) == record["candidate_queries"]
    assert sum(s["simulations"] for s in steps) == record["total_forecasts"]
    return steps, {c["step"]: c for c in calls}


def regional(steps, calls, lo, hi):
    rows = steps[lo - 1:hi]
    selected = [c for step, c in calls.items() if lo <= step <= hi]
    cost = math.fsum(r["raw_cost"] for r in rows)
    components = {key: math.fsum(r["ledger"]["marginal_cost"] * r["ledger"][key] for r in rows)
                  for key in COST_KEYS}
    local = sum(c["local_queries"] for c in selected)
    total = sum(c["public_queries"] for c in selected)
    return dict(calls=len(selected), candidates=total, global_candidates=total-local,
                local_candidates=local, forecasts=sum(r["simulations"] for r in rows),
                cost=cost, **components, raw_minus_component_sum=cost-math.fsum(components.values()))


def main():
    delivery = load(P / "delivery_hashes.json")
    for name in ("raw_records_manifest.json", "paired_results.json"):
        path = P / name
        verified(path.relative_to(R).as_posix(), {path.relative_to(R).as_posix(): delivery[name]})
    archive = load(P / "raw_records_manifest.json")
    assert archive["passed"] and archive["independent_object_readback"]
    hashes = {r["path"]: r["sha256"] for r in archive["entries"]}
    records = {r["arm"]: r for r in load(P / "paired_results.json")["records"]
               if r["week"] == "2039-03-14_100"}
    a, ac = read_arm(records["BOUNDED261_GUARD_DC"], hashes)
    b, bc = read_arm(records["LOGEIPC_DC"], hashes)
    assert [r["step"] for r in a] == [r["step"] for r in b] == list(range(1, 2018))
    assert all(all(x[k] == y[k] for k in FIELDS) for x, y in zip(a[:519], b[:519], strict=True))
    assert a[519]["before_hash"] == b[519]["before_hash"] and a[519]["action_hash"] != b[519]["action_hash"]
    both = [lo for lo in ac.keys() & bc.keys() if ac[lo]["state_hash"] == bc[lo]["state_hash"]]
    assert both == [520]
    assert 547 == max(ac.keys() ^ bc.keys())
    c, d = ac[520], bc[520]
    assert c["outcomes"][:192] != [] and len(d["outcomes"]) == 192
    feedback_keys = ("pool_id","rho","reward","strict_admissible","combined_hash")
    assert [[r[k] for k in feedback_keys] for r in c["outcomes"][:192]] == [
        [r[k] for k in feedback_keys] for r in d["outcomes"]]
    assert c["chosen_pool_id"] == 238 and c["pre_extra_choice"] == d["chosen_pool_id"] == 142
    assert c.get("load_guard") is None and c["learned_decisions"][0]["action"]
    assert not c["learned_decisions"][0]["optional_extension"]
    bands = []
    for lo, hi in (*REGIONS, (1, 2017)):
        x, y = regional(a, ac, lo, hi), regional(b, bc, lo, hi)
        bands.append(dict(start=lo, end=hi, bounded261=x, logeipc=y, delta={k:y[k]-x[k] for k in x}))
    totals = bands[-1]["delta"]
    assert totals["global_candidates"] == 256 and totals["local_candidates"] == -105
    assert totals["candidates"] == 151 and totals["forecasts"] == 212
    calls = []
    for step in sorted(ac.keys() | bc.keys()):
        row = {"step":step}
        for key, mapping in (("bounded261", ac), ("logeipc", bc)):
            if step in mapping:
                call = mapping[step]
                row[key] = {k:call[k] for k in ("state_hash","public_queries","local_queries","chosen_pool_id")}
        calls.append(row)
    result = dict(parent="4c7cb42a", common_prefix_steps=519, same_state_search_steps=both,
        later_matching_after_hash_steps=[x["step"] for x,y in zip(a[520:], b[520:], strict=True)
                                         if x["after_hash"] == y["after_hash"]],
        first_split={"step":520, "original_native_continue":True, "original_local_guard":False,
                     "original_winner":238,"logeipc_winner":142,
                     "logeipc_max_log_ei_over_fee":d["logeipc"]["final_log_ei_over_fee"]},
        bands=bands, calls=calls, archive_hash=delivery["raw_records_manifest.json"],
        new_simulations=0)
    O.mkdir(exist_ok=False)
    with (O/"summary.json").open("x",encoding="utf8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()

