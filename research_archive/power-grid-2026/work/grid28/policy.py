"""Semantic completion: finish the incumbent's substation before stopping."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "grid27"))
from policy import staged, key


def local_complete(order, observe, policy, subids, aliases, safe=.9):
    result = staged(order, observe, policy, safe)
    chosen = result["chosen_pool_id"]
    seen = {r["pool_id"] for r in result["outcomes"]}
    # Safe prefix and exhausted/no-choice cases remain unchanged.
    if chosen is None or not result["fallback"] or len(seen) == len(order):
        result.update(local_substation=None, local_queries=0)
        return result
    sub = int(subids[chosen])
    targets = {int(rep) for i,rep in aliases.items() if int(subids[int(i)])==sub}
    incumbent = next(r for r in result["outcomes"] if r["pool_id"]==chosen)
    added = 0
    for i in order:
        if i not in targets or i in seen:continue
        row = observe(i,"local_completion")
        assert row["pool_id"]==i
        result["outcomes"].append(row); seen.add(i); added+=1
        if row["strict_admissible"] and key(row)>key(incumbent):incumbent=row
    result.update(chosen_pool_id=incumbent["pool_id"], public_queries=len(seen),
                  local_substation=sub,local_queries=added,
                  stop_reason=result["stop_reason"]+"_local_complete")
    result["events"].append({"queries":len(seen),"phase":"local_completion", "chosen":incumbent["pool_id"],
                              "improved":incumbent["pool_id"]!=chosen})
    return result


def hand_checks():
    # Late winner at the same substation is recovered; a remote late winner is not.
    order=list(range(400)); subids=[0]*400; aliases={i:i for i in order}
    def feedback(i,phase):return {"pool_id":i,"reward":2. if i==399 else 1.,"rho":1.,"strict_admissible":True,"phase":phase}
    r=local_complete(order,feedback,"PATIENCE64",subids,aliases)
    assert r["chosen_pool_id"]==399 and r["public_queries"]==400 and r["local_queries"]==144
    subids[-1]=1
    r=local_complete(order,feedback,"PATIENCE64",subids,aliases)
    assert r["chosen_pool_id"]==0 and r["public_queries"]==399
    assert len({x["pool_id"] for x in r["outcomes"]})==r["public_queries"]
    return 3
