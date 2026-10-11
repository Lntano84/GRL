"""Feedback replay of frozen policies; withheld feedback never enters decision."""
import copy
import csv
import hashlib
import json
import time
from pathlib import Path
from policy import staged, key, hand_checks
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid27"
D = json.loads((OUT / "design.json").read_text())
assert not (OUT / "offline.json").exists()
for p,sha in json.loads((OUT / "schedule_freeze.json").read_text()).items():
    assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == sha
started = time.perf_counter(); rows = []; permission_checks = 0; hashes = {}


def read(p):
    raw = p.read_bytes(); hashes[p.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


for week in D["weeks"]:
    directory = ROOT / "outputs/grid26/runs" / f"r0__{week}__FULL"
    for p in sorted(directory.glob("call[0-9][0-9][0-9].json")):
        full = read(p); truth = {r["pool_id"]:r for r in full["outcomes"]}
        for method in D["rankers"]:
            own = read(directory.parent / f"r0__{week}__{method['method']}" / p.name)
            assert own["state_hash"] == full["state_hash"]
            order = own["order"]
            for policy in D["policies"]:
                seen = []
                def feedback(i, phase):
                    assert i not in seen; seen.append(i)
                    r = dict(truth[i]); r["phase"] = phase; return r
                result = staged(order, feedback, policy)
                assert seen == [r["pool_id"] for r in result["outcomes"]]
                selected = truth.get(result["chosen_pool_id"])
                assert selected is None or selected["strict_admissible"]
                # Perturb every not-dispatched result in a second complete world.
                altered = copy.deepcopy(truth)
                for i in set(truth)-set(seen):
                    altered[i].update(reward=99.,rho=.01,strict_admissible=True)
                again = staged(order, lambda i, phase: {**altered[i],"phase":phase}, policy)
                assert result == again
                permission_checks += 1
                winner = truth.get(full["chosen_pool_id"])
                regret = 0. if winner is None else (None if selected is None else selected["rho"]-winner["rho"])
                rows.append({"week":week,"call":p.name,"step":own["step"],"ranker":method["method"],"policy":policy,
                    "queries":result["public_queries"],"protected_queries":own["public_queries"],
                    "full_queries":full["public_queries"],"choice":result["chosen_pool_id"],"full_choice":full["chosen_pool_id"],
                    "same_full_choice":result["chosen_pool_id"]==full["chosen_pool_id"],
                    "rho_regret":regret,"admissible":selected is not None,"stop_reason":result["stop_reason"],
                    "events":result["events"]})
assert len(rows)==156 and permission_checks==156
summaries=[]
for m in D["rankers"]:
    for policy in D["policies"]:
        group=[r for r in rows if r["ranker"]==m["method"] and r["policy"]==policy]
        protected=sum(r["protected_queries"] for r in group)
        summaries.append({"ranker":m["method"],"policy":policy,"states":len(group),
                          "queries":sum(r["queries"] for r in group),"protected_queries":protected,
                          "query_reduction_to_protected":1-sum(r["queries"] for r in group)/protected,
                          "same_full_choice":sum(r["same_full_choice"] for r in group),
                          "changed_choice":sum(not r["same_full_choice"] for r in group),
                          "max_rho_regret":max((r["rho_regret"] or 0.) for r in group),
                          "march_critical_choice_changed":any(r["week"]=='2035-03-05_0' and r["step"]==1733 and not r["same_full_choice"] for r in group),
                          "missing_full_admissible":sum(r["full_choice"] is not None and r["choice"] is None for r in group)})
result={"passed_engineering":True,"hand_checks":hand_checks(),"permission_checks":permission_checks,
        "summaries":summaries,"rows":rows,"input_hashes":hashes,"wall_s":time.perf_counter()-started,
        "new_forecasts":0,"new_rollouts":0,"new_fits":0,"reserved_evaluations":0,
        "warning":"Developed on exposed states. Rho regret and action coverage are not closed-loop cost, survival or a safe-stopping certificate."}
(OUT / "offline.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
with (OUT / "offline_per_state.csv").open("w",newline="",encoding="utf-8-sig") as f:
    fields=[k for k in rows[0] if k!='events']; writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
    writer.writerows({k:r[k] for k in fields} for r in rows)
(OUT / "RUN_STATE.md").write_text("OFFLINE SCREEN COMPLETE. Engineering passed; pilot gate must be reviewed before AC. No new forecasts/fits/holdout.\n",encoding="utf-8")
print(json.dumps({"summaries":summaries,"hand_checks":result["hand_checks"],"permission_checks":permission_checks},indent=2))
