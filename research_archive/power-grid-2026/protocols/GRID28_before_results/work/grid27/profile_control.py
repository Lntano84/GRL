"""One exact protected baseline with hierarchical control/optimizer profiling.

Diagnosis only: wrappers add instrumentation, so this is not an acceleration
comparison. Original source/checkpoints and every delivered action stay fixed.
"""
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid27_profile"
OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT / "profile.json").exists()
sys.path[:0] = [str(ROOT / "work/grid26"), str(ROOT / "work/grid08")]
from common import ResidualControl, OptimModule, write_json
from engine import Search
from ljn_nn.LJNAgent import LJNAgentTopoNN

stack = []; totals = defaultdict(lambda: {"calls":0, "inclusive_s":0., "self_s":0.})
solver = []; saved = []


def install(cls, name, label, root=False):
    owned = name in cls.__dict__
    original = getattr(cls, name)
    def wrapped(self, *args, **kwargs):
        if not root and not stack:
            return original(self, *args, **kwargs)
        task = {"started":time.perf_counter(),"child_s":0.}
        stack.append(task)
        owner = getattr(self, "_profile_owner", "")
        tag = (owner+"." if owner else "")+label
        try:
            return original(self, *args, **kwargs)
        finally:
            elapsed = time.perf_counter()-task["started"]
            assert stack.pop() is task
            item = totals[tag]; item["calls"] += 1
            item["inclusive_s"] += elapsed; item["self_s"] += elapsed-task["child_s"]
            if stack: stack[-1]["child_s"] += elapsed
            if label == "solve_problem" and kwargs.get("solver_type") is not None:
                p = args[0]; stats = p.solver_stats
                solver.append({"owner":owner, "solver":str(kwargs["solver_type"]),
                    "status":str(p.status), "compilation_s":p.compilation_time,
                    "solver_time_s":None if stats is None else stats.solve_time,
                    "num_iters":None if stats is None else stats.num_iters})
    setattr(cls, name, wrapped)
    saved.append((cls,name,original,owned))


original_init = ResidualControl.__init__
def tagged_init(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    self.base.optim._profile_owner="base_optimizer"
    self.safe._profile_owner="restoration_optimizer"
ResidualControl.__init__ = tagged_init
install(ResidualControl,"propose","controller",True)
install(LJNAgentTopoNN,"act","base_agent")
install(Search,"choose","n1_search")
for method in ["update_parameters","run_dc","compute_optimum_unsafe","compute_optimum_safe",
               "_solve_problem","_aux_compute_kcl","_update_topo_param","_update_th_lim_param","_update_inj_param"]:
    install(OptimModule,method,method.lstrip("_"))
design = json.loads((ROOT / "outputs/grid26/design.json").read_text())
design["caps"]["episode_s"]=1200
item={"repeat":0,"week":design["weeks"][0],"method":"FULL_PROFILE","rule":"FULL","budget":None,"checkpoint":None}
write_json(OUT / "design.json",{"item":item,"claim":"One profiled exact FULL baseline; disjoint self time, hierarchical inclusive time; no speed claim.",
    "new_fits":0,"reserved_evaluations":0})
(OUT / "RUN_STATE.md").write_text("RUNNING: one serial protected March baseline, hierarchical profiling; exact historical path enforced.\n")
try:
    spec = importlib.util.spec_from_file_location("grid26_profile_worker", ROOT / "work/grid26/run.py")
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    worker.OUT=OUT
    summary=worker.worker(item,design)
    assert summary["exact_full_trajectory"]
    assert not stack
    root_s=totals["controller"]["inclusive_s"]
    total_self=sum(r["self_s"] for r in totals.values())
    assert abs(root_s-total_self)<1e-7
    records=[{"component":k,**v,"share_of_controller":v["self_s"]/root_s} for k,v in sorted(totals.items(),key=lambda kv:-kv[1]["self_s"])]
    result={"passed":True,"summary":summary,"records":records,"solver_records":solver,
            "self_time_sum_s":total_self,"root_time_s":root_s,
            "instrumentation":"Class wrappers preserve original calls and parent/child timing. Instrumented single development run, not speed/causal evidence.",
            "new_fits":0,"reserved_evaluations":0}
    write_json(OUT / "profile.json",result)
    (OUT / "RUN_STATE.md").write_text("COMPLETED: exact protected baseline; hierarchical profiling accounted; no pending worker.\n")
    print(json.dumps({"passed":True,"summary":summary,"records":records},indent=2),flush=True)
finally:
    ResidualControl.__init__=original_init
    for cls,name,original,owned in reversed(saved):
        if owned:setattr(cls,name,original)
        else:delattr(cls,name)
