"""No environment steps. Verify score transformations on hand-calculated cases."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import grid2op
from grid2op.Reward import L2RPNSandBoxScore, _AlertTrustScore, _NewRenewableSourcesUsageScore
from grid2op.utils import ScoreL2RPN2023

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid03"
checks = []
for usage, expected in [(50,-1), (80,0), (100,1)]:
    actual = float(_NewRenewableSourcesUsageScore._surlinear_func_curtailment(usage))
    assert abs(actual-expected) < 1e-7
    checks.append({"kind":"renewable_transform", "usage_percent":usage, "expected":expected, "actual":actual})
for blackout, expected in [(False,1), (True,0)]:
    actual = _AlertTrustScore._normalisation_fun(0,0,0,-3,1,blackout)
    assert actual == expected
    checks.append({"kind":"assistant_no_attack", "blackout":blackout, "expected":expected, "actual":actual})
fake_env=SimpleNamespace(gen_cost_per_MW=np.array([20,10],dtype=np.float32),
    _gen_activeprod_t=np.array([100,50],dtype=np.float32),_actual_dispatch=np.array([2,-2],dtype=np.float32),
    _sum_curtailment_mw=np.float32(-5),_storage_power=np.array([3],dtype=np.float32),
    backend=SimpleNamespace(generators_info=lambda:(np.array([100,50],dtype=np.float32),None,None),
                            loads_info=lambda:(np.array([140],dtype=np.float32),None,None)))
raw=L2RPNSandBoxScore()
raw.env_dt_over_3600=np.float32(1/12)
actual=float(raw(None,fake_env,False,False,False,False))
expected=(10+4+5+3)*20/12
assert abs(actual-expected)<1e-4
checks.append({"kind":"raw_cost_hand_fixture","actual":actual,"expected":expected})
from score_adapter import public_cost_ledger
ledger_env=SimpleNamespace(gen_cost_per_MW=np.array([20,10]),delta_time_seconds=300)
ledger_obs=SimpleNamespace(gen_p=np.array([100,50]),load_p=np.array([140]),actual_dispatch=np.array([2,-2]),
                           curtailment_mw=np.array([5]),storage_power=np.array([3]))
ledger=public_cost_ledger(ledger_obs,ledger_env,0)
assert abs(ledger["recomputed_raw_cost"]-expected)<1e-9
checks.append({"kind":"independent_ledger_hand_fixture","actual":ledger,"expected":expected})
for current,previous in [(5,5),(0,5),(6,5)]:
    ledger_obs.curtailment_mw=np.array([current])
    fake_env._sum_curtailment_mw=np.float32(previous-current)
    official=float(raw(None,fake_env,False,False,False,False))
    hand=(10+4+(current-previous)+3)*20/12
    independent=public_cost_ledger(ledger_obs,ledger_env,previous)
    assert abs(official-hand)<1e-4 and abs(independent["recomputed_raw_cost"]-hand)<1e-9
    checks.append({"kind":"curtailment_level_vs_change_hand_fixture","current_mw":current,
        "previous_mw":previous,"expected":hand,"official":official,"independent":independent})

class Stats:
    def __init__(self, scores):
        self.scores=np.array(scores,dtype=np.float32)
    def get(self,name):
        if name=="load_p":
            return np.array([[10],[10],[10],[10]],dtype=np.float32),np.zeros(4,dtype=int)
        return self.scores,np.zeros(4,dtype=int)

# Exercise the actual installed normalization and weighted-composition method with hand fixtures.
score=ScoreL2RPN2023.__new__(ScoreL2RPN2023)
score.env=SimpleNamespace(gen_cost_per_MW=np.array([10],dtype=np.float32))
score.stat_dn=Stats([20,20,20,1])
score.stat_no_overflow_rp=Stats([10,10,10,1])
score.min_losses_ratio=0.8
score.max_step=3
score.scale_assistant_score=score.scale_nres_score=100
score.min_nres_score=-100
score.min_assistant_score=-300
score.weight_op_score=0.6
score.weight_assistant_score=0.25
score.weight_nres_score=0.15
meta={"nb_timestep_played":3}
refs={"0":{"nb_step":4},"max_step":3}
for costs,expected_op,expected_total in [([20,20,20,1],0,40),([10,10,10,1],80,88)]:
    other=[{"_scores_grid_operational_cost":x,"_scores_assistant_confidence":1,"_scores_new_renewable_sources_usage":1} for x in costs]
    # Derive the official key prefix, rather than assuming it in the fixture.
    from grid2op.utils.underlying_statistics import EpisodeStatistics
    other=[{k.replace("_scores",EpisodeStatistics.KEY_SCORE):v for k,v in row.items()} for row in other]
    result,n_played,total_ts=score._compute_episode_score(0,meta,other,refs,refs)
    assert abs(result[0]-expected_total)<1e-7 and abs(result[1]-expected_op)<1e-7
    assert (n_played,total_ts)==(3,3)
    checks.append({"kind":"installed_helper_hand_fixture","actual":list(result),"expected_total":expected_total,"expected_operational":expected_op})
paths = ["utils/l2rpn_idf_2023_scores.py", "utils/l2rpn_2020_scores.py", "Reward/l2RPNSandBoxScore.py", "Reward/_alertTrustScore.py", "Reward/_newRenewableSourcesUsageScore.py", "Environment/baseEnv.py"]
base = Path(grid2op.__file__).parent
hashes = {p:hashlib.sha256((base/p).read_bytes()).hexdigest() for p in paths}
source = (base/paths[1]).read_text(encoding="utf-8")
load_twice = 'prod_p_rp, _ = self.stat_no_overflow_rp.get("load_p")' in source
assert load_twice
result={"passed":True,"physical_steps":0,"checks":checks,"source_sha256":hashes,
        "normalizer_load_read_twice":load_twice,
        "status":"Raw reward and arithmetic qualification only; competition-exact normalization not certified.",
        "unmodified_installed_grid2op":grid2op.__version__}
(OUT/"metric_unit_checks.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
