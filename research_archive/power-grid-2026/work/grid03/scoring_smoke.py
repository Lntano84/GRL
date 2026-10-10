"""Official installed score helper, bounded to 3 x 10 steps on an isolated bundled copy."""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import grid2op
import numpy as np
from grid2op.Agent import DoNothingAgent
from grid2op.utils import ScoreL2RPN2023
from lightsim2grid import LightSimBackend

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/grid03/scoring_smoke"
TARGET=ROOT/"work/grid03/scoring_smoke_env"
SOURCE=Path(grid2op.__file__).parent/"data/l2rpn_idf_2023"

if OUT.exists() or TARGET.exists():
    raise RuntimeError("Smoke artifacts exist; do not overwrite or silently recompute")
OUT.mkdir(parents=True)
TARGET.mkdir(parents=True)
source_scenarios=sorted((SOURCE/"chronics").iterdir())
selected=source_scenarios[0]
manifest=[]
for src in SOURCE.iterdir():
    if src.is_file():
        dst=TARGET/src.name
        shutil.copyfile(src,dst)
        manifest.append({"path":src.name,"sha256":hashlib.sha256(dst.read_bytes()).hexdigest()})
shutil.copytree(selected,TARGET/"chronics"/selected.name)
(OUT/"copy_manifest.json").write_text(json.dumps({"scenario":selected.name,"static":manifest},indent=2),encoding="utf-8")
start=time.monotonic()
env=grid2op.make(str(TARGET),backend=LightSimBackend(),test=True)
score=ScoreL2RPN2023(env,env_seeds=[0],agent_seeds=[0],nb_scenario=1,max_step=10,nb_process_stats=1,add_nb_highres_sim=True)
result=score.get(DoNothingAgent(env.action_space),path_save=str(OUT/"agent_episode"))
assert len(result[0])==1
total,op,nres,assistant=map(float,result[0][0])
assert result[1]==[10] and result[2]==[10]
assert abs(op)<1e-7 and abs(nres-100)<1e-7 and abs(assistant-100)<1e-7 and abs(total-40)<1e-7
stats=[]
for stat in [score.stat_dn,score.stat_no_overflow_rp]:
    metadata=stat.get_metadata()
    assert metadata["max_step"]==10 and metadata["0"]["nb_step"]==11
    stats.append({"path":str(stat.path_save_stats),"metadata":metadata})
summary={"passed":True,"score_helper_result":result,"physical_steps":30,"wall_s":time.monotonic()-start,
         "scenario_source":"Bundled 575-step scenario; isolated copy; 10-step artificial horizon for API qualification only.",
         "not_research_result":True,"not_competition_exact_certification":True,"reference_statistics":stats,
         "score_search_separation":"This smoke uses DoNothing, not LJN search; later searches retain MaxRhoReward."}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=lambda x:x.item() if hasattr(x,"item") else str(x)),encoding="utf-8")
env.close()
print(json.dumps({"passed":True,"physical_steps":30,"score":result,"wall_s":summary["wall_s"]},default=str),flush=True)
