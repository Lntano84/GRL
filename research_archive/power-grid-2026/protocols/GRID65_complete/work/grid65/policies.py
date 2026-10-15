"""Frozen FRAME/global rules and a separately trained ICAPS2022 stopping-only baseline."""
import hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('grid65_old_rules',ROOT/'work/grid63/policies.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sys.path.insert(0,str(ROOT/'work/grid64'))
import metacontrol as M
import torch
key=old.key
POLICIES=('FULL','FRAME','GLOBAL_MATCH','META_DQN0','META_DQN1')
_models={}
def get_model(policy):
    if policy not in _models:
        seed=int(policy[-1]);d=json.loads((ROOT/'outputs/grid65/design.json').read_text(encoding='utf-8'))
        item=next(r for r in d['methods'] if r['policy']==policy);path=ROOT/item['meta_model']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['meta_model_sha256']
        with torch.random.fork_rng(devices=[]):model=M.QModel()
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval()
        _models[policy]=(model,item['meta_model_sha256'])
    return _models[policy]
def run(order,observe,policy,subids,aliases,safe=.9,limit=None):
    if policy in old.POLICIES:return old.run(order,observe,policy,subids,aliases,safe)
    assert policy in POLICIES and limit is not None
    model,h=get_model(policy);result=M.run(order,observe,model,limit)
    result.update(meta_model_sha256=h,completion_budget=0,global_extra_queries=0,
                  pre_extra_choice=result['chosen_pool_id'],completion_candidate_ids=[],extra_selected_ids=[])
    return result
