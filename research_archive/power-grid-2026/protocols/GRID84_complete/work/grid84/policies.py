"""Same meta-control implementation and fixed-guard action, five seeds."""
from pathlib import Path
import functools,importlib.util
import torch
ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
    s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=module('grid84_fixed','work/grid76/policies.py')
M=module('grid84_meta','work/grid77/metacontrol.py')
POLICIES=('FRAME_GUARD',)+tuple(f'META_CONTEXT{i}' for i in range(5))
@functools.lru_cache(maxsize=5)
def model(seed):
    if seed<2:return M.load(seed)
    result=M.QModel();result.load_state_dict(torch.load(ROOT/f'outputs/grid84/models/cost2_seed{seed}/final.pt',map_location='cpu',weights_only=True));result.eval();return result
def run(order,observe,policy,subids,aliases,safe=.9,limit=None,context=None):
    assert policy in POLICIES and safe==.9 and limit is not None and context is not None
    if policy=='FRAME_GUARD':return C.run(order,observe,policy,subids,aliases,safe,limit,context=context)
    seed=int(policy[-1]);return M.run(order,observe,seed,subids,aliases,limit,context,model=model(seed))
