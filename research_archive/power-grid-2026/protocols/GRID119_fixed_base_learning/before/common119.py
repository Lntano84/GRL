"""GRID118 authorized execution. Saved public inputs only; no environment."""
from pathlib import Path
import sys
R=Path('D:/Codex/GRL研究')
sys.path.insert(0,str(R/'work/grid113_relational_learning'))
from common import *
from common import verify as verify113
sys.path.insert(0,str(R/'work/grid118_fixed_base_plan'))
from fixed_model import load_backbone,FixedBaseCorrection
import torch
from torch import nn
W=Path(__file__).parent;O=R/'outputs/grid119_fixed_base_learning';S=R/'outputs/grid113_relational_learning';PLAN=R/'outputs/grid118_fixed_base_plan'

class BaseExtra(nn.Module):
    def __init__(self,seed):
        super().__init__();self.base=load_backbone(seed)
    def forward(self,**public_inputs):return self.base(public_inputs['global41'])

def build(seed,arm):
    return BaseExtra(seed) if arm=='BASE_EXTRA' else FixedBaseCorrection(seed,arm.removeprefix('FIXED_'))

def trainable(net):return [p for p in net.parameters() if p.requires_grad]

def verify():
    for name in ['code_freeze.json','data_freeze.json']:
        for n,h in read(O/name).items():assert sha(R/n)==h,n
    for n,h in read(PLAN/'input_manifest.json').items():assert sha(R/n)==h,n
    verify113();sealed()

def gradient_flow(net):
    ans={}
    for name,p in net.named_parameters():
        if not p.requires_grad:
            assert p.grad is None,name
            continue
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        key='base' if name.startswith('base.') else 'delta.'+name.split('.')[1]
        ans[key]=ans.get(key,0.)+float(p.grad.detach().double().square().sum())
    return {k:v**.5 for k,v in ans.items()}
