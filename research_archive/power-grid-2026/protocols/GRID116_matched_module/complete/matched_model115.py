"""Engineering implementation of the proposed matched architecture; no trainer."""
from pathlib import Path
import sys,torch
from torch import nn
R=Path('D:/Codex/GRL研究');sys.path.insert(0,str(R/'work/grid113_relational_learning'))
from model import RelationalValue
BRANCH_SEED_OFFSET=20261023
class OriginalBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Names and constructor order match the original SB3 QNetwork's trainable state.
        self.q_net=nn.Sequential(nn.Linear(41,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2))
    def forward(self,global41):return self.q_net(global41)
class MatchedModule(nn.Module):
    def __init__(self,seed,arm):
        super().__init__();assert arm in ['JOINT','FACTOR'];self.arm=arm
        torch.manual_seed(seed);self.base=OriginalBackbone()
        torch.manual_seed(BRANCH_SEED_OFFSET+seed);self.delta=RelationalValue(arm)
        # Only the final correction layer is zero. No fitted GRID107/113 weights are loaded.
        nn.init.zeros_(self.delta.head[-1].weight);nn.init.zeros_(self.delta.head[-1].bias)
    def forward(self,**public_inputs):
        return self.base(public_inputs['global41'])+self.delta(**public_inputs)
