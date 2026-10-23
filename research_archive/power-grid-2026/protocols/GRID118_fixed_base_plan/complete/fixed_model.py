"""Fixed-checkpoint correction proposal. No optimizer or training entry point."""
from pathlib import Path
import io
import sys
import zipfile
import torch
from torch import nn

ROOT = Path('D:/Codex/GRL研究')
sys.path.insert(0, str(ROOT / 'work/grid115_matched_module_plan'))
from matched_model import OriginalBackbone, RelationalValue, BRANCH_SEED_OFFSET


def load_backbone(seed):
    """Read online Q tensors only; never construct SB3 or load its optimizer."""
    path = ROOT / f'outputs/grid107_direct_value/models/direct_seed{seed}_final.zip'
    with zipfile.ZipFile(path) as z:
        policy = torch.load(io.BytesIO(z.read('policy.pth')), map_location='cpu', weights_only=True)
    state = {k[len('q_net.'):]: v for k, v in policy.items() if k.startswith('q_net.')}
    net = OriginalBackbone()
    net.load_state_dict(state, strict=True)
    return net


class FixedBaseCorrection(nn.Module):
    def __init__(self, seed, arm):
        super().__init__()
        assert arm in ('JOINT', 'FACTOR')
        self.base = load_backbone(seed)
        self.base.requires_grad_(False)
        self.base.eval()
        torch.manual_seed(BRANCH_SEED_OFFSET + seed)
        self.delta = RelationalValue(arm)
        nn.init.zeros_(self.delta.head[-1].weight)
        nn.init.zeros_(self.delta.head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, **public_inputs):
        with torch.no_grad():
            base_q = self.base(public_inputs['global41'])
        return base_q + self.delta(**public_inputs)

    def correction_targets(self, global41, original_two_action_target):
        with torch.no_grad():
            return original_two_action_target - self.base(global41)

    def trainable_parameters(self):
        return tuple(p for p in self.parameters() if p.requires_grad)
