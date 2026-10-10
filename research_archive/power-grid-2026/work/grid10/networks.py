"""Same actor/critic architecture, explicit fixed native-value units.

This is static normalization from archived training returns, not adaptive PopArt.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grid08'))
from common import torch
from networks_v1 import ActorCritic as BaseActorCritic


class ActorCritic(BaseActorCritic):
    def __init__(self, kind, n_nodes, hidden=64, max_edges=372):
        super().__init__(kind, n_nodes, hidden, max_edges)
        self.register_buffer('value_mu', torch.tensor(0.))
        self.register_buffer('value_sigma', torch.tensor(1.))

    def forward(self, x, edges, glob):
        logits, value = super().forward(x, edges, glob)
        return logits, value * self.value_sigma + self.value_mu


def value_loss(prediction, targets, mu, sigma, version):
    if version == 'raw':
        return torch.nn.functional.smooth_l1_loss(prediction, targets)
    assert version == 'normalized' and sigma > 0
    return torch.nn.functional.mse_loss((prediction - mu) / sigma, (targets - mu) / sigma)
