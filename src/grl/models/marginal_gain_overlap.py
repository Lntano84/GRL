from __future__ import annotations

import torch
import torch.nn as nn

from .marginal_gain import MarginalGainPredictor, SetEncoder


class OverlapMarginalGainPredictor(MarginalGainPredictor):
    """Direct MLP reranker augmented with explicit topology overlap features."""

    model_name = "Direct-MLP+Overlap"
    model_version = "1"

    def __init__(self, embedding_dim: int, hidden_dim: int = 64, structural_dim: int = 1, overlap_dim: int = 9):
        nn.Module.__init__(self)
        self.overlap_dim = int(overlap_dim)
        input_dim = embedding_dim + structural_dim
        self.seed_encoder = SetEncoder(input_dim, hidden_dim)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(input_dim + self.overlap_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings, norm_degrees, seed_mask, candidate_indices, overlap_features):
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        if norm_degrees.dim() == 2:
            norm_degrees = norm_degrees.unsqueeze(0)
        if seed_mask.dim() == 2:
            seed_mask = seed_mask.unsqueeze(0)
        batch_size = embeddings.shape[0]
        candidates = torch.as_tensor(candidate_indices, device=embeddings.device, dtype=torch.long)
        if candidates.dim() == 0:
            candidates = candidates.expand(batch_size)
        overlap = torch.as_tensor(overlap_features, device=embeddings.device, dtype=embeddings.dtype)
        if overlap.dim() == 1:
            overlap = overlap.unsqueeze(0)
        if overlap.shape != (batch_size, self.overlap_dim):
            raise ValueError(f"overlap_features shape={tuple(overlap.shape)}, expected ({batch_size}, {self.overlap_dim})")
        node_features = torch.cat([embeddings, norm_degrees], dim=-1)
        seed_repr = self.seed_encoder(node_features, seed_mask)
        row = torch.arange(batch_size, device=embeddings.device)
        candidate_repr = self.candidate_encoder(torch.cat([node_features[row, candidates], overlap], dim=-1))
        interaction = seed_repr * candidate_repr
        difference = (seed_repr - candidate_repr).abs()
        return self.head(torch.cat([seed_repr, candidate_repr, interaction, difference], dim=-1))
