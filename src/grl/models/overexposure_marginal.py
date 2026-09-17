"""State-conditioned marginal-gain predictor for the overexposure model.

Relationship to :class:`grl.models.MarginalGainPredictor`
---------------------------------------------------------
The existing predictor is conditioned on the seed set through a permutation-invariant encoder of
the seed mask.  That is genuine state conditioning, and it is the right control here, but it only
tells the model *who* is seeded.  Under the threshold-window process the quantity that actually
drives activation is the accumulated exposure ``delta``, which the seed mask does not contain:
two different seed sets can produce very different exposure fields, and the same seed set produces
different exposure under different threshold draws.

This module therefore adds an optional exposure channel:

    ``exposure_features`` -- a per-node vector (typically the realised ``delta``, possibly plus
    the node's own window), concatenated onto the candidate features before encoding.

``use_exposure=False`` reproduces the seed-mask-only predictor exactly, so the two share one
implementation and the ablation is a single flag rather than two divergent code paths.  The
existing class is left untouched so the independent-cascade experiments keep working.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .marginal_gain import MarginalGainPredictor, SetEncoder


class StateConditionedMarginalPredictor(nn.Module):
    """Predict ``Delta(v | S)`` from the seed set and, optionally, the realised exposure field.

    Parameters
    ----------
    embedding_dim
        Dimension of the node embeddings.
    structural_dim
        Dimension of the per-node structural feature vector (normally 1, the normalised degree).
    exposure_dim
        Dimension of the per-node exposure feature vector.  ``0`` disables the channel.
    use_seed_mask
        Whether to feed the seed mask through the set encoder.  Turning this off while
        ``exposure_dim > 0`` gives a purely exposure-driven model, which is a useful ablation:
        it isolates how much of the signal is in the exposure field itself.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 64,
        structural_dim: int = 1,
        exposure_dim: int = 0,
        use_seed_mask: bool = True,
    ) -> None:
        super().__init__()
        if exposure_dim < 0:
            raise ValueError(f"exposure_dim must be >= 0, got {exposure_dim}")
        if not use_seed_mask and exposure_dim == 0:
            raise ValueError(
                "at least one of use_seed_mask / exposure_dim must be active, "
                "otherwise the model cannot see the state at all"
            )
        self.embedding_dim = embedding_dim
        self.structural_dim = structural_dim
        self.exposure_dim = exposure_dim
        self.use_seed_mask = use_seed_mask

        base_dim = embedding_dim + structural_dim
        node_dim = base_dim + exposure_dim

        hidden = hidden_dim
        self.seed_encoder = SetEncoder(node_dim, hidden) if use_seed_mask else None
        self.candidate_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Inputs to the head: seed representation (or zeros), candidate representation, their
        # product and their absolute difference.  With no set encoder the seed slot is a learned
        # bias so the head shape is unchanged across the ablation.
        self.no_seed_bias = nn.Parameter(torch.zeros(hidden)) if not use_seed_mask else None
        self.head = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _node_features(
        self,
        embeddings: torch.Tensor,
        norm_degrees: torch.Tensor,
        exposure_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        if norm_degrees.dim() == 2:
            norm_degrees = norm_degrees.unsqueeze(0)
        parts = [embeddings, norm_degrees]
        if self.exposure_dim > 0:
            if exposure_features is None:
                raise ValueError(
                    f"exposure_dim={self.exposure_dim} but exposure_features is None"
                )
            if exposure_features.dim() == 2:
                exposure_features = exposure_features.unsqueeze(0)
            if exposure_features.shape[-1] != self.exposure_dim:
                raise ValueError(
                    f"exposure_features last dim {exposure_features.shape[-1]} "
                    f"!= exposure_dim {self.exposure_dim}"
                )
            parts.append(exposure_features.to(embeddings.dtype))
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        embeddings: torch.Tensor,
        norm_degrees: torch.Tensor,
        seed_mask: torch.Tensor,
        candidate_indices: int | torch.Tensor,
        exposure_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score one candidate per batch row.

        ``candidate_indices`` is a 1-D tensor of length ``batch_size`` (or a scalar broadcast over
        the batch), matching :class:`grl.models.MarginalGainPredictor`.  To score several
        candidates per seed set, repeat the seed row and flatten -- see
        :func:`score_candidates` for the batched helper.
        """
        if seed_mask.dim() == 2:
            seed_mask = seed_mask.unsqueeze(0)
        node_features = self._node_features(embeddings, norm_degrees, exposure_features)
        batch_size = node_features.shape[0]

        candidates = torch.as_tensor(
            candidate_indices, device=node_features.device, dtype=torch.long
        )
        if candidates.dim() == 0:
            candidates = candidates.expand(batch_size)
        if candidates.dim() != 1 or candidates.shape[0] != batch_size:
            raise ValueError(
                f"candidate_indices must have one entry per batch row "
                f"({batch_size}), got shape {tuple(candidates.shape)}; "
                f"use score_candidates() to evaluate many candidates per row"
            )

        if self.seed_encoder is not None:
            seed_repr = self.seed_encoder(node_features, seed_mask.to(node_features.dtype))
        else:
            seed_repr = self.no_seed_bias.unsqueeze(0).expand(batch_size, -1)

        row = torch.arange(batch_size, device=node_features.device)
        candidate_repr = self.candidate_encoder(node_features[row, candidates])
        interaction = seed_repr * candidate_repr
        difference = (seed_repr - candidate_repr).abs()
        return self.head(torch.cat([seed_repr, candidate_repr, interaction, difference], dim=-1))

    def score_candidates(
        self,
        embeddings: torch.Tensor,
        norm_degrees: torch.Tensor,
        seed_masks: torch.Tensor,
        candidates: torch.Tensor,
        exposure_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score ``candidates[i, j]`` against seed state ``i``; returns ``(rows, cols)``.

        ``seed_masks`` is ``(rows, n, 1)``, ``candidates`` is ``(rows, cols)``.  The implementation
        flattens the candidate axis and repeats each seed row, which is what the one-candidate-per-
        row ``forward`` contract requires.
        """
        if seed_masks.dim() == 2:
            seed_masks = seed_masks.unsqueeze(0)
        rows, cols = candidates.shape
        flat_candidates = candidates.reshape(-1)
        repeated_masks = seed_masks.repeat_interleave(cols, dim=0)
        repeated_emb = embeddings.expand(rows, -1, -1).repeat_interleave(cols, dim=0)
        repeated_nd = norm_degrees.expand(rows, -1, -1).repeat_interleave(cols, dim=0)
        repeated_exp = None
        if exposure_features is not None:
            if exposure_features.dim() == 2:
                exposure_features = exposure_features.unsqueeze(0)
            repeated_exp = exposure_features.expand(rows, -1, -1).repeat_interleave(cols, dim=0)
        out = self.forward(repeated_emb, repeated_nd, repeated_masks, flat_candidates,
                           repeated_exp)
        return out.reshape(rows, cols)
