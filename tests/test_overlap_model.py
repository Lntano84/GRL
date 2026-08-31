import torch

from grl.models import MarginalGainPredictor, OverlapMarginalGainPredictor


def test_overlap_model_forward_shape_and_baseline_is_separate():
    embeddings = torch.randn(2, 4)
    degrees = torch.rand(2, 1)
    mask = torch.zeros(2, 1)
    overlap = torch.rand(9)
    baseline = MarginalGainPredictor(4, hidden_dim=8)
    model = OverlapMarginalGainPredictor(4, hidden_dim=8)

    assert baseline(embeddings, degrees, mask, 1).shape == (1, 1)
    assert model(embeddings, degrees, mask, 1, overlap).shape == (1, 1)
