import pytest

from grl.training.comparison import grouped_metrics
from grl.training.marginal_dataset import MarginalGainSample


def _sample(context_id: str, candidate: int, gain: float) -> MarginalGainSample:
    return MarginalGainSample(
        context_id=context_id,
        seed_set=[],
        candidate=candidate,
        seed_set_size=0,
        base_spread=0.0,
        extended_spread=gain,
        marginal_gain=gain,
        label_std=0.0,
    )


def test_grouped_metrics_rank_only_within_context():
    samples = [
        _sample("a", 0, 1.0),
        _sample("a", 1, 2.0),
        _sample("b", 2, 100.0),
        _sample("b", 3, 200.0),
    ]
    predictions = [1.0, 2.0, -2.0, -1.0]

    metrics = grouped_metrics(samples, predictions)

    assert metrics["contexts"] == 2
    assert metrics["samples"] == 4
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["kendall"] == pytest.approx(1.0)
    assert metrics["top_1_recall"] == 1.0
    assert metrics["top_5_recall"] == 1.0
    assert metrics["top_10_recall"] == 1.0
