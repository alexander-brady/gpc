import pytest
import torch

from gpc.trajectory import fit_horizon


def _history(*steps: float) -> torch.Tensor:
    return torch.tensor(steps).view(1, -1, 1)


@pytest.mark.parametrize(
    ("steps", "horizon", "expected"),
    [
        ((1, 2, 3, 4), 2, [3, 4]),  # keeps the latest steps
        ((1, 2, 3), 3, [1, 2, 3]),  # unchanged when it already fits
        ((5, 6), 4, [5, 5, 5, 6]),  # pads with the earliest step
        ((7,), 3, [7, 7, 7]),
    ],
)
def test_fit_horizon(steps, horizon, expected):
    assert fit_horizon(_history(*steps), horizon).flatten().tolist() == expected


def test_fit_horizon_preserves_batch_and_trailing_dims():
    history = torch.randn(2, 1, 3, 4)
    fitted = fit_horizon(history, 3)
    assert fitted.shape == (2, 3, 3, 4)
    assert (fitted == history).all()  # every padded step equals the only step


@pytest.mark.parametrize("horizon", [0, -1])
def test_fit_horizon_rejects_non_positive_horizon(horizon):
    with pytest.raises(ValueError, match="horizon"):
        fit_horizon(_history(1), horizon)


def test_fit_horizon_rejects_empty_history():
    with pytest.raises(ValueError, match="at least one step"):
        fit_horizon(torch.zeros(1, 0, 1), 2)
