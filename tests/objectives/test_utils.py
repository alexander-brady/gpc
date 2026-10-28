import pytest
from torch import Tensor, nn

from gpc.objectives import accepts_plans, frame_objective, is_frame_objective, latent_objective


def no_plans(futures: Tensor, goal: Tensor | None) -> Tensor:
    return futures


def keyword_plans(futures: Tensor, goal: Tensor | None, *, plans: Tensor) -> Tensor:
    return futures


def positional_plans(futures: Tensor, goal: Tensor | None, plans: Tensor | None = None) -> Tensor:
    return futures


def var_keyword(futures: Tensor, goal: Tensor | None, **kwargs) -> Tensor:
    return futures


def other_keyword(futures: Tensor, goal: Tensor | None, *, weights: Tensor | None = None) -> Tensor:
    return futures


class Forward(nn.Module):
    def forward(self, futures: Tensor, goal: Tensor | None) -> Tensor:
        return futures


class ForwardWithPlans(nn.Module):
    def forward(self, futures: Tensor, goal: Tensor | None, *, plans: Tensor) -> Tensor:
        return futures


class WithPlans:
    def __call__(self, futures: Tensor, goal: Tensor | None, *, plans: Tensor) -> Tensor:
        return futures


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        (no_plans, False),
        (keyword_plans, True),
        (positional_plans, True),
        (var_keyword, True),
        (other_keyword, False),
        (WithPlans(), True),
        (Forward(), False),
        (ForwardWithPlans(), True),
    ],
)
def test_accepts_plans(objective, expected):
    assert accepts_plans(objective) is expected


def test_frame_objective_marks_functions_and_classes():
    @frame_objective
    def fn(futures: Tensor, goal: Tensor | None) -> Tensor:
        return futures

    @frame_objective
    class Cls:
        def __call__(self, futures: Tensor, goal: Tensor | None) -> Tensor:
            return futures

    assert is_frame_objective(fn) and is_frame_objective(Cls())
    assert not is_frame_objective(no_plans)


def test_latent_objective_is_the_default():
    @latent_objective
    def fn(futures: Tensor, goal: Tensor | None) -> Tensor:
        return futures

    assert not is_frame_objective(fn)
    assert not is_frame_objective(no_plans)


def test_latent_objective_overrides_frame_objective():
    @latent_objective
    @frame_objective
    def fn(futures: Tensor, goal: Tensor | None) -> Tensor:
        return futures

    assert not is_frame_objective(fn)
