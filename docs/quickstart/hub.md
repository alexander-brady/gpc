# 5. Sharing on the Hub

Policies, world models and whole controllers can be saved locally with `save_pretrained`,
pushed to the [Hugging Face Hub](https://huggingface.co/docs/hub) with `push_to_hub`, and
loaded back with `from_pretrained`. Log in once with `hf auth login` before pushing.

## Models

Each component is saved to its own subfolder of the repo, so a policy and a world model can
be pushed to the same repo without overwriting each other:

```python
policy.push_to_hub("me/push-t")  # -> policy/
world_model.push_to_hub("me/push-t")  # -> world_model/
```

`from_pretrained` downloads only that component's subfolder and rebuilds the subclass that
saved it, so you don't need to know which policy class a repo holds:

```python
from gpc.models import Policy, WorldModel

policy = Policy.from_pretrained("me/push-t")  # a MyPolicy, from the policy step
world_model = WorldModel.from_pretrained("me/push-t", map_location="cuda")
```

To keep several policies in one repo, pass `subfolder=` to both calls. Keyword arguments
that are config fields, such as `prediction_horizon=8`, change the config before the model
is built.

!!! note "Components are built from their config"

    A component is rebuilt as `cls(config)`, so its constructor must take only its config.
    A subclass with its own config dataclass sets `config_class` to it:

    ```python
    @dataclass(kw_only=True)
    class MyPolicyConfig(PolicyConfig):
        hidden_size: int = 256


    class MyPolicy(Policy):
        config_class = MyPolicyConfig

        def __init__(self, config: MyPolicyConfig):
            super().__init__(config)
            self.net = nn.Linear(config.hidden_size, config.hidden_size)
    ```

    Config fields must be JSON values; override `Config.to_dict` and `Config.from_dict`
    otherwise. The class must be importable from the same module path when loading.

## Controllers

Pushing a controller saves its components and hyperparameters to one repo:

```python
controller.push_to_hub("me/push-t")
```

```
config.json         # controller hyperparameters
README.md
policy/             # loadable on its own with Policy.from_pretrained
world_model/        # loadable on its own with WorldModel.from_pretrained
objective/          # only for an ObjectiveModule
```

`from_pretrained` builds the class it is called on, so any controller can load a repo saved
by another one:

```python
from gpc import GPCOpt, GPCRank

rank = GPCRank.from_pretrained("me/push-t")  # even if a GPCOpt was pushed
opt = GPCOpt.from_pretrained("me/push-t", num_steps=50)  # override a hyperparameter
```

Hyperparameters shared by every controller, such as `action_horizon`, are always read.
Each controller class keeps its own in a separate section of `config.json`, and a
controller reads the sections of all the classes it inherits from, base classes first. So
`GPCRankOpt` picks up the `num_samples` saved by a `GPCRank` and the `num_steps` saved by a
`GPCOpt`, and anything never saved takes its default. Saving a loaded controller writes the
other controllers' sections back, so repeated pushes accumulate them.

Pass `policy=`, `world_model=` or `objective=` to use your own component instead of the
saved one. Calling `GPC.from_pretrained` builds whichever controller saved the repo.

!!! note "Custom controllers"

    A controller subclass must store each constructor argument as an attribute of the same
    name (e.g. `self.temperature = temperature`) so it can be saved.

## Objectives

How the objective is saved depends on what it is:

| Objective | Saved as |
| --------- | -------- |
| [`ObjectiveModule`][gpc.objectives.ObjectiveModule] subclass | Config and weights in `objective/`, like a model. |
| Module-level function, or a `functools.partial` of one with JSON keyword arguments | A reference such as `my_pkg.objectives:goal_distance`, imported again when loading. |
| Anything else (a lambda, a local function, an instance of a plain class) | Not saved, with a warning. Pass `objective=` to `from_pretrained`. |

A learned reward model, for example, is an `ObjectiveModule`:

```python
from gpc.objectives import ObjectiveModule


class RewardModel(ObjectiveModule):
    def __init__(self, config=None):
        super().__init__(config)
        self.head = nn.Linear(16, 1)

    def forward(self, futures, goal):
        return self.head(futures[:, :, -1]).squeeze(-1)  # [B, N]
```

!!! warning "Only load repos you trust"

    Loading imports the classes and functions a repo names. They must already be installed,
    so no code is downloaded, but a repo can still choose which of your installed code runs.
