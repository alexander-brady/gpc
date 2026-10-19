from dataclasses import is_dataclass

import pytest

from gpc.models import PolicyConfig, WorldModelConfig

CONFIG_CLASSES = [PolicyConfig, WorldModelConfig]


@pytest.mark.parametrize("cls", CONFIG_CLASSES)
def test_config_classes_are_dataclasses(cls):
    assert is_dataclass(cls)


@pytest.mark.parametrize("cls", CONFIG_CLASSES)
def test_configs_are_keyword_only(cls):
    with pytest.raises(TypeError):
        cls(1, 2)
