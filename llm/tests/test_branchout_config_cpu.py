import dataclasses

import pytest
import yaml
from conftest import LLM

from bsz_lm.branchout.config import Config, load_config

CONFIG = LLM / "configs" / "directional_branching.yaml"


def test_shipped_config_names_its_placeholders():
    config = load_config(CONFIG)
    assert config.data_root == "<DATA_ROOT>/fineweb10B" and config.recipes == "<RECIPES_CSV>"
    assert config.curve_holdout_shards == ["fineweb_train_000001.bin"]
    # every anchor up to 12,000 leaves room for a branch window of whole k=16 periods
    for step in range(config.anchor_interval, config.parent_steps, config.anchor_interval):
        assert config.branch_horizon(step) == (1024 if step < 12000 else 992)


def test_the_recipe_csv_is_required():
    value = yaml.safe_load(CONFIG.read_text())
    value.pop("recipes")
    with pytest.raises(ValueError, match="recipe CSV"):
        Config(**value)
    with pytest.raises(ValueError, match="recipe CSV"):
        dataclasses.replace(load_config(CONFIG), recipes="")
