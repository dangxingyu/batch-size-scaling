import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make_rule_sweep import all_rules, label  # noqa: E402


def test_grid_has_648_distinct_rules():
    rules = list(all_rules())
    assert len(rules) == 3**4 * 2**3
    assert len({label(512, r) for r in rules}) == len(rules)


def test_rule_application_matches_trainer():
    torch = __import__("pytest").importorskip("torch")  # noqa: F841
    from train_groupnorm_resnet import (apply_learning_rate_rule, apply_momentum_rule,
                                        apply_weight_decay_rule)
    ratio = 4096 / 256
    assert apply_learning_rate_rule(1e-3, ratio, "sqrt") == 1e-3 * 4
    assert apply_learning_rate_rule(1e-3, ratio, "linear") == 1e-3 * 16
    assert abs(apply_momentum_rule(0.9, ratio, "sample") - 0.9**16) < 1e-15
    assert apply_weight_decay_rule(1e-3, ratio, "fixed") == 1e-3


def test_label_format():
    rules = {"muon_lr_rule": "sqrt", "adam_lr_rule": "sqrt", "muon_momentum_rule": "fixed",
             "adam_beta1_rule": "fixed", "adam_beta2_rule": "fixed",
             "muon_weight_decay_rule": "fixed", "adam_weight_decay_rule": "fixed"}
    assert label(512, rules) == "b512-mlrq-alrq-mmf-b1f-b2f-mwdf-awdf"
