import csv
import importlib.util
import json
import math
import subprocess
import sys

import pytest
from conftest import LLM, write_csv

from bsz_lm.recipes import RECIPE_COLUMNS

spec = importlib.util.spec_from_file_location("make_rule_sweep", LLM / "scripts/make_rule_sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

HPARAMS = ("matrix_lr", "matrix_mu", "matrix_wd_peak", "aux_lr_emb", "aux_lr_proj", "aux_lr_misc",
           "aux_beta1", "aux_beta2", "aux_wd_peak", "aux_cooldown_frac")


def test_216_distinct_rules_per_target_batch(recipe_csv):
    base = sweep.reference_recipe(recipe_csv)
    codes = sweep.all_rules()
    assert len(codes) == len(set(codes)) == 216
    assert len({sweep.pattern_id(c) for c in codes}) == 216
    for batch in sweep.TARGET_BATCHES:
        recipes = [sweep.transfer(base, c, batch) for c in codes]
        for recipe in recipes:
            recipe.validate()
            assert recipe.train_steps == math.ceil(1_703_936_000 / batch)
        assert len({tuple(getattr(r, h) for h in HPARAMS) for r in recipes}) == 216


def test_all_fixed_at_the_reference_batch_is_the_reference(recipe_csv):
    base = sweep.reference_recipe(recipe_csv)
    same = sweep.transfer(base, "FFFFFF", sweep.REFERENCE_BATCH)
    assert same.train_steps == base.train_steps == 13000
    for h in HPARAMS:
        assert getattr(same, h) == getattr(base, h)
    # every rule, not only all-fixed, is the identity at ratio 1
    for code in sweep.all_rules():
        moved = sweep.transfer(base, code, sweep.REFERENCE_BATCH)
        assert all(math.isclose(getattr(moved, h), getattr(base, h)) for h in HPARAMS)


def test_rule_laws_and_labels(recipe_csv):
    # pattern ids of three rules in the paper's sweep (hashes of the laws, not of any values)
    assert sweep.pattern_id("FFFFFF") == "e333be792b9e"
    assert sweep.pattern_id("FFFFFE") == "9bcb9f5adcbe"
    assert sweep.pattern_id("SLSEEE") == "dc79b88a81ad"
    base = sweep.reference_recipe(recipe_csv)
    # SLSEEE at 1M tokens: ratio 8, 1625 steps, EMA exponent 13000 / 1625 = 8
    moved = sweep.transfer(base, "SLSEEE", 1048576)
    expected = {"matrix_lr": 0.02 * math.sqrt(8), "matrix_wd_peak": 0.05 * 8,
                "aux_lr_emb": 0.3 * math.sqrt(8), "aux_lr_proj": 0.004 * math.sqrt(8),
                "aux_lr_misc": 0.02 * math.sqrt(8), "matrix_mu": 0.9 ** 8,
                "aux_beta1": 0.9 ** 8, "aux_beta2": 0.99 ** 8, "aux_wd_peak": 0.002,
                "aux_cooldown_frac": 0.5}
    for name, value in expected.items():
        assert math.isclose(getattr(moved, name), value, rel_tol=1e-12), name
    assert (moved.train_steps, moved.batch, moved.label) == (1625, "1m", "MuonW-transfer-SLSEEE")
    # LFLFEF at 2M: ratio 16, 813 steps, EMA exponent 13000 / 813; a seventh letter L scales aux wd
    moved = sweep.transfer(base, "LFLFEFL", 2097152)
    assert math.isclose(moved.matrix_lr, 0.02 * 16, rel_tol=1e-12)
    assert moved.matrix_wd_peak == 0.05 and moved.matrix_mu == 0.9 and moved.aux_beta2 == 0.99
    assert math.isclose(moved.aux_beta1, 0.9 ** (13000 / 813), rel_tol=1e-12)
    assert math.isclose(moved.aux_lr_proj, 0.004 * 16, rel_tol=1e-12)
    assert math.isclose(moved.aux_wd_peak, 0.002 * 16, rel_tol=1e-12)
    for bad in ("FFFFF", "FFFFFS", "XFFFFF", "FFFFFFE"):
        with pytest.raises(ValueError):
            sweep.transfer(base, bad, 262144)


def test_the_reference_row_must_be_unique(tmp_path, muon_row):
    adam = {**muon_row, "optimizer": "adam", "matrix_mu": "", "matrix_beta1": "0.9",
            "matrix_beta2": "0.95"}
    with pytest.raises(ValueError, match="found 0"):
        sweep.reference_recipe(write_csv(tmp_path / "none.csv", [adam]))
    with pytest.raises(ValueError, match="found 2"):
        sweep.reference_recipe(write_csv(tmp_path / "two.csv",
                                         [muon_row, {**muon_row, "matrix_lr": "0.03"}]))
    # other rows may sit next to it
    assert sweep.reference_recipe(write_csv(tmp_path / "both.csv", [adam, muon_row])).matrix_lr == 0.02


def test_overrides_turn_a_rule_into_a_retuning_candidate(recipe_csv):
    base = sweep.reference_recipe(recipe_csv)
    start = sweep.transfer(base, "FFFFFF", 262144)
    moved = sweep.apply_overrides(start, ["matrix_lr=0.035", "train_steps=6500"])
    assert moved.matrix_lr == 0.035 and isinstance(moved.train_steps, int)
    for h in HPARAMS:
        if h != "matrix_lr":
            assert getattr(moved, h) == getattr(start, h)
    moved.validate()
    for bad in (["no_such_field=1"], ["matrix_lr"], ["optimizer=adam"], ["matrix_beta1=0.9"]):
        with pytest.raises(ValueError):
            sweep.apply_overrides(start, bad)


def _run(*arguments):
    return subprocess.run([sys.executable, str(LLM / "scripts/make_rule_sweep.py"), *arguments],
                          cwd=LLM, capture_output=True, text=True)


def test_csv_holds_the_rule_and_the_hyperparameters_only(tmp_path, recipe_csv):
    out = tmp_path / "rules.csv"
    done = _run("--recipes", str(recipe_csv), "--csv", str(out), "--rules", "FFFFFF,SLSEEE",
                "--batches", "262144,1048576")
    assert done.returncode == 0, done.stderr
    with out.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == ["rule", "pattern_id", *RECIPE_COLUMNS]
    assert [(r["rule"], r["batch_tokens"]) for r in rows] == [
        ("FFFFFF", "262144"), ("FFFFFF", "1048576"), ("SLSEEE", "262144"), ("SLSEEE", "1048576")]
    assert {r["pattern_id"] for r in rows} == {"e333be792b9e", "dc79b88a81ad"}
    assert all(r["matrix_beta1"] == "" for r in rows)
    # --recipes has no default
    missing = _run("--csv", str(tmp_path / "other.csv"))
    assert missing.returncode != 0 and "--recipes" in missing.stderr


def test_candidate_plan_is_accepted_by_the_trainer(tmp_path, recipe_csv, data_manifest):
    from bsz_lm import train

    plan_path = tmp_path / "candidate.json"
    done = _run("--recipes", str(recipe_csv), "--plan", str(plan_path),
                "--data-manifest", str(data_manifest), "--batches", "262144",
                "--rules", "FFFFFF", "--set", "matrix_lr=0.035")
    assert done.returncode == 0, done.stderr
    plan = json.loads(plan_path.read_text())
    assert len(plan["cases"]) == 1
    case = train._validate_plan(plan, 0)
    recipe = train._recipe(case)
    assert recipe.matrix_lr == 0.035 and recipe.batch_tokens == 262144
    assert recipe.matrix_wd_peak == 0.05 and recipe.train_steps == 6500
    assert case["experiment"]["overrides"] == ["matrix_lr=0.035"]
    assert set(case["recipe"]) == set(RECIPE_COLUMNS)
