import json
import math

import pytest
from conftest import LLM, write_csv

from bsz_lm.canonical import digest_value
from bsz_lm.cli import main as cli
from bsz_lm.collection import collect_results, write_results_csv
from bsz_lm.planning import CaseSelection, build_plan
from bsz_lm.protocol import load_protocol
from bsz_lm.recipes import RECIPE_COLUMNS, Recipe, load_recipes

PROTOCOL = LLM / "configs" / "protocol.yaml"


def test_protocol_token_budget():
    protocol = load_protocol(PROTOCOL)
    assert protocol.token_horizon == 1_703_936_000
    assert [protocol.train_steps(batch) for batch in protocol.batches] == [13000, 3250, 1625, 813]


def test_a_single_row_recipe_csv_loads(recipe_csv):
    (recipe,) = load_recipes(recipe_csv)
    assert (recipe.family, recipe.optimizer, recipe.batch_tokens) == ("W", "muon", 131072)
    assert recipe.train_steps == 13000 and isinstance(recipe.train_steps, int)
    assert recipe.matrix_lr == 0.02 and recipe.matrix_mu == 0.9
    assert recipe.matrix_beta1 is None and recipe.matrix_beta2 is None
    assert recipe.shampoo_beta is None
    # the identity is the hyperparameters alone
    assert set(recipe.scientific_dict()) == set(RECIPE_COLUMNS)
    assert recipe.recipe_id == digest_value(recipe.scientific_dict())


def test_rows_for_every_optimizer_and_both_families(tmp_path, muon_row):
    rows = [muon_row]
    common = {"matrix_mu": "", "shampoo_beta": "", "matrix_beta1": "0.9", "matrix_beta2": "0.95"}
    for optimizer in ("adam", "lion", "shampoo"):
        rows.append({**muon_row, **common, "optimizer": optimizer, "label": optimizer})
    rows.append({**muon_row, **common, "optimizer": "klsoap", "label": "klsoap",
                 "shampoo_beta": "0.95"})
    rows.append({**muon_row, "family": "H", "label": "MuonH", "batch": "2m",
                 "batch_tokens": "2097152", "train_steps": "813", "matrix_wd_peak": "0",
                 "aux_wd_peak": "0"})
    recipes = load_recipes(write_csv(tmp_path / "all.csv", rows))
    assert [r.optimizer for r in recipes] == ["muon", "adam", "lion", "shampoo", "klsoap", "muon"]
    assert recipes[-1].family == "H" and recipes[-1].train_steps == math.ceil(1_703_936_000 / 2097152)
    assert recipes[4].shampoo_beta == 0.95 and recipes[4].matrix_mu is None


@pytest.mark.parametrize("change, message", [
    ({"matrix_beta1": "0.9"}, "does not use matrix_beta1"),       # Muon has no beta1
    ({"shampoo_beta": "0.9"}, "does not use shampoo_beta"),
    ({"matrix_mu": ""}, "requires matrix_mu"),
    ({"optimizer": "adam"}, "adam does not use matrix_mu"),      # Adam row left in Muon form
    ({"optimizer": "adam", "matrix_mu": ""}, "requires matrix_beta1"),
    ({"family": "H"}, "weight decay"),                           # HyperBall has no decay
    ({"train_steps": "12999"}, "train_steps must be 13000"),
    ({"batch_tokens": "100000"}, "unexpected Track-3 batch"),
    ({"optimizer": "sgd"}, "unknown optimizer"),
    ({"matrix_lr": "nan"}, "finite"),
])
def test_invalid_rows_are_rejected(tmp_path, muon_row, change, message):
    path = write_csv(tmp_path / "bad.csv", [{**muon_row, **change}])
    with pytest.raises(ValueError, match=message):
        load_recipes(path)


def test_columns_must_be_exactly_the_hyperparameters(tmp_path, muon_row):
    extra = write_csv(tmp_path / "extra.csv", [{**muon_row, "notes": "x"}])
    with pytest.raises(ValueError, match="unknown=\\['notes'\\]"):
        load_recipes(extra)
    missing = {key: value for key, value in muon_row.items() if key != "aux_wd_peak"}
    with pytest.raises(ValueError, match="missing=\\['aux_wd_peak'\\]"):
        load_recipes(write_csv(tmp_path / "missing.csv", [missing]))
    with pytest.raises(ValueError, match="no rows"):
        load_recipes(write_csv(tmp_path / "empty.csv", [], header=RECIPE_COLUMNS))
    # column order is free
    reordered = write_csv(tmp_path / "reordered.csv", [muon_row], header=sorted(muon_row))
    assert load_recipes(reordered)[0] == load_recipes(write_csv(tmp_path / "a.csv", [muon_row]))[0]


def test_verify_recipes_and_plan_need_a_user_csv(recipe_csv, capsys):
    cli(["verify-recipes", "--recipes", str(recipe_csv)])
    summary = json.loads(capsys.readouterr().out)
    assert summary == {"recipes": 1, "families": ["W"], "optimizers": ["muon"], "batches": [131072]}
    for command in (["verify-recipes"], ["plan", "--data-manifest", "m.json", "--output", "p.json"]):
        with pytest.raises(SystemExit):
            cli(command)


def test_plan_and_collect_carry_only_the_run_results(tmp_path, recipe_csv, data_manifest):
    plan = build_plan(PROTOCOL, recipe_csv, data_manifest)
    (case,) = plan["cases"]
    assert plan["stage"] == "tuned" and case["case_id"].startswith("tuned-w-muon-b131072-")
    assert set(case) == {"schema", "stage", "protocol_id", "data_id", "source_id", "recipe",
                         "effective_train_steps", "seed", "case_id", "run_fingerprint"}
    assert set(case["recipe"]) == set(RECIPE_COLUMNS)
    assert Recipe(**case["recipe"]) == load_recipes(recipe_csv)[0]
    with pytest.raises(ValueError, match="selection is empty"):
        build_plan(PROTOCOL, recipe_csv, data_manifest, selection=CaseSelection(families=("H",)))

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    results = tmp_path / "results"
    with pytest.raises(FileNotFoundError, match="has no result"):
        collect_results(plan_path, results)
    (results / case["case_id"]).mkdir(parents=True)
    result = {"schema": "bsz_lm_result_v1", "status": "complete", "plan_id": plan["plan_id"],
              "case_id": case["case_id"], "run_fingerprint": case["run_fingerprint"],
              "terminal_validation_loss": 3.5}
    (results / case["case_id"] / "result.json").write_text(json.dumps(result))
    rows = collect_results(plan_path, results)
    assert rows == [{"case_id": case["case_id"], "family": "W", "optimizer": "muon",
                     "batch_tokens": 131072, "terminal_validation_loss": 3.5,
                     "run_fingerprint": case["run_fingerprint"]}]
    write_results_csv(tmp_path / "results.csv", rows)
    header = (tmp_path / "results.csv").read_text().splitlines()[0]
    assert header == "case_id,family,optimizer,batch_tokens,terminal_validation_loss,run_fingerprint"
    # a result written for another plan is refused
    (results / case["case_id"] / "result.json").write_text(json.dumps({**result, "plan_id": "x"}))
    with pytest.raises(ValueError, match="result mismatch"):
        collect_results(plan_path, results)
