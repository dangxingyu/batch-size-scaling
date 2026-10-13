"""Fixtures shared by the CPU tests: a recipe CSV written by the test and a tiny data manifest."""

import csv
import struct
from pathlib import Path

import pytest
import yaml

LLM = Path(__file__).resolve().parents[1]

# a synthetic 128K MuonW row with round values (not a tuned configuration)
MUON_W_128K = {
    "family": "W", "optimizer": "muon", "label": "MuonW", "batch": "128k",
    "batch_tokens": "131072", "train_steps": "13000", "matrix_algorithm": "MuonW",
    "matrix_lr": "0.02", "matrix_mu": "0.9", "matrix_beta1": "", "matrix_beta2": "",
    "shampoo_beta": "", "matrix_wd_peak": "0.05", "aux_lr_emb": "0.3",
    "aux_lr_proj": "0.004", "aux_lr_misc": "0.02", "aux_beta1": "0.9",
    "aux_beta2": "0.99", "aux_cooldown_frac": "0.5", "aux_wd_peak": "0.002",
}


def write_csv(path, rows, header=None):
    header = list(header or rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return Path(path)


@pytest.fixture
def muon_row():
    return dict(MUON_W_128K)


@pytest.fixture
def recipe_csv(tmp_path, muon_row):
    """A recipe CSV with the single 128K MuonW row."""
    return write_csv(tmp_path / "recipes.csv", [muon_row])


@pytest.fixture
def data_manifest(tmp_path):
    """A content-hashed manifest of two tiny shards with the FineWeb header."""
    from bsz_lm.data_manifest import build_data_manifest, write_data_manifest

    data = yaml.safe_load((LLM / "configs/protocol.yaml").read_text())["data"]
    shards = tmp_path / "shards"
    shards.mkdir()
    for name in ("fineweb_train_000001.bin", "fineweb_val_000000.bin"):
        header = struct.pack("<iii", data["binary_magic"], data["binary_version"], 8) + bytes(1012)
        (shards / name).write_bytes(header + bytes(16))
    manifest = build_data_manifest(shards, train_glob=data["train_glob"],
                                   validation_glob=data["validation_glob"],
                                   expected_magic=data["binary_magic"],
                                   expected_version=data["binary_version"])
    path = tmp_path / "manifest.json"
    write_data_manifest(path, manifest)
    return path
