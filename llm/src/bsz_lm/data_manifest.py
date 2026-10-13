"""FineWeb binary-shard validation and immutable data manifests."""

from __future__ import annotations

import glob
import struct
from pathlib import Path
from typing import Any

from bsz_lm.canonical import digest_file, digest_value, write_json

HEADER_BYTES = 256 * 4


def read_header(path: str | Path) -> tuple[int, int, int]:
    shard = Path(path)
    with shard.open("rb") as handle:
        raw = handle.read(12)
    if len(raw) != 12:
        raise ValueError(f"truncated shard header: {shard}")
    return struct.unpack("<iii", raw)


def inspect_shard(
    path: str | Path,
    *,
    expected_magic: int,
    expected_version: int,
    hash_contents: bool,
) -> dict[str, Any]:
    shard = Path(path).resolve()
    magic, version, tokens = read_header(shard)
    if magic != expected_magic:
        raise ValueError(f"magic mismatch in {shard}: {magic}")
    if version != expected_version:
        raise ValueError(f"version mismatch in {shard}: {version}")
    expected_size = HEADER_BYTES + 2 * tokens
    size = shard.stat().st_size
    if tokens <= 0 or size != expected_size:
        raise ValueError(
            f"invalid token payload in {shard}: tokens={tokens}, size={size}, "
            f"expected={expected_size}"
        )
    result: dict[str, Any] = {
        "name": shard.name,
        "bytes": size,
        "tokens": tokens,
    }
    if hash_contents:
        result["sha256"] = digest_file(shard)
    return result


def build_data_manifest(
    root: str | Path,
    *,
    train_glob: str,
    validation_glob: str,
    expected_magic: int,
    expected_version: int,
    hash_contents: bool = True,
) -> dict[str, Any]:
    directory = Path(root).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)

    def inspect(pattern: str) -> list[dict[str, Any]]:
        matches = sorted(Path(value) for value in glob.glob(str(directory / pattern)))
        if not matches:
            raise FileNotFoundError(f"no data shards match {directory / pattern}")
        return [
            inspect_shard(
                path,
                expected_magic=expected_magic,
                expected_version=expected_version,
                hash_contents=hash_contents,
            )
            for path in matches
        ]

    body = {
        "schema": "bsz_lm_data_v1",
        "dataset": "fineweb10B",
        "binary_magic": expected_magic,
        "binary_version": expected_version,
        "content_hashes": hash_contents,
        "train_glob": train_glob,
        "validation_glob": validation_glob,
        "train": inspect(train_glob),
        "validation": inspect(validation_glob),
    }
    # The root is execution-local metadata and deliberately excluded from the identity.
    return {**body, "data_id": digest_value(body), "root": str(directory)}


def write_data_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    write_json(path, manifest)


def load_data_manifest(path: str | Path) -> dict[str, Any]:
    import json

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != "bsz_lm_data_v1":
        raise ValueError("unsupported data manifest schema")
    body = {key: item for key, item in value.items() if key not in {"data_id", "root"}}
    if value.get("data_id") != digest_value(body):
        raise ValueError("data manifest identity mismatch")
    if not value.get("content_hashes"):
        raise ValueError("plans require content-hashed data manifests")
    return value


def verify_runtime_data(
    contract: dict[str, Any], root: str | Path, *, hash_contents: bool = False
) -> None:
    """Verify that an execution-local root realizes a plan's data identity."""

    directory = Path(root).resolve()
    for split in ("train", "validation"):
        pattern = contract[f"{split}_glob"]
        actual_names = sorted(path.name for path in directory.glob(pattern))
        expected_names = [item["name"] for item in contract[split]]
        if actual_names != expected_names:
            raise ValueError(
                f"runtime {split} shard set mismatch: actual={actual_names}, "
                f"expected={expected_names}"
            )
        for expected in contract[split]:
            shard = inspect_shard(
                directory / expected["name"],
                expected_magic=int(contract["binary_magic"]),
                expected_version=int(contract["binary_version"]),
                hash_contents=hash_contents,
            )
            for field in ("name", "bytes", "tokens"):
                if shard[field] != expected[field]:
                    raise ValueError(f"runtime data mismatch for {expected['name']}: {field}")
            if hash_contents and shard["sha256"] != expected["sha256"]:
                raise ValueError(f"runtime data hash mismatch for {expected['name']}")
