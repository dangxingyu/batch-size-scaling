"""Canonical serialization and immutable scientific identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def digest_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def digest_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def digest_tree(root: str | Path, *, ignored_names: Iterable[str] = ()) -> str:
    """Hash relative paths and contents of a source tree in a stable order."""

    directory = Path(root)
    ignored = set(ignored_names)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and not any(part in ignored for part in path.relative_to(directory).parts)
    )
    hasher = hashlib.sha256()
    for path in files:
        relative = path.relative_to(directory).as_posix().encode()
        hasher.update(len(relative).to_bytes(8, "big"))
        hasher.update(relative)
        hasher.update(bytes.fromhex(digest_file(path)))
    return hasher.hexdigest()


def write_json(path: str | Path, value: Any, *, exists_ok: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if exists_ok else "x"
    with destination.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
