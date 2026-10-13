"""Authentication of external runtime components that are not vendored into this package."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_shampoo_checkout(repository: str | Path, contract: dict[str, Any]) -> Path:
    """Return the import root only if it exactly realizes the pinned Git tree."""

    root = Path(repository).resolve()
    if not (root / ".git").exists():
        raise FileNotFoundError(f"not a Git checkout: {root}")
    commit = str(contract["commit"])
    subdirectory = str(contract["subdirectory"])
    expected_tree = str(contract["package_tree"])
    package_path = f"{subdirectory}/distributed_shampoo"
    actual_tree = _git(root, "rev-parse", f"{commit}:{package_path}")
    if actual_tree != expected_tree:
        raise ValueError(f"Shampoo Git tree mismatch: {actual_tree} != {expected_tree}")
    difference = subprocess.run(["git", "diff", "--quiet", commit, "--", package_path], cwd=root)
    if difference.returncode != 0:
        raise ValueError("working-tree Shampoo package differs from the pinned commit")
    if _git(root, "status", "--porcelain", "--", package_path):
        raise ValueError("working-tree Shampoo package has uncommitted or untracked files")
    import_root = root / subdirectory
    if not (import_root / "distributed_shampoo" / "__init__.py").is_file():
        raise FileNotFoundError("pinned Shampoo package is absent from the checkout")
    return import_root
