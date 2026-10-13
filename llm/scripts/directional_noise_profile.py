"""Per-direction gradient noise along a stored basis, and the chi buckets it defines.

The paper's directional critical batch size depends on chi_i = c_i / h_i^2, the gradient-noise
variance of a direction over the square of its curvature: directions with small chi (sharp
and quiet) leave the noise-dominated regime first.  A split by curvature order alone cannot
test that, because the top curvature directions may also be the noisiest.  These helpers turn
a profile of per-direction noise variances into column buckets that a branch can slow.
Pure functions only; the GPU stage that measures the variances lives in the driver.
"""

import json
from pathlib import Path

FLOOR = 1e-30


def chi_values(eigenvalues, variances):
    """chi_u = Var(u^T P^{1/2} g_B) / lam_u^2 for each stored direction, in stored order."""
    if len(eigenvalues) != len(variances):
        raise ValueError("one eigenvalue per variance")
    return [float(v) / max(float(lam) * float(lam), FLOOR) for lam, v in zip(eigenvalues, variances)]


def chi_buckets(chi, size):
    """Column indices of the `size` lowest-chi, highest-chi and middle-chi directions.

    Returned sorted so that a projector built from them is a plain subspace, not an order."""
    n = len(chi)
    if not 0 < size <= n // 2:
        raise ValueError("bucket size must leave the low and high buckets disjoint")
    order = sorted(range(n), key=lambda i: chi[i])
    middle = (n - size) // 2
    return {"low_chi": sorted(order[:size]),
            "high_chi": sorted(order[n - size:]),
            "mid_chi": sorted(order[middle:middle + size])}


def resolve_columns(task):
    """The stored-basis columns a branch task asks for, or None for the default prefix.

    `basis_columns` lists them directly; `basis_columns_from` names a noise-profile result and
    one of its buckets, so the selection is made by the measurement rather than by hand."""
    if task.get("basis_columns") is not None and task.get("basis_columns_from") is not None:
        raise ValueError("give basis_columns or basis_columns_from, not both")
    if task.get("basis_columns") is not None:
        return [int(c) for c in task["basis_columns"]]
    spec = task.get("basis_columns_from")
    if spec is None:
        return None
    report = json.loads(Path(spec["profile"]).read_text())
    try:
        return [int(c) for c in report["buckets"][spec["bucket"]]]
    except KeyError as error:
        raise ValueError(f"profile has no bucket {spec.get('bucket')!r}") from error
