"""Wide Hessian bases for the rank sweep: a fully reorthogonalized Lanczos whose
Krylov vectors are sharded across the data-parallel ranks (bf16 storage, fp32
arithmetic in column chunks), a sharded orthonormal basis with the RankBands
projection interface, and an orthonormal random basis of the same width.

Every rank holds the same replicated Lanczos vector (the Hessian-vector product
is data parallel and all-reduced), but only its contiguous slice of every stored
vector. Reorthogonalization coefficients are all-reduced, and the corrected
slice is all-gathered back into the replicated vector, so all ranks stay
bitwise consistent. Falls back to a single shard without torch.distributed.
"""

from __future__ import annotations

import math
import time

import torch
import torch.distributed as dist

CHUNK = 128  # Krylov/basis columns converted to fp32 at a time


def world_info(device):
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def shard_bounds(dimension, rank, world):
    sizes = [dimension // world + (1 if i < dimension % world else 0) for i in range(world)]
    lo = sum(sizes[:rank])
    return lo, lo + sizes[rank], sizes


def all_reduce_sum(value):
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def all_gather_uneven(local, sizes, out=None):
    """Concatenate per-rank slices of possibly unequal length into one vector."""
    world = len(sizes)
    if world == 1:
        if out is None:
            return local.clone()
        out.copy_(local)
        return out
    width = max(sizes)
    padded = torch.zeros(width, device=local.device, dtype=local.dtype)
    padded[: local.numel()].copy_(local)
    gathered = torch.empty(world * width, device=local.device, dtype=local.dtype)
    dist.all_gather_into_tensor(gathered, padded)
    pieces = [gathered[i * width:i * width + sizes[i]] for i in range(world)]
    if out is None:
        return torch.cat(pieces)
    torch.cat(pieces, out=out)
    return out


class ShardedBasis:
    """Orthonormal columns of a (dimension x width) matrix, stored as this rank's slice."""

    def __init__(self, shard, dimension, rank, world, sizes=None):
        self.shard = shard.contiguous()
        self.dimension, self.rank, self.world = int(dimension), int(rank), int(world)
        lo, hi, computed = shard_bounds(self.dimension, self.rank, self.world)
        self.sizes = sizes or computed
        if self.shard.shape[0] != hi - lo:
            raise ValueError("shard height does not match the rank's slice")
        self.lo, self.hi = lo, hi

    @property
    def width(self):
        return int(self.shard.shape[1])

    def narrow(self, width):
        if not 0 < width <= self.width:
            raise ValueError("requested rank exceeds the stored basis")
        return ShardedBasis(self.shard[:, :width], self.dimension, self.rank, self.world, self.sizes)

    def select(self, columns):
        """The listed stored columns, in the given order, as a new basis.

        Unlike `narrow`, which keeps a prefix (the top-r Ritz vectors), this picks an arbitrary
        subset, which is how a branch slows a subspace chosen by something other than
        curvature order, such as the noise-to-curvature ratio of each direction.
        """
        index = torch.as_tensor(list(columns), dtype=torch.long, device=self.shard.device)
        if index.numel() == 0 or int(index.min()) < 0 or int(index.max()) >= self.width \
                or index.unique().numel() != index.numel():
            raise ValueError("column selection must be a non-empty set of distinct stored columns")
        return ShardedBasis(self.shard[:, index].contiguous(), self.dimension, self.rank, self.world, self.sizes)

    def coefficients(self, value):
        """U^T v, all-reduced (fp32)."""
        piece = value[self.lo:self.hi].float()
        out = torch.zeros(self.width, device=value.device, dtype=torch.float32)
        for start in range(0, self.width, CHUNK):
            out[start:start + CHUNK] = self.shard[:, start:start + CHUNK].float().T @ piece
        return all_reduce_sum(out)

    def expand(self, coefficients):
        """U c, gathered to the replicated full vector (fp32)."""
        local = torch.zeros(self.hi - self.lo, device=self.shard.device, dtype=torch.float32)
        for start in range(0, self.width, CHUNK):
            local += self.shard[:, start:start + CHUNK].float() @ coefficients[start:start + CHUNK]
        return all_gather_uneven(local, self.sizes)

    def project(self, value):
        return self.expand(self.coefficients(value))

    def gram_error(self):
        gram = torch.zeros(self.width, self.width, device=self.shard.device)
        for a in range(0, self.width, CHUNK):
            block = self.shard[:, a:a + CHUNK].float()
            for b in range(0, self.width, CHUNK):
                gram[a:a + CHUNK, b:b + CHUNK] = block.T @ self.shard[:, b:b + CHUNK].float()
        all_reduce_sum(gram)
        return float((gram - torch.eye(self.width, device=gram.device)).abs().max())

    def gather_columns(self, count):
        """The first `count` columns as a replicated dense (dimension x count) fp32 matrix."""
        columns = [all_gather_uneven(self.shard[:, j].float(), self.sizes) for j in range(count)]
        return torch.stack(columns, dim=1).contiguous()

    def state(self):
        return {"shard": self.shard.cpu(), "dimension": self.dimension, "rank": self.rank,
                "world": self.world, "sizes": list(self.sizes), "dtype": str(self.shard.dtype)}

    @classmethod
    def from_state(cls, state, device):
        return cls(state["shard"].to(device), state["dimension"], state["rank"], state["world"], state["sizes"])


class WideRankBands:
    """RankBands-compatible projector over a sharded basis: one band per width, plus the complement."""

    def __init__(self, basis, widths):
        if not widths or any(not isinstance(w, int) or w <= 0 for w in widths) or sum(widths) != basis.width:
            raise ValueError("positive band widths must partition the retained basis")
        self.full = basis
        self.bases = {}
        start = 0
        for index, width in enumerate(widths):
            self.bases[f"rank_band_{index}"] = (start, start + width)
            start += width
        self.orthogonality_error = basis.gram_error()

    def _band(self, band):
        start, stop = self.bases[band]
        if start == 0:
            return self.full.narrow(stop)
        return ShardedBasis(self.full.shard[:, start:stop], self.full.dimension, self.full.rank,
                            self.full.world, self.full.sizes)

    def project(self, value, band):
        if band == "remainder":
            return value.float() - self.full.project(value)
        return self._band(band).project(value)

    def leakage(self, value, band):
        return float((value - self.project(value, band)).square().sum()
                     / value.square().sum().clamp_min(1e-30))


@torch.no_grad()
def wide_lanczos(hvp, *, dimension, top_rank, iterations, device, seed, krylov_dtype=torch.bfloat16,
                 audit_rank=None, trace_probes=8):
    """Sharded full-reorthogonalization Lanczos; returns (ShardedBasis of the top_rank
    Ritz vectors by algebraic value, report). Same recurrence and restart policy as
    bsz_lm.branchout.spectral.top_ritz; storage in `krylov_dtype`, arithmetic fp32."""
    rank, world = world_info(device)
    if not 0 < top_rank < dimension or iterations <= top_rank:
        raise ValueError("need 0 < top_rank < dimension and iterations > top_rank")
    lo, hi, sizes = shard_bounds(dimension, rank, world)
    count = min(iterations, dimension)
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(dimension, device=device, generator=generator)
    q /= q.norm()
    krylov = torch.empty((hi - lo, count), device=device, dtype=krylov_dtype)
    diagonal, off_diagonal = [], []
    restarts = 0
    beta = torch.zeros((), device=device)
    previous = torch.zeros_like(q)
    started = time.perf_counter()

    def reorthogonalize(vector, active):
        piece = vector[lo:hi].float()
        for _ in range(2):
            coefficients = torch.zeros(active, device=device)
            for start in range(0, active, CHUNK):
                stop = min(start + CHUNK, active)
                coefficients[start:stop] = krylov[:, start:stop].float().T @ piece
            all_reduce_sum(coefficients)
            for start in range(0, active, CHUNK):
                stop = min(start + CHUNK, active)
                piece -= krylov[:, start:stop].float() @ coefficients[start:stop]
        return all_gather_uneven(piece, sizes)

    for j in range(count):
        krylov[:, j].copy_(q[lo:hi].to(krylov_dtype))
        product = hvp(q).detach().float()
        if product.shape != q.shape or not torch.isfinite(product).all():
            raise FloatingPointError("invalid Hessian-vector product")
        residual = product - beta * previous
        alpha = torch.dot(q, residual)
        residual -= alpha * q
        residual = reorthogonalize(residual, j + 1)
        next_beta = residual.norm()
        diagonal.append(alpha)
        if j + 1 == count:
            break
        if float(next_beta) <= 1e-6 * max(1.0, float(product.norm())):
            candidate = torch.randn(dimension, device=device, generator=generator)
            candidate = reorthogonalize(candidate, j + 1)
            norm = candidate.norm()
            if float(norm) < 1e-6:
                break
            previous, q = q, candidate / norm
            beta = torch.zeros_like(next_beta)
            restarts += 1
        else:
            previous, q = q, residual / next_beta
            beta = next_beta
        off_diagonal.append(beta)
    count = len(diagonal)
    if count < top_rank:
        raise RuntimeError("insufficient independent Ritz vectors")
    small = torch.diag(torch.stack(diagonal).double().cpu())
    if count > 1:
        off = torch.stack(off_diagonal[: count - 1]).double().cpu()
        small += torch.diag(off, 1) + torch.diag(off, -1)
    values, rotations = torch.linalg.eigh(small)
    # Stochastic Lanczos quadrature from this single random start: weights tau_i^2 = first
    # components of T's eigenvectors; d * sum_i tau_i^2 [theta_i >= t] estimates the number of
    # eigenvalues above t and d * sum_i tau_i^2 theta_i [theta_i >= t] their summed mass.
    weights = rotations[0].double() ** 2
    slq_order = torch.argsort(values, descending=True)
    slq_theta = values[slq_order]
    slq_count = torch.cumsum(weights[slq_order], 0) * dimension
    slq_mass = torch.cumsum(weights[slq_order] * slq_theta, 0) * dimension
    order = torch.argsort(values, descending=True)[:top_rank]
    selected = values[order].float()
    rotation = rotations[:, order].float().to(device)
    shard = torch.empty((hi - lo, top_rank), device=device, dtype=krylov_dtype)
    for out in range(0, top_rank, CHUNK):
        block = torch.zeros((hi - lo, min(CHUNK, top_rank - out)), device=device, dtype=torch.float32)
        for start in range(0, count, CHUNK):
            stop = min(start + CHUNK, count)
            block += krylov[:, start:stop].float() @ rotation[start:stop, out:out + block.shape[1]]
        shard[:, out:out + block.shape[1]] = block.to(krylov_dtype)
        del block
    krylov = None  # release the Krylov shard (the closure above is no longer used)
    basis = ShardedBasis(shard, dimension, rank, world, sizes)
    del shard
    lanczos_seconds = time.perf_counter() - started
    # Independent HVP audit of the retained pairs (all, or the first audit_rank).
    audit = top_rank if audit_rank is None else min(audit_rank, top_rank)
    radius = float(values.abs().max())
    absolute, relative = [], []
    block_sq = 0.0
    for j in range(audit):
        u = all_gather_uneven(basis.shard[:, j].float(), sizes)
        r = hvp(u) - selected[j].to(device) * u
        a = float(r.norm())
        absolute.append(a)
        relative.append(a / max(abs(float(selected[j])), radius * 1e-4, 1e-20))
        block_sq += a * a
    trace = None
    if trace_probes:
        estimates = []
        for _ in range(trace_probes):
            z = torch.randint(0, 2, (dimension,), device=device, generator=generator).float() * 2 - 1
            estimates.append(float(torch.dot(z, hvp(z))))
        trace = {"mean": sum(estimates) / len(estimates), "probes": trace_probes,
                 "stderr": (torch.tensor(estimates).std(unbiased=True) / math.sqrt(len(estimates))).item()
                 if len(estimates) > 1 else None}
    positive = values.clamp_min(0)
    cumulative = torch.cumsum(selected.double().clamp_min(0), 0)
    slq_trace = float((weights * values).sum() * dimension)
    slq = {"theta": slq_theta.tolist(), "count_above": slq_count.tolist(), "mass_above": slq_mass.tolist(),
           "trace_estimate": slq_trace, "note": "single random start; count/mass of eigenvalues >= theta"}
    for target in (0.3, 0.5):
        reference = trace["mean"] if trace else slq_trace
        hit = [(c, m) for c, m in zip(slq_count.tolist(), slq_mass.tolist()) if reference and m / reference >= target]
        slq[f"rank_for_mass_{int(100 * target)}"] = hit[0][0] if hit else None
    return basis, {
        "slq": slq,
        "algorithm": "sharded_fully_reorthogonalized_lanczos_with_random_restarts",
        "ordering": "largest_algebraic_signed_eigenvalues",
        "krylov_dtype": str(krylov_dtype), "world": world,
        "iterations": count, "restarts": restarts, "top_rank": top_rank, "dimension": dimension,
        "eigenvalues": selected.tolist(), "all_ritz_values": values.tolist(),
        "absolute_residuals": absolute, "relative_residuals": relative,
        "audited_rank": audit,
        "max_relative_residual": max(relative) if relative else None,
        "converged_at_0.01": sum(1 for r in relative if r <= 0.01),
        "converged_at_0.1": sum(1 for r in relative if r <= 0.1),
        "block_residual": math.sqrt(block_sq),
        "orthogonality_error": basis.gram_error(),
        "cumulative_positive_mass_over_ritz": (cumulative / positive.sum().double().clamp_min(1e-30)).tolist(),
        "hutchinson_trace": trace,
        "lanczos_seconds": lanczos_seconds,
        "global_top_ordering_certified": count == dimension,
    }


@torch.no_grad()
def random_orthonormal_basis(dimension, width, device, seed, dtype=torch.bfloat16, rows=1 << 20):
    """Sharded orthonormal random basis (Cholesky QR twice); the same on every launch.
    Stored in `dtype`, arithmetic in fp32 over row blocks of `rows`, so the peak is one
    row block (not a second full-width fp32 copy: 30 GiB at width 768 on a 1/8 shard)."""
    rank, world = world_info(device)
    lo, hi, sizes = shard_bounds(dimension, rank, world)
    # Each rank draws only its slice; the draw is reproducible for a fixed world size.
    generator = torch.Generator(device=device).manual_seed(seed * 1009 + rank)
    shard = torch.empty((hi - lo, width), device=device, dtype=dtype)
    for start in range(0, hi - lo, rows):
        count = min(rows, hi - lo - start)
        shard[start:start + count] = torch.randn(count, width, device=device, generator=generator, dtype=torch.float32).to(dtype)
    for _ in range(2):
        gram = torch.zeros((width, width), device=device)
        for start in range(0, hi - lo, rows):
            block = shard[start:start + rows].float()
            gram += block.T @ block
        all_reduce_sum(gram)
        factor = torch.linalg.cholesky(gram.double()).float()
        inverse = torch.linalg.inv(factor.double()).T.float()  # (L^T)^{-1}, applied as a GEMM
        for start in range(0, hi - lo, rows):
            block = shard[start:start + rows].float()
            shard[start:start + rows] = (block @ inverse).to(dtype)
    return ShardedBasis(shard, dimension, rank, world, sizes)


def _apply_inverse_factor(shard, factor, rows=1 << 20):
    """X <- X R^{-1} with R = L^T from a Cholesky factor L of X^T X, applied as a GEMM.

    torch.linalg.solve_triangular against a million-column right-hand side takes seconds per
    call on the GPU (94 s per 16-column mapping at full size); the factor is only
    (width x width), so inverting it once and multiplying is exact to working precision and
    orders of magnitude faster.  Done in row blocks so the temporary stays small.
    """
    inverse = torch.linalg.inv(factor.double()).T.to(shard.dtype)  # (L^{-1})^T = (L^T)^{-1}
    for start in range(0, shard.shape[0], rows):
        block = shard[start:start + rows]
        shard[start:start + rows] = block @ inverse
    return shard


@torch.no_grad()
def basis_overlap(a, b):
    """Mean squared principal cosine between the spans of two sharded bases of equal width
    (1 = same subspace), from the singular values of a^T b."""
    if a.width != b.width or a.dimension != b.dimension:
        raise ValueError("overlap needs two bases of the same width and dimension")
    cross = torch.zeros((a.width, b.width), device=a.shard.device)
    for start in range(0, a.width, CHUNK):
        cross[start:start + CHUNK] = a.shard[:, start:start + CHUNK].float().T @ b.shard.float()
    all_reduce_sum(cross)
    cos = torch.linalg.svdvals(cross.double()).clamp(max=1.0)
    return float((cos ** 2).mean())


@torch.no_grad()
def subspace_refresh(hvp, basis, *, shift=0.0, audit=16):
    """One warm-started block subspace-iteration pass on H + shift*I (the shift keeps the
    iteration aimed at the algebraically largest eigenvalues when H has negative ones):
    Y = (H + shift I) U, new basis = orth(Y) (Cholesky QR twice, sharded), Rayleigh quotients
    of H on the old subspace as value estimates, an independent residual audit of the first
    `audit` new columns, and the overlap with the previous span."""
    rank, world = basis.rank, basis.world
    lo, hi, sizes = basis.lo, basis.hi, basis.sizes
    width = basis.width
    device = basis.shard.device
    y_shard = torch.zeros((hi - lo, width), device=device, dtype=torch.float32)
    gram_old = torch.zeros((width, width), device=device)
    started = time.perf_counter()
    for j in range(width):
        u = all_gather_uneven(basis.shard[:, j].float(), sizes)
        y = hvp(u)
        gram_old[:, j] = basis.coefficients(y)  # U^T H u_j (all-reduced)
        y_shard[:, j] = y[lo:hi] + shift * u[lo:hi]
    rayleigh = torch.linalg.eigvalsh(0.5 * (gram_old + gram_old.T).double()).flip(0).float()
    # orthonormalize Y (two Cholesky-QR passes) into the new sharded basis
    shard = y_shard
    for _ in range(2):
        gram = all_reduce_sum(shard.T @ shard)
        factor = torch.linalg.cholesky(gram.double()).float()
        shard = _apply_inverse_factor(shard, factor)
    new = ShardedBasis(shard.to(basis.shard.dtype), basis.dimension, rank, world, sizes)
    del y_shard, shard
    # overlap with the previous span: singular values of U_old^T U_new
    cross = torch.zeros((width, width), device=device)
    for start in range(0, width, CHUNK):
        cross[start:start + CHUNK] = basis.shard[:, start:start + CHUNK].float().T @ new.shard.float()
    all_reduce_sum(cross)
    cos = torch.linalg.svdvals(cross.double()).clamp(max=1.0)
    residuals = []
    for j in range(min(audit, width)):
        u = all_gather_uneven(new.shard[:, j].float(), sizes)
        y = hvp(u)
        theta = float(torch.dot(u, y))
        residuals.append(float((y - theta * u).norm() / max(abs(theta), 1e-20)))
    return new, {"method": "warm_block_subspace_iteration_one_pass", "width": width, "shift": shift,
                 "rayleigh_values_old_span": rayleigh.tolist(), "audit_relative_residuals": residuals,
                 "overlap_with_previous": float((cos ** 2).mean()), "min_principal_cos": float(cos.min()),
                 "orthogonality_error": new.gram_error(), "hvp_calls": width + min(audit, width),
                 "seconds": time.perf_counter() - started}


@torch.no_grad()
def basis_drift(hvp, basis, *, ranks, iterations, seed, trace_probes=16, audit=16, krylov_dtype=torch.bfloat16):
    """How far the top eigenspace has moved since `basis` (nested top-r Ritz vectors of the
    anchor Hessian, columns ordered by value) was computed, measured against the Hessian at
    the current weights. With Y = H U0 and A = U0^T H U0 (all-reduced, fp32), per nested r:

      rayleigh_mass  tr(A_rr): curvature the frozen top-r still sees
      residual       ||H U0_r - U0_r A_rr||_F / ||H U0_r||_F: how far U0_r is from invariant
      overlap        ||U0_r^T U_r||_F^2 / r against a fresh sharded Lanczos basis U (mean
                     squared principal cosine; 1 = unchanged, r/d = unrelated)
      min_cos        smallest principal cosine of the same pair
      retention      tr(A_rr) / sum_{i<=r} lambda_i(H): frozen top-r curvature over fresh top-r
      share_frozen / share_fresh   the two curvature sums over the Hutchinson trace of H

    The fresh solve reports its own convergence (audit residuals of its first `audit` columns).
    Nothing here touches the training state; the caller supplies an HVP on a copy."""
    device = basis.shard.device
    width = basis.width
    sizes = basis.sizes
    ranks = [int(r) for r in ranks if 0 < int(r) <= width]
    started = time.perf_counter()
    gram = torch.zeros((width, width), device=device)
    norms_sq = torch.zeros(width, device=device)
    for j in range(width):
        u = all_gather_uneven(basis.shard[:, j].float(), sizes)
        y = hvp(u)
        gram[:, j] = basis.coefficients(y)
        norms_sq[j] = torch.dot(y, y)
    gram = 0.5 * (gram + gram.T)
    fresh, report = wide_lanczos(hvp, dimension=basis.dimension, top_rank=max(ranks), iterations=iterations,
                                 device=device, seed=seed, krylov_dtype=krylov_dtype, audit_rank=audit,
                                 trace_probes=trace_probes)
    cross = torch.zeros((width, fresh.width), device=device)
    for start in range(0, width, CHUNK):
        cross[start:start + CHUNK] = basis.shard[:, start:start + CHUNK].float().T @ fresh.shard.float()
    all_reduce_sum(cross)
    values = torch.tensor(report["eigenvalues"], device=device)
    trace = (report.get("hutchinson_trace") or {}).get("mean")
    per_rank = {}
    for r in ranks:
        block = gram[:r, :r]
        leak = float((norms_sq[:r].sum() - (gram[:r, :r] ** 2).sum()).clamp_min(0))
        cos = torch.linalg.svdvals(cross[:r, :r].double()).clamp(max=1.0)
        mass, fresh_mass = float(block.trace()), float(values[:r].sum())
        per_rank[str(r)] = {"rayleigh_mass": mass, "fresh_mass": fresh_mass,  # str keys: same in memory and in JSON
                       "residual": math.sqrt(leak / max(float(norms_sq[:r].sum()), 1e-30)),
                       "overlap": float((cos ** 2).mean()), "min_cos": float(cos.min()),
                       "retention": mass / fresh_mass if fresh_mass else None,
                       "share_frozen": mass / trace if trace else None,
                       "share_fresh": fresh_mass / trace if trace else None}
    del fresh, cross
    return {"method": "frozen_rayleigh_block_vs_fresh_sharded_lanczos", "width": width, "ranks": ranks,
            "iterations": report["iterations"], "per_rank": per_rank,
            "rayleigh_values_old_span": torch.linalg.eigvalsh(gram.double()).flip(0).tolist(),
            "fresh_eigenvalues": report["eigenvalues"], "fresh_relative_residuals": report["relative_residuals"],
            "fresh_converged_at_0.01": report["converged_at_0.01"], "fresh_max_relative_residual": report["max_relative_residual"],
            "hutchinson_trace": report.get("hutchinson_trace"),
            "hvp_calls": width + report["iterations"] + min(audit, max(ranks)) + trace_probes,
            "seconds": time.perf_counter() - started}


@torch.no_grad()
def refresh_tracked_basis(hvp, pre_basis, apply_map, *, shift=0.0, audit=16, map_precision=None):
    """One tracking step for a subspace that lives in preconditioned coordinates.

    `hvp` is the conjugated operator P^{1/2} H P^{1/2}, whose eigenvectors u are what the
    anchor stored; the projector the branch actually applies is orth(P^{1/2} U).  A refresh
    therefore has to iterate U itself and re-map the result, exactly as the initial load does.
    Feeding the parameter-space projector to the conjugated operator instead would return a
    basis in the wrong coordinates after a single pass, and every later refresh would compound
    it.  Returns (new_pre_basis, new_parameter_basis, report); with `apply_map` None the two
    bases are the same object and this reduces to `subspace_refresh`.
    """
    new_pre, report = subspace_refresh(hvp, pre_basis, shift=shift, audit=audit)
    if apply_map is None:
        return new_pre, new_pre, report
    mapped, mapping = map_and_orthonormalize(new_pre, apply_map, matmul_precision=map_precision)
    return new_pre, mapped, dict(report, parameter_space_mapping=mapping)


def map_and_orthonormalize(basis, apply_map, rows=1 << 20, block=4, matmul_precision=None):
    """Push a sharded basis through a linear map and re-orthonormalize the image.

    A preconditioned eigenvector u solves P^{1/2} H P^{1/2} u = lam u, so the subspace it
    describes in PARAMETER space is spanned by P^{1/2} u, not by u.  Slowing the right thing
    therefore means projecting onto orth(P^{1/2} U), which is what this returns.  The map is
    applied column by column, then two Cholesky-QR passes orthonormalize the image in place,
    matching `random_orthonormal_basis`.

    The image is stored in the basis's own dtype (bf16 by default) and every pass computes
    in fp32 one row block at a time: at rank 768 an fp32 image is 33 GB per rank, which
    together with the input shard and the model does not fit in GPU memory.
    """
    device = basis.shard.device
    dtype = basis.shard.dtype
    width = basis.width
    lo, hi, sizes = basis.lo, basis.hi, basis.sizes
    phases, clock = {}, [time.perf_counter()]

    def lap(name):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        now = time.perf_counter()
        phases[name] = phases.get(name, 0.0) + now - clock[0]
        clock[0] = now

    mapped = torch.empty((hi - lo, width), device=device, dtype=dtype)
    apply_block = getattr(apply_map, "block", None)
    # The map is ~0.5 TFLOP per column (four GEMMs per matrix); `matmul_precision` "high"
    # runs those on TF32 tensor cores, the orthonormalization below stays at the default.
    previous_precision = torch.get_float32_matmul_precision()
    if matmul_precision:
        torch.set_float32_matmul_precision(str(matmul_precision))
    if apply_block is not None and block > 1:
        # Gather `block` columns and push them through the map as one batch (the per-matrix
        # GEMMs then run once per block); a block of 4 full fp32 columns is ~1.4 GB at 85M dims.
        for start in range(0, width, block):
            columns = torch.stack([all_gather_uneven(basis.shard[:, j].float(), sizes)
                                   for j in range(start, min(start + block, width))], dim=1)
            lap("gather")
            image = apply_block(columns)
            lap("map")
            mapped[:, start:start + columns.shape[1]] = image[lo:hi].to(dtype)
            del columns, image
            lap("scatter")
    else:
        for j in range(width):
            column = all_gather_uneven(basis.shard[:, j].float(), sizes)
            lap("gather")
            image = apply_map(column)
            lap("map")
            mapped[:, j] = image[lo:hi].to(dtype)
            lap("scatter")
    torch.set_float32_matmul_precision(previous_precision)
    conditioning = []
    for _ in range(2):
        gram = torch.zeros((width, width), device=device)
        for start in range(0, hi - lo, rows):
            block = mapped[start:start + rows].float()
            gram += block.T @ block
        all_reduce_sum(gram)
        lap("gram")
        conditioning.append(float(torch.linalg.cond(gram.double())))
        factor = torch.linalg.cholesky(gram.double()).float()
        inverse = torch.linalg.inv(factor.double()).T.float()  # (L^T)^{-1}, applied as a GEMM
        lap("cond_cholesky")
        for start in range(0, hi - lo, rows):
            block = mapped[start:start + rows].float()
            mapped[start:start + rows] = (block @ inverse).to(dtype)
        lap("solve")
    out = ShardedBasis(mapped, basis.dimension, basis.rank, basis.world, sizes)
    error = out.gram_error()
    lap("gram_error")
    return out, {"width": width, "gram_condition": conditioning, "orthogonality_error": error,
                 "phase_seconds": phases, "matmul_precision": str(matmul_precision or previous_precision)}

