"""Muon's preconditioner as a linear map, and its exact square root.

Muon's direction is Orth(M), the polar factor of the momentum, and Orth = grad of the nuclear
norm.  Its Jacobian is therefore the Hessian of a convex function: symmetric and positive
semidefinite, which is what makes P^{1/2} H P^{1/2} well posed for Muon at all.

With the thin SVD M = U diag(s) V^T of a tall matrix, the Jacobian acts on E as

    P E = U W V^T + (I - U U^T) E V diag(1/s) V^T,   W_ij = (A_ij - A_ji)/(s_i + s_j),

where A = U^T E V.  Writing S for the antisymmetric part of A, the quadratic form is
sum_ij 2 S_ij^2/(s_i+s_j) + sum_j ||B_:,j||^2 / s_j with B = (I - U U^T) E V, manifestly
nonnegative, and the square root is

    P^{1/2} E = U (S * C) V^T + B diag(s^{-1/2}) V^T,   C_ij = sqrt(2/(s_i + s_j)).

Small singular values are damped as s + damping, since Muon's own iteration does not invert
directions the momentum has not explored.
"""

import torch


class MatrixPolarPreconditioner:
    """P^{1/2} for one matrix, built from the SVD of its momentum."""

    def __init__(self, momentum, damping=1e-3, scale=1.0):
        if momentum.ndim != 2:
            raise ValueError("expected a matrix")
        self.transposed = momentum.shape[0] < momentum.shape[1]
        tall = momentum.T if self.transposed else momentum
        u, s, vh = torch.linalg.svd(tall.float(), full_matrices=False)
        if damping < 0:
            raise ValueError("damping must be non-negative")
        self.u, self.v = u, vh.T
        self.s = s + damping
        self.scale = float(scale) ** 0.5  # P carries the per-matrix scalar, so P^{1/2} its root
        self.pair = (self.s[:, None] + self.s[None, :]).clamp_min(1e-30)

    @classmethod
    def batch(cls, momenta, damping=1e-3, scales=None, method="svd"):
        """Factors for many matrices at once: momenta of one (transposed) shape share one batched
        SVD call, which is what makes the 72-matrix rebuild cheap enough to run every few steps.
        `momenta` maps name -> matrix, `scales` name -> per-matrix scalar (default 1)."""
        groups = {}
        for name, value in momenta.items():
            if value.ndim != 2:
                raise ValueError("expected a matrix")
            transposed = value.shape[0] < value.shape[1]
            tall = value.T if transposed else value
            groups.setdefault((transposed, tuple(tall.shape)), []).append((name, tall))
        if method not in {"svd", "eigh"}:
            raise ValueError("method must be svd or eigh")
        out = {}
        for (transposed, _), members in groups.items():
            stacked = torch.stack([tall.float() for _, tall in members])
            if method == "svd":
                u, sv, vh = torch.linalg.svd(stacked, full_matrices=False)
            else:
                # The thin SVD through the n x n Gram matrix: one batched symmetric
                # eigensolve (fp64) replaces the m x n SVDs, which cuSOLVER runs one at a
                # time.  s = sqrt(eig(M^T M)), V = its eigenvectors, U = M V / s; the columns
                # of U are re-orthonormalized so a small singular value cannot leak.
                gram = stacked.double().transpose(-1, -2) @ stacked.double()
                evals, evecs = torch.linalg.eigh(gram)
                order = torch.flip(torch.arange(evals.shape[-1], device=evals.device), dims=(0,))
                evals, evecs = evals[..., order], evecs[..., :, order]
                sv = evals.clamp_min(0).sqrt()
                u = stacked.double() @ evecs / sv.clamp_min(1e-12).unsqueeze(-2)
                u, r = torch.linalg.qr(u)
                # QR may flip a column's sign; U and V must keep the same sign convention
                signs = torch.diagonal(r, dim1=-2, dim2=-1).sign()
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                u = u * signs.unsqueeze(-2)
                u, sv, vh = u.float(), sv.float(), evecs.transpose(-1, -2).float()
            for index, (name, _) in enumerate(members):
                factor = cls.__new__(cls)
                factor.transposed = transposed
                factor.u, factor.v = u[index], vh[index].T
                if damping < 0:
                    raise ValueError("damping must be non-negative")
                factor.s = sv[index] + damping
                factor.scale = float(1.0 if scales is None else scales[name]) ** 0.5
                factor.pair = (factor.s[:, None] + factor.s[None, :]).clamp_min(1e-30)
                out[name] = factor
        return out

    def half_batch(self, values):
        """P^{1/2} applied to a batch [k, m, n] of matrices of the momentum's shape (same
        arithmetic as `half`, the batch dimension broadcast through the GEMMs)."""
        if values.ndim != 3:
            raise ValueError("expected a batch of matrices")
        tall = values.transpose(-1, -2) if self.transposed else values
        tall = tall.float()
        projected = tall @ self.v                                   # E V
        a = self.u.T @ projected                                    # A = U^T E V
        antisymmetric = 0.5 * (a - a.transpose(-1, -2))
        head = self.u @ (antisymmetric * (2.0 / self.pair).sqrt())
        complement = projected - self.u @ a                         # B = (I - U U^T) E V
        out = (head + complement * self.s.rsqrt()) @ self.v.T
        out = out * self.scale
        return out.transpose(-1, -2) if self.transposed else out

    def half(self, value):
        """P^{1/2} applied to a matrix of the same shape as the momentum."""
        tall = value.T if self.transposed else value
        tall = tall.float()
        projected = tall @ self.v                      # E V
        a = self.u.T @ projected                       # A = U^T E V
        antisymmetric = 0.5 * (a - a.T)
        head = self.u @ (antisymmetric * (2.0 / self.pair).sqrt())
        complement = projected - self.u @ a            # B = (I - U U^T) E V
        out = (head + complement * self.s.rsqrt()) @ self.v.T
        out = out * self.scale
        return out.T if self.transposed else out

    def full(self, value):
        """P itself, from the defining formula rather than by squaring `half`."""
        tall = value.T if self.transposed else value
        tall = tall.float()
        projected = tall @ self.v
        a = self.u.T @ projected
        w = (a - a.T) / self.pair
        complement = projected - self.u @ a
        out = (self.u @ w + complement * self.s.reciprocal()) @ self.v.T
        out = out * (self.scale ** 2)
        return out.T if self.transposed else out
