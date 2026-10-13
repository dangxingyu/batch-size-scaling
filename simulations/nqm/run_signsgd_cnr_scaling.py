"""Controlled CNR experiments for momentum SignSGD learning-rate scaling.

The SignSGD counterpart of the RMSProp study in run_rmsprop_cnr_scaling.py:
same scalar model (h=1), same three-stage tuning (coarse grid, independent
selection sample, held-out validation), same metric (expected terminal excess
loss).  Unlike that study, theta0 is random (see `train`).

  m <- beta*m + (1-beta)*g,   theta <- theta - eta*sign(m)

At a frozen state the EMA is Gaussian, so the movement ansatz has a closed
form: the mean displacement per update is eta*A(B) with
A(B) = erf(mu / (sqrt(2) s_B)) and s_B^2 = (1-beta)/(1+beta) * c/B.
Preserving displacement per processed sample gives eta(B)/eta(1) = B*A(1)/A(B),
which tends to B when signal dominates and to sqrt(B) when noise dominates.

python nqm/run_signsgd_cnr_scaling.py
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import numpy as np

erf = np.vectorize(math.erf)

OUT = Path(__file__).resolve().parents[1] / 'outputs/signsgd_cnr_scaling'
T = 4096
BATCHES = 2 ** np.arange(9)
C = np.array([1000., 100., 1., .01])
BETAS = [0., .9, .99]
MU = 1.
TAG = ''


def settings_tag():
    """Short digest of the noise levels so runs with different CNR sets cache apart."""
    key = ','.join(f'{x:.10g}' for x in C)
    return hashlib.sha1(key.encode()).hexdigest()[:8]


def train(etas, batch, beta, chains, seed):
    """Terminal excess loss for each learning rate and noise level."""
    etas = np.asarray(etas)
    if etas.ndim == 1:
        etas = np.broadcast_to(etas[:, None], (len(etas), len(C)))
    eta = etas[:, None, :]
    rng = np.random.default_rng(seed)
    # One initialization shared across learning-rate candidates and noise levels:
    # a fixed theta0 would let SignSGD's constant step land exactly on the optimum
    # whenever theta0/eta is an integer, which makes the tuned eta multimodal.
    w = np.broadcast_to(rng.standard_normal((chains, 1)),
                        (len(etas), chains, len(C))).copy()
    m = np.zeros_like(w)
    for _ in range(T // batch):
        # Common draws across learning-rate candidates and noise conditions.
        g = w + np.sqrt(C / batch) * rng.standard_normal((chains, 1))
        m = beta * m + (1 - beta) * g
        w -= eta * np.where(m >= 0, 1., -1.)
    loss = .5 * w * w
    assert np.isfinite(loss).all()
    return loss.mean(1)


def train_sweep():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    coarse = np.geomspace(1e-6, 2., 121)
    for beta in BETAS:
        name = f'beta{beta:g}'
        for b in BATCHES:
            cache = OUT / f'train_{name}_{TAG}_B{b}.npz'
            if not cache.exists():
                risk = train(coarse, int(b), beta, 512, 940001 + int(b))
                candidates = []
                for i in range(len(C)):
                    centers = []
                    for k in np.argsort(risk[:, i]):
                        if all(abs(np.log(coarse[k] / x)) > .35 for x in centers):
                            centers.append(coarse[k])
                        if len(centers) == 3:
                            break
                    candidates.append(np.unique(np.concatenate(
                        [x * np.geomspace(1 / 1.35, 1.35, 17) for x in centers]
                        + [coarse[np.argsort(risk[:, i])[:6]]])))
                length = max(map(len, candidates))
                grid = np.stack([np.pad(x, (0, length - len(x)), mode='edge')
                                 for x in candidates], axis=1)
                # Independent sample selects among the refined neighborhoods.
                selection = train(grid, int(b), beta, 2048, 950001 + int(b))
                indices = selection.argmin(0)
                chosen = grid[indices, np.arange(len(C))]
                assert np.all((chosen > coarse[0]) & (chosen < coarse[-1]))
                # The validation sample never selects the reported learning rate.
                validation = np.stack([train(grid, int(b), beta, 1024,
                                             960001 + 1000 * rep + int(b))
                                       for rep in range(8)])
                np.savez_compressed(cache, T=T, batch=b, c=C, beta=beta, grid=grid,
                                    selection=selection, indices=indices,
                                    eta=chosen, validation=validation)
            with np.load(cache) as d:
                assert int(d['T']) == T and int(d['batch']) == b
                assert float(d['beta']) == beta
                np.testing.assert_array_equal(d['c'], C)
                grid, validation = d['grid'], d['validation']
                profile = validation.mean(0)
                idx = d['indices']
                for i, c in enumerate(C):
                    near = grid[profile[:, i] <= 1.05 * profile[:, i].min(), i]
                    loss = validation[:, idx[i], i]
                    records.append(dict(
                        beta=beta, batch=int(b), c=float(c), cnr=float(1. / c),
                        eta=float(d['eta'][i]), loss=float(loss.mean()),
                        loss_se=float(loss.std(ddof=1) / np.sqrt(8)),
                        near5_eta_min=float(near.min()), near5_eta_max=float(near.max()),
                        heldout_loss_ratio_to_grid_min=float(loss.mean() / profile[:, i].min())))
            print(f'Training {name}, B={b}: '
                  f'eta={[round(r["eta"], 6) for r in records[-len(C):]]}', flush=True)
    with (OUT / 'training.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(records)
    return records


def movement_ansatz(beta):
    """eta(B)/eta(1) preserving mean displacement per processed sample."""
    s = np.sqrt((1 - beta) / (1 + beta) * C[None, :] / BATCHES[:, None])
    a = erf(MU / (np.sqrt(2) * s))
    return BATCHES[:, None] * a[0][None, :] / a


def frozen_check(beta, seed=970001, chains=200000):
    """Simulate the stationary EMA at a frozen state to verify the closed form."""
    rng = np.random.default_rng(seed)
    s = np.sqrt(C[None, :] / BATCHES[:, None])
    m = np.zeros((len(BATCHES), len(C), chains))
    burn, sample = 3000, 3000
    total = np.zeros((len(BATCHES), len(C)))
    for step in range(burn + sample):
        g = MU + s[:, :, None] * rng.standard_normal((len(BATCHES), len(C), chains))
        m = beta * m + (1 - beta) * g
        if step >= burn:
            total += np.where(m >= 0, 1., -1.).mean(2)
    return total / sample


def main():
    records = train_sweep()
    ansatz = {b: movement_ansatz(b) for b in BETAS}

    # Verify the closed-form response against a frozen-state simulation.
    checks = {}
    for beta in BETAS:
        if beta == 0.:
            continue
        simulated = frozen_check(beta)
        s = np.sqrt((1 - beta) / (1 + beta) * C[None, :] / BATCHES[:, None])
        predicted = erf(MU / (np.sqrt(2) * s))
        error = float(np.abs(simulated - predicted).max())
        checks[f'beta{beta:g}_max_abs_response_error'] = error
        print(f'frozen check beta={beta}: max |simulated - closed form| = {error:.2e}',
              flush=True)
        assert error < 5e-3, (simulated, predicted)

    protocol = dict(
        T=T, BATCHES=BATCHES.tolist(), h=1, theta0='theta0 ~ N(0,1), shared across candidates and noise levels', c=C.tolist(),
        cnr=(1. / C).tolist(), betas=BETAS,
        update='g=theta+sqrt(c/B)z; m=beta*m+(1-beta)*g; theta-=eta*sign(m)',
        objective='Expected terminal excess loss theta^2/2',
        search='121 broad log-grid candidates; refine 3 separated neighborhoods per noise level',
        initialization_note='A fixed theta0 makes the tuned eta multimodal for SignSGD '
                            '(constant step size hits the optimum exactly when theta0/eta '
                            'is an integer); a random theta0 removes that lattice effect',
        ansatz_reference_point='Frozen theta = 1',
        coarse_paths=512, independent_selection_paths=2048,
        heldout_groups=8, heldout_paths_per_group=1024,
        near_optimal='Candidates within 5% of minimum heldout mean loss; descriptive',
        movement_ansatz='eta(B)/eta(1)=B*A(1)/A(B), A(B)=erf(mu/(sqrt(2)s_B)), '
                        's_B^2=(1-beta)/(1+beta)*c/B; matches mean displacement per example only',
        frozen_check=checks)
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    np.savez_compressed(OUT / 'ansatz.npz', batches=BATCHES, c=C, betas=BETAS,
                        **{f'ansatz_beta{b:g}': ansatz[b] for b in BETAS})
    print('wrote', OUT / 'protocol.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cnr', type=float, nargs='*',
                        help='Curvature-to-noise ratios (default 0.001 0.01 1 100)')
    parser.add_argument('--betas', type=float, nargs='*',
                        help='Momentum retention values (default 0 0.9 0.99)')
    parser.add_argument('--outdir', type=Path)
    args = parser.parse_args()
    if args.cnr:
        C = 1. / np.asarray(sorted(args.cnr), dtype=float)[::-1].copy()
    if args.betas:
        BETAS = list(args.betas)
    if args.outdir:
        OUT = args.outdir
    TAG = settings_tag()
    main()
