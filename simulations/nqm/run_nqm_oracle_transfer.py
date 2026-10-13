"""Oracle-normalized batch-size transfer for momentum SignSGD on an NQM.

For every batch size B in a dyadic sweep, independently tune the shared
(eta, 1-beta) recipe to minimize expected total terminal loss at that batch.
Then, from every source batch B0, transport the B0-optimal recipe to every
target batch B under the SDE and convex rules and measure regret against the
B-optimal recipe:

    regret_i(B0 -> B; rule) = R_i(transported) / R_i(oracle at B) - 1.

This separates rule error from the intrinsic difficulty of the target batch,
which the ratio R_i(B)/R_i(B0) would conflate.

Every search stage is cached under <outdir>/search/ and reused when the script runs again.
"""
from pathlib import Path
import argparse
import csv
import json
import time
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "outputs/nqm_oracle_transfer"
T = 4096
H = np.array([1., .1, .01])
C = np.array([.01, .01, .01])
CHAINS = dict(coarse=192, refined=768, selection=4096, evaluate=8192)
GROUPS = 8


def evaluate(configs, batch, chains, seed, *, return_iterates=False):
    """Each row is eta, alpha=1-beta. Common random draws across configurations."""
    assert 1 <= batch <= T and T % batch == 0
    configs = np.asarray(configs, dtype=float)
    eta = configs[:, 0, None, None]
    alpha = configs[:, 1, None, None]
    rng = np.random.default_rng(seed)
    initial = rng.standard_normal((chains, 3))
    w = np.broadcast_to(initial, (len(configs), chains, 3)).copy()
    m = np.zeros_like(w)
    sigma = np.sqrt(C / batch)
    for _ in range(T // batch):
        gradient = w * H + sigma * rng.standard_normal((chains, 3))
        m *= 1 - alpha
        m += alpha * gradient
        w -= eta * np.where(m >= 0, 1., -1.)
    loss = .5 * H * w * w
    assert np.isfinite(loss).all()
    return w if return_iterates else loss


def mesh(etas, alphas):
    e, a = np.meshgrid(etas, alphas, indexing='ij')
    return np.stack([e.ravel(), a.ravel()], axis=-1)


def search_stage(name, configs, batch, chains, seed):
    cache = OUT / 'search' / f'b{batch}_{name}.npz'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        d = dict(np.load(cache))
        if int(d['T']) != T:
            raise ValueError(f'{cache}: cached budget differs from T={T}')
        return d['configs'], d['risk']
    configs = np.unique(np.asarray(configs), axis=0)
    results = []
    start = time.monotonic()
    for k in range(0, len(configs), 96):
        results.append(evaluate(configs[k:k + 96], batch, chains, seed).mean(1))
    risk = np.concatenate(results)
    np.savez_compressed(cache, T=T, batch=batch, configs=configs, risk=risk,
                        chains=chains, seed=seed)
    idx = int(np.argmin(risk.sum(-1)))
    print(f'  B={batch} {name}: {len(configs)} configs, {time.monotonic()-start:.1f}s, '
          f'best eta/alpha={configs[idx]}, total={risk[idx].sum():.8g}', flush=True)
    return configs, risk


def tune(batch, seed0):
    """Independently tune the shared recipe at one batch size."""
    coarse = mesh(np.geomspace(1e-5, 2., 33), np.geomspace(1e-5, 1., 17))
    coarse = np.vstack([coarse, [.01, .01]])
    cfg, risk = search_stage('coarse', coarse, batch, CHAINS['coarse'], seed0)
    centers = []
    for j in np.argsort(risk.sum(-1)):
        if all(np.linalg.norm(np.log(cfg[j] / c)) > .55 for c in centers):
            centers.append(cfg[j])
        if len(centers) == 6:
            break
    candidates = [cfg[np.argsort(risk.sum(-1))[:24]]]
    for e, a in centers:
        candidates.append(mesh(e * np.geomspace(.55, 1.8, 7),
                               np.clip(a * np.geomspace(.45, 2.2, 7), 1e-7, 1.)))
    fine, frisk = search_stage('refined', np.vstack(candidates), batch,
                               CHAINS['refined'], seed0 + 1)
    shortlist = fine[np.argsort(frisk.sum(-1))[:24]]
    shortlist = np.vstack([shortlist, cfg[np.argsort(risk.sum(-1))[:6]], [.01, .01]])
    valid, vrisk = search_stage('selection', shortlist, batch,
                                CHAINS['selection'], seed0 + 2)
    return valid[int(np.argmin(vrisk.sum(-1)))]


def transport(recipe, source, target):
    """SDE and convex transport of one recipe from `source` to `target` batch."""
    eta, alpha = float(recipe[0]), float(recipe[1])
    kappa = target / source
    # beta = (1-alpha)^kappa; alpha == 1 means beta == 0, which every rule fixes.
    sde_alpha = 1. if alpha >= 1. else -np.expm1(kappa * np.log1p(-alpha))
    return np.array([[eta * kappa, sde_alpha],
                     [eta * np.sqrt(kappa), alpha]])


def main():
    batches = 2 ** np.arange(int(np.log2(T)) + 1)
    n = len(batches)

    print('Oracle tuning at each batch size', flush=True)
    oracle = np.empty((n, 2))
    for j, b in enumerate(batches):
        oracle[j] = tune(int(b), 20260921 + 100 * j)

    # Row = source batch, column = target batch, plus the oracle row for reference.
    print('\nEvaluating the transfer matrix', flush=True)
    risk = np.empty((2, n, n, GROUPS, 3))      # rule, source, target, group, direction
    oracle_risk = np.empty((n, GROUPS, 3))     # target, group, direction
    configs = np.empty((2, n, n, 2))
    for jt, bt in enumerate(batches):
        stack = [transport(oracle[js], int(bs), int(bt))
                 for js, bs in enumerate(batches)]
        configs[:, :, jt] = np.stack(stack, axis=1)
        block = np.concatenate([np.concatenate(stack, axis=0), oracle[jt][None]], axis=0)
        for rep in range(GROUPS):
            loss = evaluate(block, int(bt), CHAINS['evaluate'], 20262000 + rep).mean(1)
            risk[:, :, jt, rep] = loss[:-1].reshape(n, 2, 3).transpose(1, 0, 2)
            oracle_risk[jt, rep] = loss[-1]
        print(f'  target B={bt}: oracle total={oracle_risk[jt].mean(0).sum():.6g}', flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'oracle_transfer.npz', T=T, batches=batches, h=H, c=C,
                        oracle=oracle, configs=configs, risk=risk,
                        oracle_risk=oracle_risk)

    # Regret relative to the target-batch optimum.
    reg = risk / oracle_risk[None, None, :, :, :] - 1
    total = risk.sum(-1) / oracle_risk.sum(-1)[None, None] - 1
    with (OUT / 'oracle_transfer.csv').open('w') as f:
        wr = csv.writer(f, lineterminator='\n')
        wr.writerow(['rule', 'source_batch', 'target_batch', 'ratio', 'eta', 'beta',
                     'direction', 'cnr', 'loss', 'oracle_loss', 'regret', 'regret_sd',
                     'total_regret', 'total_regret_sd'])
        for r, rule in enumerate(['sde', 'convex']):
            for js, bs in enumerate(batches):
                for jt, bt in enumerate(batches):
                    for i in range(3):
                        wr.writerow([rule, int(bs), int(bt), int(bt) / int(bs),
                                     configs[r, js, jt, 0], 1 - configs[r, js, jt, 1],
                                     i + 1, H[i] ** 2 / C[i],
                                     risk[r, js, jt, :, i].mean(),
                                     oracle_risk[jt, :, i].mean(),
                                     reg[r, js, jt, :, i].mean(),
                                     reg[r, js, jt, :, i].std(ddof=1),
                                     total[r, js, jt].mean(),
                                     total[r, js, jt].std(ddof=1)])

    protocol = dict(
        T=T, batches=batches.tolist(), h=H.tolist(), c=C.tolist(),
        cnr=(H ** 2 / C).tolist(),
        initialization='theta0 ~ N(0,I3), m0=0', weight_decay=0,
        optimizer='m=beta*m+(1-beta)*g; theta=theta-eta*sign(m)',
        oracle='Independent (eta,beta) search at every batch size, minimizing '
               'expected total terminal loss over the three directions',
        oracle_recipes={int(b): dict(eta=float(oracle[j, 0]), beta=float(1 - oracle[j, 1]))
                        for j, b in enumerate(batches)},
        scaling=dict(sde='eta*=kappa; beta=beta^kappa',
                     convex='eta*=sqrt(kappa); beta unchanged'),
        metric='regret_i = R_i(transported from B0)/R_i(oracle at B) - 1',
        evaluation=f'{GROUPS} independent groups of {CHAINS["evaluate"]} paths, '
                   'separate from search and selection',
        qualification='Best found in numerical search, not a certified global optimum')
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')

    # Diagonal check: transporting a recipe to its own batch is the identity.
    d = np.arange(n)
    np.testing.assert_allclose(configs[0, d, d], oracle, rtol=1e-12)
    np.testing.assert_allclose(configs[1, d, d], oracle, rtol=1e-12)
    assert np.abs(total[:, d, d]).max() < 1e-9, total[:, d, d]
    print('\nDiagonal identity check passed (zero regret when B0 == B).', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--T', type=int, default=4096, help='Sample budget, a power of two')
    parser.add_argument('--outdir', type=Path)
    args = parser.parse_args()
    T = args.T
    if T < 1 or T & (T - 1):
        parser.error('--T must be a positive power of two')
    OUT = args.outdir if args.outdir is not None else (
        OUT if T == 4096 else OUT.parent / f'nqm_oracle_transfer_t{T}')
    main()
