"""Controlled curvature-to-noise ratio (CNR) experiments for RMSProp and Adam learning-rate scaling.

python nqm/run_rmsprop_cnr_scaling.py [--outdir DIR]
All training curves use actual stochastic RMSProp/Adam, not a mean-Fisher closure. Tuned learning
rates go to training.csv (RMSProp) and adam.csv; the frozen-state movement-ansatz ratios and the
exact SGD optimum go to summary.json. Per-batch tuning results are cached as .npz in the output
directory and reused when the script runs again.
"""
from pathlib import Path
import csv
import json
import numpy as np

OUT = Path(__file__).resolve().parents[1]/'outputs/rmsprop_cnr_scaling'
T = 4096
BATCHES = 2**np.arange(9)
C = np.array([100., 1., .01])
EPS = 1e-8
SETTINGS = [(.9, False), (.9, True), (.99, True)]
ADAM_SETTINGS = [(.9, .9, True), (.9, .99, True)]


def train(etas, batch, beta, correction, chains, seed, beta1=0.):
    """beta1=0 is RMSProp (no first moment: the raw gradient is the numerator)."""
    etas = np.asarray(etas)
    if etas.ndim == 1:
        etas = np.broadcast_to(etas[:, None], (len(etas), len(C)))
    eta = etas[:, None, :]
    rng = np.random.default_rng(seed)
    w = np.ones((len(etas), chains, len(C)))
    v = np.zeros_like(w)
    m = np.zeros_like(w)
    for step in range(1, T//batch+1):
        # Common draws across LR candidates and noise conditions.
        g = w + np.sqrt(C/batch)*rng.standard_normal((chains, 1))
        v = beta*v+(1-beta)*g*g
        denominator = v/(1-beta**step) if correction else v
        if beta1 == 0.:
            numerator = g
        else:
            m = beta1*m+(1-beta1)*g
            numerator = m/(1-beta1**step) if correction else m
        w -= eta*numerator/(np.sqrt(denominator)+EPS)
    loss = .5*w*w
    assert np.isfinite(loss).all()
    return loss.mean(1)


def train_sweep():
    records = []
    coarse = np.geomspace(1e-5, 1., 101)
    for beta, correction in SETTINGS:
        name = f'beta{beta:g}_bc{int(correction)}'
        for b in BATCHES:
            cache = OUT/f'train_{name}_B{b}.npz'
            if not cache.exists():
                risk = train(coarse, int(b), beta, correction, 512, 840001+int(b))
                candidates = []
                for i in range(len(C)):
                    centers = []
                    for k in np.argsort(risk[:, i]):
                        if all(abs(np.log(coarse[k]/x))>.35 for x in centers):
                            centers.append(coarse[k])
                        if len(centers)==3:
                            break
                    candidates.append(np.unique(np.concatenate([
                        x*np.geomspace(1/1.35, 1.35, 17) for x in centers
                    ] + [coarse[np.argsort(risk[:, i])[:6]]])))
                length = max(map(len, candidates))
                grid = np.stack([np.pad(x, (0, length-len(x)), mode='edge') for x in candidates], axis=1)
                # Independent sample selects among multiple refined candidate neighborhoods.
                selection = train(grid, int(b), beta, correction, 2048, 850001+int(b))
                indices = selection.argmin(0)
                chosen = grid[indices, np.arange(len(C))]
                assert np.all((chosen>coarse[0]) & (chosen<coarse[-1]))
                # The validation sample never selects the reported LR.
                validation = np.stack([train(grid, int(b), beta, correction, 1024,
                                     860001+1000*rep+int(b)) for rep in range(8)])
                np.savez_compressed(cache, T=T, batch=b, c=C, beta=beta,
                                    bias_correction=correction, grid=grid, selection=selection,
                                    indices=indices, eta=chosen, validation=validation)
            with np.load(cache) as d:
                assert int(d['T'])==T
                assert int(d['batch'])==b and float(d['beta'])==beta
                assert bool(d['bias_correction'])==correction
                np.testing.assert_array_equal(d['c'], C)
                grid, validation = d['grid'], d['validation']
                profile = validation.mean(0)
                idx = d['indices']
                for i, c in enumerate(C):
                    near = grid[profile[:, i] <= 1.05*profile[:, i].min(), i]
                    loss = validation[:, idx[i], i]
                    records.append(dict(beta=beta, bias_correction=correction, batch=int(b),
                                        c=float(c), initial_snr=float(b/c), eta=float(d['eta'][i]),
                                        loss=float(loss.mean()), loss_se=float(loss.std(ddof=1)/np.sqrt(8)),
                                        terminal_snr=float(2*loss.mean()*b/c),
                                        near5_eta_min=float(near.min()), near5_eta_max=float(near.max()),
                                        heldout_loss_ratio_to_grid_min=float(loss.mean()/profile[:, i].min())))
            print(f'Training {name}, B={b}: eta={[r["eta"] for r in records[-3:]]}', flush=True)
    with (OUT/'training.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(records)
    return records


def adam_sweep():
    """Same tuning protocol as train_sweep, with Adam's first moment enabled."""
    records = []
    coarse = np.geomspace(1e-5, 1., 101)
    for beta1, beta, correction in ADAM_SETTINGS:
        name = f'adam_b1{beta1:g}_b2{beta:g}_bc{int(correction)}'
        for b in BATCHES:
            cache = OUT/f'train_{name}_B{b}.npz'
            if not cache.exists():
                risk = train(coarse, int(b), beta, correction, 512, 940001+int(b), beta1=beta1)
                candidates = []
                for i in range(len(C)):
                    centers = []
                    for k in np.argsort(risk[:, i]):
                        if all(abs(np.log(coarse[k]/x))>.35 for x in centers):
                            centers.append(coarse[k])
                        if len(centers)==3:
                            break
                    candidates.append(np.unique(np.concatenate([
                        x*np.geomspace(1/1.35, 1.35, 17) for x in centers
                    ] + [coarse[np.argsort(risk[:, i])[:6]]])))
                length = max(map(len, candidates))
                grid = np.stack([np.pad(x, (0, length-len(x)), mode='edge') for x in candidates], axis=1)
                selection = train(grid, int(b), beta, correction, 2048, 950001+int(b), beta1=beta1)
                indices = selection.argmin(0)
                chosen = grid[indices, np.arange(len(C))]
                assert np.all((chosen>coarse[0]) & (chosen<coarse[-1]))
                validation = np.stack([train(grid, int(b), beta, correction, 1024,
                                     960001+1000*rep+int(b), beta1=beta1) for rep in range(8)])
                np.savez_compressed(cache, T=T, batch=b, c=C, beta1=beta1, beta=beta,
                                    bias_correction=correction, grid=grid, selection=selection,
                                    indices=indices, eta=chosen, validation=validation)
            with np.load(cache) as d:
                assert int(d['T'])==T and int(d['batch'])==b
                assert float(d['beta1'])==beta1 and float(d['beta'])==beta
                np.testing.assert_array_equal(d['c'], C)
                for i, c in enumerate(C):
                    records.append(dict(beta1=beta1, beta=beta, bias_correction=correction,
                                        batch=int(b), c=float(c), eta=float(d['eta'][i])))
            print(f'Adam {name}, B={b}: eta={[r["eta"] for r in records[-3:]]}', flush=True)
    return records


def frozen_sweep():
    cache = OUT/'frozen_response.npz'
    if not cache.exists():
        all_responses = []
        for beta in [.9, .99]:
            groups = []
            sigma = np.sqrt(C[None, :, None]/BATCHES[:, None, None])
            for rep in range(8):
                rng = np.random.default_rng(870001+rep)
                vp = np.broadcast_to(1+sigma*sigma, (len(BATCHES), len(C), 512)).copy()
                vm = vp.copy()
                response = np.zeros(vp.shape[:2])
                # >20 memory times even at beta=.99; initial v influence <2e-9.
                for step in range(2048+512):
                    z = rng.standard_normal((1, 1, 512))
                    gp, gm = 1+sigma*z, 1-sigma*z
                    vp = beta*vp+(1-beta)*gp*gp
                    vm = beta*vm+(1-beta)*gm*gm
                    if step>=2048:
                        response += .5*(gp/(np.sqrt(vp)+EPS)+gm/(np.sqrt(vm)+EPS)).mean(2)/512
                groups.append(response)
            all_responses.append(groups)
            print(f'Frozen-state response beta={beta} complete', flush=True)
        np.savez_compressed(cache, responses=np.array(all_responses), betas=[.9,.99],
                            batches=BATCHES, c=C, mu=1., burnin=2048,
                            sampling_steps=512, antithetic_pairs_per_group=512)
    with np.load(cache) as d:
        np.testing.assert_array_equal(d['batches'], BATCHES)
        np.testing.assert_array_equal(d['c'], C)
        response = d['responses'].copy()
    # Match mean displacement per processed sample, not complete dynamics.
    ratios = BATCHES[None, None, :, None]*response[:, :, :1, :]/response
    approximation = BATCHES[:, None]*np.sqrt((1+C[None, :]/BATCHES[:, None])/(1+C[None, :]))
    assert np.isfinite(ratios).all() and (response>0).all()
    np.testing.assert_allclose(ratios[:, :, 0], 1.)
    return ratios, approximation


def sgd_sweep():
    etas = np.geomspace(1e-7, 1.999, 20000)
    q = (1-etas)**2
    rows = []
    for b in BATCHES:
        steps = T//b
        residual = q**steps
        geometric = (1-residual)/(1-q)
        risk = .5*(residual[:, None]+etas[:, None]**2*C[None, :]/b*geometric[:, None])
        idx = risk.argmin(0)
        rows.append(etas[idx])
    return np.array(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=train_sweep()
    ratios, approximation=frozen_sweep()
    sgd=sgd_sweep()
    adam=adam_sweep()
    with (OUT/'adam.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=adam[0].keys())
        writer.writeheader(); writer.writerows(adam)
    (OUT/'summary.json').write_text(json.dumps(dict(sgd_eta=sgd.tolist(),
        frozen_max_relative_approximation_error=float(abs(ratios.mean(1)/approximation[None,:,:]-1).max()),
        frozen_ratios_mean=ratios.mean(1).tolist(),
        frozen_ratios_se=(ratios.std(1,ddof=1)/np.sqrt(8)).tolist()),indent=2)+'\n')
    protocol=dict(T=T,BATCHES=BATCHES.tolist(),h=1,theta0=1,c=C.tolist(),
                  epsilon=EPS,momentum=0,weight_decay=0,v0=0,
                  update='g=theta+sqrt(c/B)z; v=beta2*v+(1-beta2)*g^2; theta-=eta*g/(sqrt(v)+eps)',
                  bias_correction='Where enabled, divide v by 1-beta2^step before sqrt',
                  training_settings=SETTINGS,objective='Expected terminal excess loss theta^2/2',
                  search='101 broad log-grid candidates; refine 3 separated candidate neighborhoods independently per noise level',
                  coarse_paths=512,independent_selection_paths=2048,heldout_groups=8,heldout_paths_per_group=1024,
                  near_optimal='Candidates within 5% of minimum heldout mean loss; descriptive, not confidence intervals',
                  frozen='theta fixed at1; mu1; stationary EMA measured after2048 burnin steps; 8 groups of512 antithetic pairs,512 correlated samples per pair',
                  local_matching='eta(B)/eta(1)=B*A(1)/A(B), A=E[g/(sqrt(v)+eps)]; matches mean drift per example only',
                  approximation='B*sqrt((1+c/B)/(1+c)); mean-denominator approximation, not exact RMSProp identity',
                  cnr='h^2/c=1/c=1/chi; fixed model property, with h=1',
                  qualification='Gradient SNR=B*theta^2*CNR evolves during training; slopes summarize endpoints, not universal powers')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print('DONE',flush=True)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path)
    args = parser.parse_args()
    if args.outdir:
        OUT = args.outdir
    main()
