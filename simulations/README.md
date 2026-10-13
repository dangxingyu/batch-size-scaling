# Simulations

Noisy-quadratic-model (NQM) and matrix-model experiments. Run the commands below from this
directory. The numpy scripts (`run_*.py`, `tune_*.py`) run on CPU. The torch scripts
(`*_torch.py`) default to `--device cuda` and fall back to CPU when no GPU is available. Every
script writes to `outputs/<experiment>/` by default, except the momentum refinement and
validation stages, which require `--outdir`.

```sh
uv venv .venv && uv pip install -p .venv numpy torch
source .venv/bin/activate
```

## NQM (`nqm/`)

| experiment | command |
|---|---|
| Scaling-rule transfer for SignSGD with momentum (Fig. 4a–c) | `python nqm/run_nqm_oracle_transfer.py --T 16384 --outdir outputs/nqm_oracle_transfer_t16384` |
| Tuned learning rate vs. CNR, SignSGD (Fig. 4d) | `python nqm/run_signsgd_cnr_scaling.py --cnr 0.001 0.03 1 --betas 0 0.9 --outdir outputs/signsgd_cnr_scaling` |
| Tuned learning rate vs. CNR, RMSProp and Adam (Fig. 9–11) | `python nqm/run_rmsprop_cnr_scaling.py` |
| Finite-budget tuning of SGD (appendix) | `python nqm/tune_sgd_finite_budget.py` |
| Finite-budget tuning of Newton, SignSGD, RMSProp (appendix) | `python nqm/tune_nqm_other_optimizers.py` |
| Tuned optimizers with momentum (Fig. 5 and appendix) | pipeline below |

The CNR scripts cache each tuned batch size as an `.npz` file in their output directory and
reuse it when run again.

Momentum pipeline (each stage can be sharded with `--shard-index/--num-shards`; concatenate the
shard CSVs before the next stage):

1. Broad search over learning rate and momentum/moment coefficients at every batch size, with
   `python nqm/sweep_nqm_momentum_torch.py` (Monte Carlo; writes `outputs/nqm_momentum_sweep/`)
   or its deterministic counterpart `python nqm/sweep_nqm_momentum_finite_volume_torch.py`
   (exact second-moment recursions for SGD-M and Newton-M, finite-volume grids for SignSGD-M
   and Adam; writes `outputs/nqm_momentum_fv/`). Both cover all four optimizers by default;
   `--optimizers` selects a subset.
2. `python nqm/refine_nqm_momentum_local_torch.py --input-csv <concatenated best_shard CSV of
   the search> --outdir outputs/nqm_momentum_refined`: dense local search around each incumbent.
3. `python nqm/validate_nqm_momentum_mc_torch.py --input-csv <concatenated refined_shard CSV>
   --outdir outputs/nqm_momentum_validation`: independent high-chain float64 re-evaluation of
   the selected hyperparameters.

The crossover figure (Fig. 5) uses `--noise-scale 0.25` at every stage; give that run its own
`--outdir` at every stage so it does not overwrite the default-noise results.

## Matrix models (`matrix_models/`)

`sweep_matrix_models_torch.py` tunes SGD-M, SignSGD-M, Adam, Muon and EMA-Shampoo on the matrix
linear, logistic and normalized-linear models and writes one JSON per (model, optimizer, batch).
Settings used for the paper (sharded over ten GPUs, `I = 0..9`):

```sh
# baseline noise
python matrix_models/sweep_matrix_models_torch.py --outdir outputs/matrix_models_f64 --T 4096 \
  --batches 2,4,8,16,32,64,128,256,512,1024 --device cuda --broad-configs 192 --local-configs 256 \
  --local-seeds 8 --search-chains 16 --refine-chains 48 --select-chains 192 --select-pool 24 \
  --finalists 6 --validation-chains 2048 --report-chains 8192 --validation-chunk 2048 \
  --shard-index $I --num-shards 10
# high noise: same, but --outdir outputs/matrix_models_high_noise_f64 (the file names are fixed,
# so a shared folder would overwrite the baseline), plus --sigma-epsilon 4
# --logistic-temperature 4 --search-dtype float64, with --broad-configs 256 --local-configs 384
# --local-seeds 10 --search-chains 24 --refine-chains 72 --select-chains 384 --select-pool 32
# --finalists 8 --validation-chains 4096 --report-chains 16384
```
