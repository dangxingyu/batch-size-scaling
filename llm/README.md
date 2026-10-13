# Language-model experiments

Code for the language-model experiments of the paper:

- the optimizer crossover experiments (Section 4, Figures 1 and 3, Appendix F): five optimizers
  (Adam, Lion, Muon, KL-SOAP, Shampoo) under decoupled weight decay (W) and HyperBall (H),
  each tuned at 128K, 512K, 1M and 2M tokens per batch;
- the scaling-rule sweeps and the full retune in the language-model panel of Figure 2 (MuonW
  cases whose hyperparameters are transferred from the 128K configuration);
- directional branching (Section 6, Figures 6 and 17, Appendix G).

The model and data pipeline are adapted from the Modded-NanoGPT optimization benchmark (Track 3):
a 12-layer, width-768 GPT trained on FineWeb10B for a fixed budget of 1,703,936,000 tokens.

No hyperparameters ship with this code. Every run reads them from a recipe CSV that you write
(format below).

## Layout

```
configs/protocol.yaml                    model, data, token budget, schedule profiles
configs/directional_branching.yaml       base-run and branch settings for Section 6
data/cached_fineweb10B.py                FineWeb10B download (Modded-NanoGPT's script)
src/bsz_lm/                              model, optimizers, data loader, recipe loader, trainer, CLI
src/bsz_lm/branchout/                    base-run (parent) trainer with checkpoints, branch runner
scripts/run_directional_collapse.py      directional-branching driver (anchor statistics, held-subspace basis, branches)
scripts/muon_preconditioner.py           Muon polar-Jacobian preconditioner P (closed form from the SVD)
scripts/directional_*.py                 subspace construction (Lanczos, basis mapping), channel engines, noise profile
scripts/make_rule_sweep.py               the 216-rule scaling sweep (hyperparameter table or a runnable plan)
scripts/fetch_shampoo.sh                 fetches the pinned Distributed Shampoo package
examples/directional_branching_manifest.json   the Figure 6 arms for one anchor
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate      # or: uv venv && uv pip install -e .
pip install -e .                                   # installs the bsz_lm package and the bsz-lm command
```

Training requires CUDA, PyTorch 2.8 or newer, and 8 GPUs per run.

**Data.** Tokenized FineWeb10B shards (`fineweb_train_*.bin`, `fineweb_val_*.bin`) from the
Hugging Face dataset `kjj0/fineweb10B-gpt2`, downloaded by Modded-NanoGPT's script:

```bash
pip install huggingface_hub
python data/cached_fineweb10B.py 18     # into data/fineweb10B/: the validation shard + 18 training shards
```

Eighteen training shards (1.8B tokens) cover every crossover and scaling-rule run;
`bsz-lm manifest-data` checks this for each crossover batch size. Directional branching reserves
`fineweb_train_000001.bin` for its loss curves, which leaves 17 x 762 = 12,954 base batches of
131,072 tokens in 18 shards, fewer than the 13,000 steps of its base run: download at least 19
training shards for it (`python data/cached_fineweb10B.py 19`). A plan binds the exact shard set,
so make the data manifest from the directory you train on.

**Shampoo.** Shampoo cases import Meta's `distributed_shampoo` from the upstream Modded-NanoGPT
repository (https://github.com/KellerJordan/modded-nanogpt), commit
`7e65c226ddb695fb9752798e0c8fd4ee6bf7dced`, subdirectory
`records/track_3_optimization/results/20260513_shampoo_1_4_power`. Point
`TRACK3_SHAMPOO_REPOSITORY` at a clean checkout of that commit; the trainer verifies the pinned
tree before importing it. It is not vendored here; `scripts/fetch_shampoo.sh [DEST]` makes such a
checkout (a sparse clone, about 5 MB) and prints the `export` line.

## Recipe CSV

A recipe CSV holds one training configuration per row: absolute hyperparameters only. It has a
header line with exactly these 20 columns (any order) and one or more rows:

```
family,optimizer,label,batch,batch_tokens,train_steps,matrix_algorithm,matrix_lr,matrix_mu,matrix_beta1,matrix_beta2,shampoo_beta,matrix_wd_peak,aux_lr_emb,aux_lr_proj,aux_lr_misc,aux_beta1,aux_beta2,aux_cooldown_frac,aux_wd_peak
```

The model's parameters are split into the hidden matrices (every 2-D weight inside the 12
transformer blocks), trained by the matrix optimizer named in `optimizer`, and the auxiliary
parameters (token embedding, output projection, and all biases and RMSNorm gains), trained by
AdamW (`eps` 1e-10). All learning rates and weight decays are dimensionless per-step
coefficients.

| column | meaning |
|---|---|
| `family` | `W`: decoupled weight decay, `W <- (1 - lr * wd) W - lr * U`. `H`: HyperBall, `W <- r (W - lr * r * U/‖U‖) / ‖W - lr * r * U/‖U‖‖` per matrix, with `r = ‖W‖_F`, which therefore keeps its value at initialization; `H` uses no weight decay anywhere and its own initialization |
| `optimizer` | hidden-matrix optimizer: `muon`, `klsoap`, `shampoo`, `adam` or `lion` |
| `label` | free text naming the row (e.g. `MuonW`) |
| `batch` | free-text batch label (e.g. `128k`); the batch itself is `batch_tokens` |
| `batch_tokens` | global batch: tokens per optimizer step summed over the 8 GPUs, in sequences of 1,024 tokens. One of 131072, 262144, 524288, 1048576, 2097152 (128 to 2,048 sequences) |
| `train_steps` | `ceil(1703936000 / batch_tokens)`: 13000, 6500, 3250, 1625 or 813 |
| `matrix_algorithm` | free text naming the matrix optimizer (e.g. `MuonW`); recorded in the plan, not read by the trainer |
| `matrix_lr` | peak learning rate of the hidden matrices (`lr` above, before warmup and cooldown). For `H`, the step length relative to `‖W‖_F` |
| `matrix_mu` | Muon momentum (Nesterov form) |
| `matrix_beta1` | first-moment coefficient: Adam and KL-SOAP `beta1`, Shampoo `betas[0]`, Lion interpolation coefficient of the update `sign(beta1 m + (1 - beta1) g)` |
| `matrix_beta2` | second-moment coefficient: Adam and KL-SOAP `beta2`, Shampoo `betas[1]` (also its Adam-grafting `beta2`), Lion momentum coefficient `m <- beta2 m + (1 - beta2) g` |
| `shampoo_beta` | KL-SOAP coefficient of the Kronecker-factor and eigenvalue moving averages |
| `matrix_wd_peak` | hidden-matrix weight decay `wd` above; the decay per step is `lr_t * wd`, so it follows the learning-rate schedule. `0` for `H` |
| `aux_lr_emb` | AdamW peak learning rate of the token embedding |
| `aux_lr_proj` | AdamW peak learning rate of the output projection |
| `aux_lr_misc` | AdamW peak learning rate of the biases and RMSNorm gains |
| `aux_beta1`, `aux_beta2` | AdamW `betas` of the auxiliary parameters |
| `aux_cooldown_frac` | fraction of training at the end over which the auxiliary learning rates decay linearly to zero (`0`: no cooldown). The matrix learning rate uses the same fraction for W/muon, W/shampoo, W/adam and W/lion and a full-run linear decay for KL-SOAP-W and every H row (`configs/protocol.yaml`, `schedule_profiles`, which also sets the fixed 250-step warmups of the Adam and H/Lion rows) |
| `aux_wd_peak` | AdamW weight decay of the auxiliary parameters (decay per step `lr_t * aux_wd_peak`). `0` for `H` |

Columns `matrix_mu`, `matrix_beta1`, `matrix_beta2` and `shampoo_beta` are set or left empty
according to the optimizer; an empty cell is an empty string between two commas:

| optimizer | set | empty |
|---|---|---|
| `muon` | `matrix_mu` | `matrix_beta1`, `matrix_beta2`, `shampoo_beta` |
| `klsoap` | `matrix_beta1`, `matrix_beta2`, `shampoo_beta` | `matrix_mu` |
| `shampoo`, `adam`, `lion` | `matrix_beta1`, `matrix_beta2` | `matrix_mu`, `shampoo_beta` |

All other columns are always set. For example, a 128K MuonW row has the form

```
W,muon,MuonW,128k,131072,13000,MuonW,<matrix_lr>,<matrix_mu>,,,,<matrix_wd_peak>,<aux_lr_emb>,<aux_lr_proj>,<aux_lr_misc>,<aux_beta1>,<aux_beta2>,<aux_cooldown_frac>,<aux_wd_peak>
```

`bsz-lm verify-recipes --recipes recipes.csv` checks a file against this format and prints what
it contains.

## Optimizer crossover runs (Section 4, Figures 1 and 3)

```bash
export TRACK3_DATA_ROOT=$PWD/data/fineweb10B
bsz-lm verify-recipes --recipes recipes.csv
bsz-lm manifest-data --root "$TRACK3_DATA_ROOT" --output runs/fineweb10b.manifest.json
bsz-lm plan --recipes recipes.csv --data-manifest runs/fineweb10b.manifest.json \
    --output runs/plan.json [--families W --optimizers muon --batches 131072]
# run every case of the plan, each on one 8-GPU node (cases are independent; run them in any order)
n=$(python -c "import json; print(len(json.load(open('runs/plan.json'))['cases']))")
for i in $(seq 0 $((n - 1))); do
  TRACK3_OUTPUT_ROOT=runs/results torchrun --standalone --nproc_per_node=8 \
      -m bsz_lm.train --plan runs/plan.json --case-index "$i"
done
# only after every case has finished: collect stops at the first case without a result
bsz-lm collect --plan runs/plan.json --results-root runs/results --output runs/results.csv
```

A plan has one case per selected recipe row and fixes every case by digest of the protocol,
recipe, trainer source and data manifest. Each case writes `runs/results/<case id>/result.json`
and refuses to overwrite an existing case directory. The reported loss is the validation loss on
10,485,760 held-out tokens at the end of training; `results.csv` lists, per case, the case id,
family, optimizer, batch tokens, that loss and the run fingerprint.

## Scaling-rule sweep and retuning (Section 3, Figure 2(a))

`scripts/make_rule_sweep.py` generates the 216 transfer rules at each of 256K, 512K, 1M and 2M
tokens (864 cases) from the 128K MuonW row of a recipe CSV (`--recipes`, required; the file must
have exactly one row with `family` W, `optimizer` muon and `batch_tokens` 131072). A rule is a
six-letter code: matrix learning rate, matrix weight decay and the shared auxiliary Adam
learning-rate multiplier are each F (fixed), S (times sqrt of the batch ratio) or L (times the
ratio); Muon momentum and the auxiliary Adam beta1 and beta2 are each F (fixed) or E
(`beta_B = beta_128K ** (13000 / steps_B)`). Auxiliary weight decay and the cooldown fraction stay
fixed; an optional seventh letter scales the auxiliary weight decay (`--aux-wd-rules SL`). Each
rule also has a `pattern_id`, a hash of its six laws; `--csv` writes one row per rule and batch
with the columns `rule`, `pattern_id` and the recipe columns.

```bash
python scripts/make_rule_sweep.py --recipes recipes.csv --csv runs/rules.csv   # table only
python scripts/make_rule_sweep.py --recipes recipes.csv --plan runs/rule_sweep_plan.json \
    --data-manifest runs/fineweb10b.manifest.json [--batches 2097152] [--rules FLFFFE]
TRACK3_OUTPUT_ROOT=runs/results torchrun --standalone --nproc_per_node=8 \
    -m bsz_lm.train --plan runs/rule_sweep_plan.json --case-index 0     # likewise for every case
```

Full retuning uses the coordinate-descent search in `../coordinate_descent/`. Each candidate it
requests is the all-fixed rule at the target batch with the moved hyperparameters set by
`--set`, written as a one-case plan and run by the same trainer:

```bash
python scripts/make_rule_sweep.py --recipes recipes.csv --plan runs/candidate.json \
    --data-manifest runs/fineweb10b.manifest.json --batches 262144 --rules FFFFFF \
    --set matrix_lr=0.035 --set matrix_wd_peak=0.025
```

## Directional branching (Section 6, Appendix G)

Directional branching runs MuonW only. It needs at least 19 training shards (see Data).

0. **Placeholders.** Replace the placeholders before the first run, with absolute paths:
   in `configs/directional_branching.yaml`, `<DATA_ROOT>` (the directory that contains the
   `fineweb10B/` folder, i.e. `<LLM_DIR>/data` after the download above) and `<RECIPES_CSV>`
   (a recipe CSV with one W,muon row at `batch_tokens` 131072, the hyperparameters of the base
   run); in `examples/directional_branching_manifest.json`, `<LLM_DIR>` (this directory) and
   `<RUNS>` (an output root).
1. **Base run with checkpoints** (the 128K MuonW run, checkpoints every 1,000 steps):
   ```bash
   torchrun --standalone --nproc_per_node=8 -m bsz_lm.branchout parent \
       --algorithm muon --config configs/directional_branching.yaml --output <RUNS>/parent
   ```
2. **Anchors and branches.** Run each task of the manifest in dependency order (`requires`),
   from this directory:
   ```bash
   PYTHONPATH=src:scripts torchrun --standalone --nproc_per_node=8 \
       scripts/run_directional_collapse.py --manifest examples/directional_branching_manifest.json \
       --task <task id> --output <RUNS>/<task id>
   ```
   - `anchor-*`: statistics at the anchor used by every branch;
   - `basis-*`: top eigenvectors of `P^{1/2} G P^{1/2}` (G: Gauss-Newton matrix on a 128K-token
     batch; P: Muon's polar Jacobian at the anchor momentum, singular values damped by 1e-3),
     mapped to parameter space with `P^{1/2}` and orthonormalized; also a random basis;
   - `control-*` (128K everywhere), `fully-scaled-*` (hidden matrices at 2M, i.e. 16x the batch
     with sqrt(16) times the learning rate), `top16-held-*` / `top768-held-*` (the top-r
     subspace stays at 128K, its complement moves to 2M) and `random768-held-*`.

   Each branch runs 1,024 base steps and writes `result.json` (final validation loss on
   10,485,760 tokens) and `curve.jsonl` (loss on 1,048,576 held-out tokens every 32 steps).
   The branch penalty is the final validation loss minus that of the control branch from the
   same anchor.

   The example manifest is for the anchor at step 1,000. For another anchor, copy its tasks and
   change `step`, the checkpoint name (`checkpoint-<step>.pt` with the step zero-padded to six digits, e.g. `checkpoint-012000.pt`), the task ids and the `requires`
   lists that name them, and the `anchor` and `wide_basis` paths, which name task ids. A branch
   may not run past step 13,000 of the schedule, so at the anchor 12,000 set `branch_steps` to
   992 (the largest multiple of the period 16 that fits).

## Licence

The model and training code derive from Modded-NanoGPT (MIT licence, Copyright (c) 2024
Keller Jordan); see `LICENSE-modded-nanogpt`.
