# CIFAR-5M scaling-rule sweep and full retuning

One-pass training of a GroupNorm ResNet-18 on CIFAR-5M with the MuonW--AdamW
hybrid optimizer (Muon on convolution weights; AdamW on the classifier, biases
and GroupNorm parameters; both with decoupled weight decay).
Every run processes 100 data checkpoints of 45,056 fresh examples
(4,505,600 examples, no reuse) and reports the validation loss averaged over
the last 10 checkpoints (`--summary_window 10`) on 4,944 held-out examples.

| file | purpose |
|---|---|
| `train_groupnorm_resnet_cifar5m.py` | CIFAR-5M trainer (data preparation, one training run) |
| `train_groupnorm_resnet.py` | model, Muon implementation and scaling-rule helpers (also a CIFAR-10 trainer) |
| `make_rule_sweep.py` | enumerates the 648 scaling rules x 6 target batches and prints commands |
| `requirements.txt` | Python dependencies |

The Newton–Schulz iteration in `train_groupnorm_resnet.py` is adapted from Keller Jordan's
Muon (https://github.com/KellerJordan/Muon, MIT License).

## Setup

```bash
pip install -r requirements.txt
```

Training requires a CUDA GPU.

## Data

Download the six official CIFAR-5M parts (`part0.npz` ... `part5.npz`,
Nakkiran et al., 2021) into `<DATA_ROOT>/raw/`, then run

```bash
python train_groupnorm_resnet_cifar5m.py --data <DATA_ROOT> --prepare_data_only
```

This extracts the image and label arrays of each part into memory-mappable
`<DATA_ROOT>/prepared/images_partK.npy` and `labels_partK.npy`, and writes
`prepared/manifest.json`. Parts 0--4 are read sequentially as the training
stream; the first `--validation_size` examples of part 5 form the validation set.
`--split_seed` seeds the deterministic training augmentation.

## Scaling-rule grid (Figure 2(b), Appendix E)

A rule scales a batch-256 configuration to batch `B` with ratio `B/256`:
learning rates and weight decays use `fixed` / `sqrt` / `linear` powers of the
ratio (Muon and Adam weight decays are chosen separately), and the Muon momentum
and Adam `beta1`, `beta2` are either `fixed` or `sample` (`beta -> beta**ratio`,
preserving the decay per example). The trainer applies the rule itself.

```bash
python make_rule_sweep.py --config <CONFIG_JSON> --count   # 648 rules x 6 batches = 3888 runs
python make_rule_sweep.py --config <CONFIG_JSON> --data <DATA_ROOT> --out <OUT> > commands.txt
```

`<CONFIG_JSON>` holds the batch-256 configuration and the run protocol:

```json
{
  "base_batch_size": 256,
  "hyperparameters": {"muon_lr": <float>, "adam_lr": <float>, "muon_momentum": <float>,
                      "adam_beta1": <float>, "adam_beta2": <float>,
                      "muon_weight_decay": <float>, "adam_weight_decay": <float>},
  "protocol": {"epochs": 100, "summary_window": 10, "train_size": 45056,
               "validation_size": 4944, "eval_batch_size": 1024, "split_seed": <int>,
               "workers": 8},
  "target_batch_sizes": [64, 128, 512, 1024, 2048, 4096],
  "grid_seed": 42
}
```

Learning rates and weight decays are absolute values at batch 256 (decoupled weight decay,
applied as `lr * wd` per step); `muon_momentum`, `adam_beta1` and `adam_beta2` are the
retention coefficients.

The grid uses seed 42; the four rules shown in Figure 2(b) and the reference were
also trained with seeds 43 and 44. Each run needs one GPU (bf16).

## Full retuning

Full retuning at batches 512--4096 runs coordinate descent over all seven
hyperparameters (see `../coordinate_descent/`), with every candidate trained by
the same script with explicit values and `fixed` rules, e.g.

```bash
python train_groupnorm_resnet_cifar5m.py --data <DATA_ROOT> --result <OUT>/run.json \
  --epochs 100 --summary_window 10 --batch_size 512 --base_batch_size 256 \
  --train_size 45056 --validation_size 4944 --eval_batch_size 1024 \
  --split_seed <SPLIT_SEED> --seed 42 --workers 8 \
  --muon_lr <MUON_LR> --muon_lr_rule fixed \
  --adam_lr <ADAM_LR> --adam_lr_rule fixed \
  --muon_momentum <MUON_MOMENTUM> --muon_momentum_rule fixed \
  --adam_beta1 <ADAM_BETA1> --adam_beta1_rule fixed \
  --adam_beta2 <ADAM_BETA2> --adam_beta2_rule fixed \
  --muon_weight_decay <MUON_WD> --adam_weight_decay <ADAM_WD> \
  --weight_decay_rule fixed --muon_weight_decay_rule fixed --adam_weight_decay_rule fixed
```

The result JSON contains the realized hyperparameters, the per-checkpoint
history and `validation_loss_mean` (the summary loss used in the paper).
Weights & Biases logging is off by default (`--wandb_mode disabled`).
