"""Enumerate the CIFAR-5M scaling-rule grid and print one training command per run.

A rule chooses, independently, fixed / sqrt / linear scaling (in the batch ratio
B / 256) for the Muon learning rate, the Adam learning rate, the Muon weight decay
and the Adam weight decay, and fixed / sample-preserving scaling (beta -> beta**ratio)
for the Muon momentum and the two Adam moment coefficients: 3^4 * 2^3 = 648 rules.
The trainer applies the rule to the batch-256 configuration itself, so each command
passes the configuration's values plus the rule names. The configuration file is a JSON
object; its format is described in README.md.

    python make_rule_sweep.py --config <CONFIG_JSON> --data <DATA_ROOT> --out <OUT> > commands.txt
    python make_rule_sweep.py --config <CONFIG_JSON> --count   # prints 648 rules x 6 batches
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LR_RULES = ("fixed", "sqrt", "linear")
MOMENTUM_RULES = ("fixed", "sample")
WD_RULES = ("fixed", "sqrt", "linear")
ABBREVIATION = {"fixed": "f", "sqrt": "q", "linear": "l", "sample": "s"}


def all_rules():
    for mlr, alr, mm, b1, b2, mwd, awd in itertools.product(
            LR_RULES, LR_RULES, MOMENTUM_RULES, MOMENTUM_RULES, MOMENTUM_RULES, WD_RULES, WD_RULES):
        yield {"muon_lr_rule": mlr, "adam_lr_rule": alr, "muon_momentum_rule": mm,
               "adam_beta1_rule": b1, "adam_beta2_rule": b2,
               "muon_weight_decay_rule": mwd, "adam_weight_decay_rule": awd}


def label(batch_size: int, rules: dict[str, str]) -> str:
    """Run label (run name and result-file stem), e.g. b512-mlrq-alrq-mmf-b1f-b2f-mwdf-awdf."""
    a = {k: ABBREVIATION[v] for k, v in rules.items()}
    return (f"b{batch_size}-mlr{a['muon_lr_rule']}-alr{a['adam_lr_rule']}-mm{a['muon_momentum_rule']}"
            f"-b1{a['adam_beta1_rule']}-b2{a['adam_beta2_rule']}"
            f"-mwd{a['muon_weight_decay_rule']}-awd{a['adam_weight_decay_rule']}")


def command(config: dict, batch_size: int, rules: dict[str, str], seed: int,
            data: str, out: str, python: str) -> list[str]:
    h, p = config["hyperparameters"], config["protocol"]
    name = label(batch_size, rules)
    cmd = [python, str(HERE / "train_groupnorm_resnet_cifar5m.py"),
           "--data", data, "--result", f"{out}/{name}-s{seed}.json",
           "--batch_size", str(batch_size), "--base_batch_size", str(config["base_batch_size"]),
           "--seed", str(seed), "--run_name", f"{name}-s{seed}"]
    for key in ("epochs", "summary_window", "train_size", "validation_size", "eval_batch_size",
                "split_seed", "workers"):
        cmd += [f"--{key}", str(p[key])]
    for key in ("muon_lr", "adam_lr", "muon_momentum", "adam_beta1", "adam_beta2",
                "muon_weight_decay", "adam_weight_decay"):
        cmd += [f"--{key}", repr(h[key])]
    for key, value in rules.items():
        cmd += [f"--{key}", value]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True,
                        help="JSON with the batch-256 configuration and the run protocol (see README.md)")
    parser.add_argument("--data", default="<DATA_ROOT>")
    parser.add_argument("--out", default="<OUT>")
    parser.add_argument("--python", default="python")
    parser.add_argument("--batch", type=int, action="append",
                        help="restrict to these target batches (default: all six)")
    parser.add_argument("--count", action="store_true", help="only print the number of runs")
    parser.add_argument("--jsonl", action="store_true", help="print JSON lines instead of shell commands")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    batches = args.batch or config["target_batch_sizes"]
    rules = list(all_rules())
    if args.count:
        print(f"{len(rules)} rules x {len(batches)} batches = {len(rules) * len(batches)} runs")
        return
    for b in batches:
        for r in rules:
            cmd = command(config, b, r, config["grid_seed"], args.data, args.out, args.python)
            print(json.dumps({"label": label(b, r), "command": cmd}) if args.jsonl else " ".join(cmd))


if __name__ == "__main__":
    main()
