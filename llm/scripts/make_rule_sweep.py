"""The language-model scaling-rule sweep: 216 transfer rules x 4 target batches.

Every rule transports a 128K MuonW reference configuration (the row with family W, optimizer
muon and batch_tokens 131072 of the recipe CSV given by --recipes; format in README.md) to a
larger batch B with batch ratio r = B / 128K and steps(B) = ceil(1,703,936,000 / B).  A rule
is a six-letter code, one letter per transferred entry, in this order:

    1. matrix (Muon) learning rate          F fixed / S times sqrt(r) / L times r
    2. matrix weight decay                  F / S / L
    3. auxiliary Adam learning rates        F / S / L  (embedding, output projection and scalar
                                                        rates share one multiplier)
    4. Muon momentum                        F fixed / E beta(B) = beta_128K ** (13000 / steps(B))
    5. auxiliary Adam beta1                 F / E
    6. auxiliary Adam beta2                 F / E

E keeps the decay of the moving average over the whole run unchanged.  The auxiliary weight
decay and the cooldown fraction stay at their 128K values.  An optional seventh letter (F/S/L)
scales the auxiliary weight decay like the positive quantities (the appendix's auxiliary
weight-decay comparison); the main sweep uses F.

Labels.  Each rule has two equivalent labels: the letter code used in the paper (e.g. SFSFFE)
and a `pattern_id`, the first 12 hex digits of the SHA-256 of the canonical JSON of the six laws
({"matrix_learning_rate": "sqrt", ..., "aux_beta2": "ema_by_steps"}).  Both are emitted.

Retuning candidates.  `--set KEY=VALUE` (repeatable) overrides recipe fields after the rule is
applied, so a coordinate-descent candidate is any rule (usually FFFFFF) plus the moved values.

Usage (from the llm/ directory):

    # the transferred hyperparameters as a table (no data needed)
    python scripts/make_rule_sweep.py --recipes recipes.csv --csv runs/rules.csv
    # a plan that `python -m bsz_lm.train --plan ... --case-index i` runs directly
    python scripts/make_rule_sweep.py --recipes recipes.csv --plan runs/rule_sweep_plan.json \
        --data-manifest runs/fineweb10b.manifest.json [--batches 2097152] [--rules FLFFFE]
    # one coordinate-descent candidate at 256K tokens
    python scripts/make_rule_sweep.py --recipes recipes.csv --plan runs/candidate.json \
        --data-manifest runs/fineweb10b.manifest.json --batches 262144 --rules FFFFFF \
        --set matrix_lr=0.035 --set matrix_wd_peak=0.025
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

LLM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLM / "src"))

from bsz_lm.canonical import digest_file, digest_tree, digest_value  # noqa: E402
from bsz_lm.data_manifest import load_data_manifest  # noqa: E402
from bsz_lm.protocol import load_protocol  # noqa: E402
from bsz_lm.recipes import load_recipes  # noqa: E402

REFERENCE_BATCH = 131072
REFERENCE_STEPS = 13000
TOTAL_TOKENS = 1_703_936_000
TARGET_BATCHES = (262144, 524288, 1048576, 2097152)
EXPONENT = {"F": 0.0, "S": 0.5, "L": 1.0}
BATCH_LABEL = {131072: "128k", 262144: "256k", 524288: "512k", 1048576: "1m", 2097152: "2m"}
LAW_AXES = ("matrix_learning_rate", "matrix_weight_decay", "aux_learning_rate",
            "matrix_momentum", "aux_beta1", "aux_beta2")
POSITIVE_LAW = {"F": "fixed", "S": "sqrt", "L": "linear"}
EMA_LAW = {"F": "fixed", "E": "ema_by_steps"}


def all_rules(aux_wd_letters: str = "") -> list[str]:
    """The 216 six-letter codes (with each optional seventh letter appended)."""
    codes = ["".join(p) + "".join(e) for p in itertools.product("FSL", repeat=3)
             for e in itertools.product("FE", repeat=3)]
    return [c + a for c in codes for a in aux_wd_letters] if aux_wd_letters else codes


def laws(code: str) -> dict[str, str]:
    code = code.upper()
    return {**{a: POSITIVE_LAW[c] for a, c in zip(LAW_AXES[:3], code[:3])},
            **{a: EMA_LAW[c] for a, c in zip(LAW_AXES[3:], code[3:6])}}


def pattern_id(code: str) -> str:
    text = json.dumps(laws(code), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def reference_recipe(recipes_path: str | Path):
    """The single W/muon row at the reference batch of a recipe CSV."""
    matches = [r for r in load_recipes(recipes_path)
               if r.family == "W" and r.optimizer == "muon" and r.batch_tokens == REFERENCE_BATCH]
    if len(matches) != 1:
        raise ValueError(f"{recipes_path} must have exactly one W/muon row at batch_tokens "
                         f"{REFERENCE_BATCH}, found {len(matches)}")
    return matches[0]


def transfer(base, code: str, batch: int):
    """The reference recipe transported to `batch` by rule `code`."""
    code = code.upper()
    if (len(code) not in (6, 7) or any(c not in "FSL" for c in code[:3] + code[6:])
            or any(c not in "FE" for c in code[3:6])):
        raise ValueError(f"bad rule code {code!r}")
    r = batch / REFERENCE_BATCH
    steps = math.ceil(TOTAL_TOKENS / batch)
    p_lr, p_wd, p_aux, e_mu, e_b1, e_b2 = code[:6]
    p_awd = code[6] if len(code) == 7 else "F"

    def ema(beta, flag):
        return beta ** (REFERENCE_STEPS / steps) if flag == "E" else beta

    return dataclasses.replace(
        base, label=f"MuonW-transfer-{code}", batch=BATCH_LABEL[batch], batch_tokens=batch,
        train_steps=steps,
        matrix_lr=base.matrix_lr * r ** EXPONENT[p_lr],
        matrix_wd_peak=base.matrix_wd_peak * r ** EXPONENT[p_wd],
        aux_lr_emb=base.aux_lr_emb * r ** EXPONENT[p_aux],
        aux_lr_proj=base.aux_lr_proj * r ** EXPONENT[p_aux],
        aux_lr_misc=base.aux_lr_misc * r ** EXPONENT[p_aux],
        matrix_mu=ema(base.matrix_mu, e_mu),
        aux_beta1=ema(base.aux_beta1, e_b1),
        aux_beta2=ema(base.aux_beta2, e_b2),
        aux_wd_peak=base.aux_wd_peak * r ** EXPONENT[p_awd],
    )


def apply_overrides(recipe, assignments):
    """Replace recipe fields from KEY=VALUE strings, keeping each field's numeric type."""
    if not assignments:
        return recipe
    fields = dataclasses.asdict(recipe)
    changes = {}
    for item in assignments:
        name, sep, text = item.partition("=")
        if not sep or name not in fields:
            raise ValueError(f"bad override {item!r}")
        current = fields[name]
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            raise ValueError(f"{name} is not a numeric hyperparameter")
        changes[name] = int(text) if isinstance(current, int) else float(text)
    return dataclasses.replace(recipe, label=f"{recipe.label}-candidate", **changes)


def make_case(protocol, recipe, *, stage, seed, data_id, source_id, experiment):
    """Same identity construction as bsz_lm.planning, with an explicit seed."""
    scientific = {
        "schema": "bsz_lm_case_v1", "stage": stage, "protocol_id": protocol.protocol_id,
        "data_id": data_id, "source_id": source_id, "recipe": recipe.scientific_dict(),
        "effective_train_steps": recipe.train_steps, "seed": int(seed), "experiment": experiment,
    }
    fingerprint = digest_value(scientific)
    return {
        **scientific,
        "case_id": (f"{stage}-{recipe.family.lower()}-{recipe.optimizer}-b{recipe.batch_tokens}-"
                    f"{fingerprint[:12]}"),
        "run_fingerprint": fingerprint,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipes", required=True,
                        help="recipe CSV holding the 128K MuonW reference row (format in README.md)")
    parser.add_argument("--csv", help="write the transferred hyperparameters here, one row per rule")
    parser.add_argument("--plan", help="write a bsz_lm plan here (needs --data-manifest)")
    parser.add_argument("--data-manifest")
    parser.add_argument("--protocol", default=str(LLM / "configs/protocol.yaml"))
    parser.add_argument("--rules", default="", help="comma-separated codes (default: all 216)")
    parser.add_argument("--aux-wd-rules", default="",
                        help="seventh letters to append to all 216 codes, e.g. SL")
    parser.add_argument("--batches", default=",".join(map(str, TARGET_BATCHES)))
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--stage", default="transfer")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a recipe field after the rule (repeatable)")
    args = parser.parse_args()
    if not (args.csv or args.plan):
        parser.error("give --csv and/or --plan")

    base = reference_recipe(Path(args.recipes))
    codes = [c for c in args.rules.split(",") if c] or all_rules(args.aux_wd_rules)
    batches = [int(b) for b in args.batches.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    pairs = [(code.upper(), batch, apply_overrides(transfer(base, code, batch), args.set))
             for code in codes for batch in batches]
    for _, _, recipe in pairs:
        recipe.validate()

    if args.csv:
        fields = list(dataclasses.asdict(base).keys())
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["rule", "pattern_id", *fields])
            writer.writeheader()
            for code, _, recipe in pairs:
                writer.writerow({"rule": code, "pattern_id": pattern_id(code),
                                 **dataclasses.asdict(recipe)})
    if args.plan:
        if not args.data_manifest:
            parser.error("--plan needs --data-manifest")
        protocol = load_protocol(args.protocol)
        manifest = load_data_manifest(args.data_manifest)
        source_id = digest_tree(LLM / "src/bsz_lm",
                                ignored_names={"__pycache__", ".pytest_cache", ".ruff_cache"})
        cases = [make_case(protocol, recipe, stage=args.stage, seed=seed,
                           data_id=str(manifest["data_id"]), source_id=source_id,
                           experiment={"transfer_rule": code, "pattern_id": pattern_id(code),
                                       "reference_batch": REFERENCE_BATCH,
                                       "batch_ratio": batch / REFERENCE_BATCH,
                                       "aux_wd_rule": (code + "F")[6],
                                       **({"overrides": sorted(args.set)} if args.set else {})})
                 for code, batch, recipe in pairs for seed in seeds]
        if len({c["case_id"] for c in cases}) != len(cases):
            raise SystemExit("duplicate case identifiers")
        body = {"schema": "bsz_lm_plan_v1", "protocol": protocol.value,
                "protocol_id": protocol.protocol_id, "protocol_sha256": digest_file(args.protocol),
                "recipes_sha256": digest_file(args.recipes),
                "data": {k: v for k, v in manifest.items() if k != "root"}, "source_id": source_id,
                "selection": {"rules": codes, "batches": batches, "seeds": seeds},
                "stage": args.stage, "cases": cases}
        Path(args.plan).parent.mkdir(parents=True, exist_ok=True)
        Path(args.plan).write_text(json.dumps({**body, "plan_id": digest_value(body)},
                                              indent=1, sort_keys=True) + "\n")
    print(json.dumps({"rules": len(codes), "batches": batches, "rows": len(pairs),
                      "csv": args.csv, "plan": args.plan}))


if __name__ == "__main__":
    main()
