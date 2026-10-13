"""Seekable base-batch stream: branches never replay/skip their parent's prefix."""

from __future__ import annotations

import bisect
from pathlib import Path

import torch

from bsz_lm.branchout.config import Config
from bsz_lm.data import load_data_shard
from bsz_lm.data_manifest import inspect_shard


class Batches:
    def __init__(self, config: Config, device: torch.device, rank: int, world: int):
        self.config, self.device, self.rank, self.world = config, device, rank, world
        self.files, self.ends, self.metadata = {}, {}, {}
        self.cache = {}
        for count in (config.base_batch_tokens, config.hessian_tokens, config.validation_tokens,
                      config.curve_eval_tokens):
            if count % (world * config.sequence_length):
                raise ValueError("global token counts must divide world * sequence_length")
        if config.synthetic:
            self.metadata = {"synthetic": True, "seed": config.seed}
            return
        for split in ("train", "val", "curve"):
            if split == "curve" and not config.curve_holdout_shards:
                continue
            paths = sorted(Path(config.data_root).glob(f"fineweb_{split}_*.bin"))
            if split == "train":
                missing = set(config.curve_holdout_shards) - {p.name for p in paths}
                if missing:
                    raise FileNotFoundError(f"missing reserved train shards: {sorted(missing)}")
                paths = [p for p in paths if p.name not in config.curve_holdout_shards]
            elif split == "curve":
                paths = [Path(config.data_root) / name for name in config.curve_holdout_shards]
            if not paths:
                raise FileNotFoundError(f"missing {split} data under {config.data_root}")
            self.files[split] = paths
            self.metadata[split] = [inspect_shard(
                p, expected_magic=20240520, expected_version=1, hash_contents=split == "curve"
            ) for p in paths]
            total, ends = 0, []
            for entry in self.metadata[split]:
                # Same shard boundary as bsz_lm.data: a shard serves (tokens - 2) // batch batches.
                total += (entry["tokens"] - 2) // config.base_batch_tokens
                ends.append(total)
            self.ends[split] = ends
        if config.curve_holdout_shards:
            if {p.resolve() for p in self.files["train"]} & {
                p.resolve() for p in self.files["curve"]
            }:
                raise ValueError("training and curve holdout overlap")
            if self.metadata["curve"][0]["tokens"] <= config.curve_eval_tokens:
                raise ValueError("holdout cannot provide the fixed contiguous evaluation prefix")

    def batch(self, step: int, *, split: str = "train", tokens: int | None = None):
        cfg = self.config
        if split not in {"train", "val", "curve"}:
            raise ValueError("unknown data split")
        count = cfg.base_batch_tokens if tokens is None else tokens
        if step < 0 or count % (self.world * cfg.sequence_length):
            raise ValueError("invalid data position or token count")
        if cfg.synthetic:
            seed = cfg.seed + step + {"train": 0, "val": 10000019, "curve": 20000033}[split]
            generator = torch.Generator(device=self.device).manual_seed(seed)
            stream = torch.randint(cfg.vocabulary_size, (count + 1,), device=self.device,
                                   generator=generator, dtype=torch.int64)
        else:
            if split in {"val", "curve"}:
                if step != 0:
                    raise ValueError("evaluation always starts at its fixed beginning")
                if split not in self.files:
                    raise ValueError("curve holdout not configured")
                path, offset = self.files[split][0], 0
            else:
                if count > cfg.base_batch_tokens:
                    raise ValueError("training/probe reads may not cross a base-batch boundary")
                index = bisect.bisect_right(self.ends[split], step)
                if index == len(self.files[split]):
                    raise EOFError("training data exhausted; refusing to wrap/reuse tokens")
                previous = self.ends[split][index - 1] if index else 0
                path, offset = self.files[split][index], (step - previous) * cfg.base_batch_tokens
            # One cached shard per stream; evaluation does not evict training data.
            if split not in self.cache or self.cache[split][0] != path:
                self.cache[split] = (path, load_data_shard(path))
            stream = self.cache[split][1][offset:offset + count + 1]
            if len(stream) != count + 1:
                raise ValueError("insufficient contiguous tokens for requested batch")
        local = count // self.world
        start = self.rank * local
        part = stream[start:start + local + 1].to(device=self.device, dtype=torch.int64)
        return (part[:-1].to(torch.int32).view(-1, cfg.sequence_length),
                part[1:].view(-1, cfg.sequence_length))
