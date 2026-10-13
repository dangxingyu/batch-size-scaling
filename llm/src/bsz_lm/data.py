"""Torch loader that reads the global token stream in Modded-NanoGPT's order."""

from __future__ import annotations

import glob
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor

HEADER_INTS = 256


def load_data_shard(path: str | Path, *, magic: int = 20240520, version: int = 1) -> Tensor:
    shard = Path(path)
    header = torch.from_file(str(shard), shared=False, size=HEADER_INTS, dtype=torch.int32)
    if int(header[0]) != magic:
        raise ValueError(f"magic number mismatch in {shard}")
    if int(header[1]) != version:
        raise ValueError(f"unsupported binary version in {shard}")
    tokens = int(header[2])
    with shard.open("rb", buffering=0) as handle:
        output = torch.empty(tokens, dtype=torch.uint16, pin_memory=torch.cuda.is_available())
        handle.seek(HEADER_INTS * 4)
        read = handle.readinto(output.numpy())
    if read != 2 * tokens:
        raise ValueError(f"token count mismatch in {shard}")
    return output


def distributed_data_generator(
    filename_pattern: str,
    batch_tokens: int,
    *,
    sequence_length: int = 1024,
    device: torch.device | str = "cuda",
) -> Iterator[tuple[Tensor, Tensor]]:
    files = sorted(Path(filename) for filename in glob.glob(filename_pattern))
    if not files:
        raise FileNotFoundError(filename_pattern)
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if batch_tokens % world or batch_tokens % sequence_length:
        raise ValueError("global token batch must divide world size and sequence length")
    local_tokens = batch_tokens // world
    files_iterator = iter(files)
    tokens, position = load_data_shard(next(files_iterator)), 0
    while True:
        if position + batch_tokens + 1 >= len(tokens):
            tokens, position = load_data_shard(next(files_iterator)), 0
        start = position + rank * local_tokens
        buffer = tokens[start : start + local_tokens + 1]
        inputs = buffer[:-1].to(device=device, dtype=torch.int32, non_blocking=True)
        targets = buffer[1:].to(device=device, dtype=torch.int64, non_blocking=True)
        position += batch_tokens
        yield inputs.view(-1, sequence_length), targets.view(-1, sequence_length)
