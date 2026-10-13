#!/usr/bin/env bash
# Fetch Meta's Distributed Shampoo package, which the Shampoo cases import, from Modded-NanoGPT at
# the commit pinned in configs/protocol.yaml (runtime.shampoo_source). This is a sparse partial
# clone; the trainer verifies the package's Git tree before importing it.
#
#   bash scripts/fetch_shampoo.sh [DEST]        # default DEST: third_party/modded-nanogpt
#   export TRACK3_SHAMPOO_REPOSITORY=<DEST>
set -euo pipefail
DEST="${1:-third_party/modded-nanogpt}"
REPO=https://github.com/KellerJordan/modded-nanogpt.git
COMMIT=7e65c226ddb695fb9752798e0c8fd4ee6bf7dced
SUBDIR=records/track_3_optimization/results/20260513_shampoo_1_4_power
git clone --quiet --filter=blob:none --no-checkout "$REPO" "$DEST"
git -C "$DEST" sparse-checkout set "$SUBDIR"
git -C "$DEST" checkout --quiet "$COMMIT"
echo "export TRACK3_SHAMPOO_REPOSITORY=$(cd "$DEST" && pwd)"
