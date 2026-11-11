#!/usr/bin/env bash
# Propose the two demo runs from plan.md §9. Nothing starts until you approve each one
# (`/nightshift approve <id>` in chat, or the Approve button in Hermes Desktop; there is no CLI approve).
#
#   PYTHON=/path/to/python-with-torch scripts/demo.sh [workdir]
#
# Run 1 "tinygpt": lr 3e-3 is too high once training is past step 600, so the loss blows up; Nightshift rolls
#                  back to the last healthy checkpoint with lr 1.5e-3 and the run finishes.
# Run 2 "tinygpt-big-batch": batch 128 runs out of memory (simulated); Nightshift restarts it with batch 64 and
#                  grad_accum 2.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-$HOME/nightshift-demo}"
PY="${PYTHON:-python3}"
HERMES="${HERMES:-hermes}"

"$PY" -c "import torch" 2>/dev/null || { echo "PYTHON must have torch installed (got: $PY)"; exit 1; }
# Safe to re-run before every take: old checkpoints are removed so a rollback can only pick this take's.
# The dataset is fetched once into $WORK/data and reused, so a recording never waits on the network.
rm -rf "$WORK/run1/out" "$WORK/run2/out"
mkdir -p "$WORK/data" "$WORK/run1/data" "$WORK/run2/data"
if [ ! -s "$WORK/data/tinyshakespeare.txt" ]; then
  (cd "$WORK" && "$PY" -c "import sys; sys.path.insert(0, '$ROOT/examples/tinygpt'); \
from pathlib import Path; from train import load_text; load_text(Path('data'))")
fi
for run in run1 run2; do
  cp "$ROOT/examples/tinygpt/train.py" "$WORK/$run/train.py"
  cp "$WORK/data/tinyshakespeare.txt" "$WORK/$run/data/tinyshakespeare.txt"
done

"$HERMES" nightshift launch --name tinygpt --cwd "$WORK/run1" \
  --param lr=0.003 --param batch_size=32 --param grad_accum=1 \
  -- "$PY" train.py --lr {lr} --batch-size {batch_size} --grad-accum {grad_accum} \
     --steps 1500 --unstable-after 600 --lr-critical 0.002 --resume {resume}

"$HERMES" nightshift launch --name tinygpt-big-batch --cwd "$WORK/run2" \
  --param lr=0.001 --param batch_size=128 --param grad_accum=1 \
  -- "$PY" train.py --lr {lr} --batch-size {batch_size} --grad-accum {grad_accum} \
     --steps 800 --oom-batch 128 --resume {resume}
