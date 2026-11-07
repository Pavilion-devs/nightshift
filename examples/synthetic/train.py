#!/usr/bin/env python3
"""A fake trainer with realistic loss curves and real failure modes. Stdlib only; for tests, CI, dry runs.

It behaves like a real run in the ways Nightshift cares about:
  * loss decays with noise, logs nanoGPT-style lines (optionally JSONL / TensorBoard), saves checkpoints;
  * ``--resume PATH`` continues from a checkpoint;
  * instability is caused by the learning rate: past ``--unstable-after`` steps, any lr above
    ``--lr-critical`` makes the loss blow up and then go NaN. Halving the lr really fixes it;
  * ``--oom-batch N``: batch size >= N dies with a CUDA-style out-of-memory error; halving it fixes it;
  * ``--stall-at``, ``--overfit-after``, ``--crash-at`` for the other detectors.

Examples:
  python train.py --lr 3e-3 --unstable-after 600 --steps 2000          # diverges at ~600, NaN shortly after
  python train.py --lr 1.5e-3 --resume out/ckpt_500.json                # recovers
  python train.py --batch-size 64 --oom-batch 64                        # OOM after a few steps
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import sys
import time
from pathlib import Path


def masked_crc(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0x82F63B78 if crc & 1 else crc >> 1
    crc ^= 0xFFFFFFFF
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def tb_scalar(wall: float, step: int, tag: str, value: float) -> bytes:
    t = tag.encode()
    val = b"\x0a" + _varint(len(t)) + t + b"\x15" + struct.pack("<f", value)
    summary = b"\x0a" + _varint(len(val)) + val
    data = (
        b"\x09"
        + struct.pack("<d", wall)
        + b"\x10"
        + _varint(step)
        + b"\x2a"
        + _varint(len(summary))
        + summary
    )
    header = struct.pack("<Q", len(data))
    return header + struct.pack("<I", masked_crc(header)) + data + struct.pack("<I", masked_crc(data))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--out", default="out")
    ap.add_argument("--resume", default="")
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds per step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--unstable-after", type=int, default=-1)
    ap.add_argument("--lr-critical", type=float, default=2e-3)
    ap.add_argument("--oom-batch", type=int, default=0)
    ap.add_argument("--stall-at", type=int, default=-1)
    ap.add_argument("--stall-seconds", type=float, default=3600)
    ap.add_argument("--overfit-after", type=int, default=-1)
    ap.add_argument("--crash-at", type=int, default=-1)
    ap.add_argument("--jsonl", default="", help="also append metrics to this JSONL file")
    ap.add_argument("--tb", default="", help="also write TensorBoard scalars to this dir")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start, state_loss_offset = 0, 0.0
    if args.resume:
        ck = json.loads(Path(args.resume).read_text())
        start = int(ck["step"])
        state_loss_offset = float(ck.get("offset", 0.0))
        print(
            f"resuming from {args.resume} at step {start} (lr={args.lr:g}, batch_size={args.batch_size})",
            flush=True,
        )
    else:
        print(
            f"starting fresh (lr={args.lr:g}, batch_size={args.batch_size}, grad_accum={args.grad_accum})",
            flush=True,
        )

    rng = random.Random(args.seed * 1_000_003 + start)
    jsonl = open(args.jsonl, "a", buffering=1) if args.jsonl else None  # noqa: SIM115
    tb = None
    if args.tb:
        Path(args.tb).mkdir(parents=True, exist_ok=True)
        tb = open(Path(args.tb) / f"events.out.tfevents.{int(time.time())}.synthetic.{os.getpid()}", "ab")  # noqa: SIM115

    blowup = 0.0
    eff_batch = args.batch_size * args.grad_accum
    for step in range(start + 1, args.steps + 1):
        if args.oom_batch and args.batch_size >= args.oom_batch and step >= start + 5:
            print("Traceback (most recent call last):", file=sys.stderr, flush=True)
            print(
                f"torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB "
                f"(batch_size={args.batch_size})",
                file=sys.stderr,
                flush=True,
            )
            return 1
        if step == args.crash_at:
            print("RuntimeError: dataloader worker 3 exited unexpectedly", file=sys.stderr, flush=True)
            return 2
        if step == args.stall_at:
            print(f"step {step}: waiting on data...", flush=True)
            time.sleep(args.stall_seconds)

        base = 1.2 + 3.0 * math.exp(-step / 400.0) + state_loss_offset
        noise_scale = args.noise * (32.0 / max(1, eff_batch)) ** 0.5
        loss = base * (1.0 + rng.gauss(0.0, noise_scale))
        if args.unstable_after >= 0 and step > args.unstable_after and args.lr > args.lr_critical:
            blowup = blowup * 1.15 + 0.05
            loss = base + blowup
        if blowup > 60:
            loss = math.nan
        grad_norm = (0.8 + 0.2 * rng.random()) * (1.0 + blowup)
        if math.isnan(loss):
            grad_norm = math.inf

        if step % args.log_every == 0:
            print(f"step {step}: loss {loss:.4f}, lr {args.lr:.2e}, grad_norm {grad_norm:.3f}", flush=True)
            if jsonl:
                jsonl.write(
                    json.dumps({"step": step, "loss": loss, "lr": args.lr, "grad_norm": grad_norm}) + "\n"
                )
            if tb:
                tb.write(tb_scalar(time.time(), step, "train/loss", loss))
                tb.flush()
        if step % args.eval_every == 0 and math.isfinite(loss):
            val = base * 1.04 + 0.02
            if args.overfit_after >= 0 and step > args.overfit_after:
                val += 0.002 * (step - args.overfit_after)
            print(f"step {step}: train loss {loss:.4f}, val loss {val:.4f}", flush=True)
            if jsonl:
                jsonl.write(json.dumps({"step": step, "val_loss": val}) + "\n")
        if step % args.ckpt_every == 0:
            ck_path = out / f"ckpt_{step}.json"
            ck_path.write_text(json.dumps({"step": step, "loss": loss if math.isfinite(loss) else None}))
            print(f"saved checkpoint {ck_path}", flush=True)
        if args.sleep:
            time.sleep(args.sleep)

    print("training complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
