#!/usr/bin/env python3
"""tinygpt: a character-level GPT on Shakespeare, the trainer used in Nightshift's demo.

A real model and a real training loop (PyTorch, AdamW, CUDA / Apple MPS / CPU) with checkpoints and resume. Two
failure modes can be switched on for the demo. Both are labelled as injected; both act through the real
training dynamics, so recovering from them works the same way it would for a real run:

  --unstable-after K --lr-critical X   after step K, if the configured lr is above X, the optimizer's effective
                                       lr is multiplied by --instability-gain: the loss genuinely blows up.
                                       Halving the lr below X (what Nightshift's rollback does) really fixes it.
  --oom-batch N                        a batch size >= N raises an MPS/CUDA-style out-of-memory error
                                       (simulated: tiny models never run out of memory for real).

  python train.py --lr 3e-3 --unstable-after 600 --lr-critical 2e-3 --resume {resume}
"""

from __future__ import annotations

import argparse
import math
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
FALLBACK = (
    "Shall I compare thee to a summer's day? Thou art more lovely and more temperate: Rough winds do shake "
    "the darling buds of May, And summer's lease hath all too short a date; Sometime too hot the eye of heaven "
    "shines, And often is his gold complexion dimm'd; And every fair from fair sometime declines, By chance or "
    "nature's changing course untrimm'd; But thy eternal summer shall not fade, Nor lose possession of that "
    "fair thou ow'st; Nor shall death brag thou wander'st in his shade, When in eternal lines to time thou "
    "grow'st: So long as men can breathe or eyes can see, So long lives this, and this gives life to thee.\n"
)


def load_text(data_dir: Path) -> str:
    path = data_dir / "tinyshakespeare.txt"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(DATA_URL, timeout=20) as r:  # noqa: S310 - fixed https URL
                path.write_bytes(r.read())
            print(f"downloaded tiny shakespeare to {path}", flush=True)
        except OSError as exc:
            print(f"could not download the dataset ({exc}); using a built-in sonnet instead", flush=True)
            return FALLBACK * 400
    return path.read_text()


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))
        self.register_buffer("mask", torch.triu(torch.ones(block, block, dtype=torch.bool), diagonal=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=self.mask[:t, :t], need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab: int, block: int, n_layer: int = 4, n_head: int = 4, n_embd: int = 128) -> None:
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(block, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block, 0.0) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        t = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        logits = self.head(self.ln(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--out", default="out")
    ap.add_argument("--data", default="data")
    ap.add_argument("--resume", default="")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--unstable-after", type=int, default=-1)
    ap.add_argument("--lr-critical", type=float, default=2e-3)
    ap.add_argument("--instability-gain", type=float, default=60.0)
    ap.add_argument("--oom-batch", type=int, default=0)
    args = ap.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    text = load_text(Path(args.data))
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    def batch(src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(src) - args.block - 1, (args.batch_size,))
        x = torch.stack([src[i : i + args.block] for i in ix])
        y = torch.stack([src[i + 1 : i + args.block + 1] for i in ix])
        return x.to(device), y.to(device)

    model = TinyGPT(len(chars), args.block).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = int(ck["step"])
        print(f"resuming from {args.resume} at step {start} (lr={args.lr:g}, batch_size={args.batch_size})", flush=True)
    else:
        print(f"tinygpt on {device}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
              f"lr={args.lr:g}, batch_size={args.batch_size}, grad_accum={args.grad_accum}", flush=True)  # fmt: skip

    @torch.no_grad()
    def estimate(src: torch.Tensor, iters: int = 10) -> float:
        model.eval()
        losses = [model(*batch(src))[1].item() for _ in range(iters)]
        model.train()
        return sum(losses) / len(losses)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_last = time.time()
    for step in range(start + 1, args.steps + 1):
        if args.oom_batch and args.batch_size >= args.oom_batch and step >= start + 5:
            raise RuntimeError(
                f"MPS backend out of memory (simulated by --oom-batch {args.oom_batch}: "
                f"batch_size={args.batch_size} would need more memory than this demo allows)"
            )
        lr = args.lr * min(1.0, step / max(1, args.warmup))
        if args.unstable_after >= 0 and step > args.unstable_after and args.lr > args.lr_critical:
            lr *= args.instability_gain  # injected: the instability a too-high lr causes late in training
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.grad_accum):
            _, loss = model(*batch(train_data))
            (loss / args.grad_accum).backward()
            total += loss.item() / args.grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
        opt.step()

        if step % args.log_every == 0:
            dt = (time.time() - t_last) * 1000 / args.log_every
            t_last = time.time()
            print(f"step {step}: loss {total:.4f}, lr {lr:.2e}, grad_norm {grad_norm:.3f}, dt {dt:.0f}ms", flush=True)
        if step % args.eval_every == 0 and math.isfinite(total):
            print(f"step {step}: train loss {estimate(train_data):.4f}, val loss {estimate(val_data):.4f}", flush=True)
        if step % args.ckpt_every == 0:
            path = out / f"ckpt_{step}.pt"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, path)
            print(f"saved checkpoint {path}", flush=True)
    print("training complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
