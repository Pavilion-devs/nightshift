---
name: nightshift
description: Launch, watch and explain ML training runs with Nightshift, which detects NaN, spikes, divergence, OOM and stalls and recovers runs within a playbook the user approves.
---

# Nightshift: supervising training runs

Use the `nightshift_*` tools whenever the user wants a training run started, watched overnight, or explained.

## Launching a run (`nightshift_launch`)

1. Find the training command. Read the script's argument parser so you know the real flag names for the
   learning rate, batch size, gradient accumulation and resuming from a checkpoint.
2. Put every value recovery may change in a `{placeholder}` and give its starting value in `params`:
   `python train.py --lr {lr} --batch_size {batch_size} --grad_accum {grad_accum} --resume {resume}`
   with `params: {"lr": 3e-4, "batch_size": 64, "grad_accum": 1}`.
   - `{resume}` becomes a checkpoint path on rollback and is dropped on a fresh start. If the script cannot
     resume, say so: Nightshift will then stop a broken run instead of recovering it.
   - If the flags use other names (`--learning-rate`), keep the placeholder names `lr`, `batch_size` and
     `grad_accum`, or set `playbook.lr_param` etc. to match.
3. If checkpoints are not saved under a usual name (`ckpt_*.pt`, `checkpoint-*`, `*.ckpt`), pass
   `checkpoint_patterns`, e.g. `["out/ckpt_*.pt"]`.
4. If the trainer writes metrics to a file rather than stdout (JSONL, CSV or TensorBoard), add it to `sources`.
5. The tool returns an **approval card**. Nothing has started. Show the user the command, the directory and
   the playbook lines, and tell them to approve with `/nightshift approve <run_id>` (or the Approve button in
   the Nightshift pane). You cannot approve it yourself.

## Watching a run started elsewhere (`nightshift_watch`)

Give the log file (or JSONL/CSV/TensorBoard path). It is read-only: alerts only, no process control.

## Explaining what happened

- `nightshift_status`: every run's health at a glance.
- `nightshift_run`: one run's incidents, the diagnosis, the actions taken and the loss curve. Quote the
  actions (they say exactly which checkpoint was restored and which parameter changed) rather than guessing.
- `nightshift_report`: the overnight summary.
- `nightshift_ack`: mark incidents as seen once the user has acknowledged them.

## Stopping

Stopping is the user's call: tell them to run `/nightshift stop <run_id>` or press Stop in the pane.
