"""Tool schemas: what the model sees. Descriptions say plainly that launching needs a human approval."""

from __future__ import annotations

from typing import Any

_SOURCES = {
    "type": "array",
    "description": (
        "Extra metric files to read instead of the process output: "
        '[{"type": "jsonl"|"csv"|"tfevents"|"log", "path": "relative/to/cwd"}]. '
        "Leave empty to parse the training command's own stdout (works for most trainers)."
    ),
    "items": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["log", "jsonl", "csv", "tfevents"]},
            "path": {"type": "string"},
            "pattern": {"type": "string", "description": "Optional regex with named groups for log lines."},
        },
        "required": ["type", "path"],
    },
}

LAUNCH: dict[str, Any] = {
    "name": "nightshift_launch",
    "description": (
        "Propose an ML training run for Nightshift to launch and supervise overnight: it watches the metrics, "
        "detects NaN, loss spikes, divergence, out-of-memory crashes and stalls, and recovers the run by "
        "rolling back to the last healthy checkpoint with a lower learning rate (or halving the batch size on "
        "OOM), within limits the user approves. This tool does NOT start anything: it returns an approval "
        "card. Show the card to the user and ask them to approve it with `/nightshift approve <run_id>` or "
        "the Approve button in the Nightshift pane. Put tunable values in {placeholders} so recovery can "
        "change them, e.g. command `python train.py --lr {lr} --batch-size {batch_size} --resume {resume}` "
        'with params {"lr": 3e-4, "batch_size": 64}. {resume} is filled with a checkpoint path on '
        "rollback and dropped on a fresh start; without it Nightshift can only stop a broken run, not "
        "recover it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The training command, with {placeholders}."},
            "params": {
                "type": "object",
                "description": "Initial values for every {placeholder} except {resume}.",
                "additionalProperties": {"type": ["number", "string", "integer"]},
            },
            "cwd": {
                "type": "string",
                "description": "Working directory. Defaults to the session's directory.",
            },
            "name": {"type": "string", "description": "Short name for the run."},
            "sources": _SOURCES,
            "checkpoint_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Glob(s) for checkpoints relative to cwd, e.g. ['out/ckpt_*.pt']. "
                "Default: common layouts (ckpt*, checkpoint-*, *.pt, *.ckpt, *.safetensors).",
            },
            "playbook": {
                "type": "object",
                "description": "Optional overrides: max_interventions (default 3), cooldown_s (600), "
                "lr_param ('lr'), lr_factor (0.5), batch_param ('batch_size'), accum_param ('grad_accum'), "
                "rules {incident_kind: 'notify'|'stop'|'rollback_restart'|'oom_restart'}, "
                "env_passthrough (list of environment variable names the trainer may see beyond a minimal "
                "environment, e.g. ['WANDB_API_KEY']; shown to the user on the approval card).",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
}

WATCH: dict[str, Any] = {
    "name": "nightshift_watch",
    "description": (
        "Watch a training run that is already running elsewhere, read-only: Nightshift reads its log or metric "
        "files and alerts on NaN, spikes, divergence, stalls and overfitting, but never touches the process."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "log_path": {
                "type": "string",
                "description": "The run's log file (stdout redirected to a file).",
            },
            "sources": _SOURCES,
            "cwd": {"type": "string"},
            "name": {"type": "string"},
        },
        "additionalProperties": False,
    },
}

STATUS: dict[str, Any] = {
    "name": "nightshift_status",
    "description": (
        "Show the training runs Nightshift is watching: health, step, latest train/val loss, open incidents, "
        "interventions and pending approvals. Read-only."
    ),
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}

RUN: dict[str, Any] = {
    "name": "nightshift_run",
    "description": (
        "Details of one run: command, params, every incident (with diagnosis), every action taken and a "
        "downsampled loss curve. Use it to explain what happened to a run. Read-only."
    ),
    "parameters": {
        "type": "object",
        "properties": {"run_id": {"type": "string"}},
        "required": ["run_id"],
        "additionalProperties": False,
    },
}

REPORT: dict[str, Any] = {
    "name": "nightshift_report",
    "description": (
        "The overnight report: runs, incidents, automatic recoveries, stops and an estimate of GPU-hours "
        "of training rescued, for the last N hours. Read-only."
    ),
    "parameters": {
        "type": "object",
        "properties": {"hours": {"type": "number", "description": "Look-back window, default 12."}},
        "additionalProperties": False,
    },
}

ACK: dict[str, Any] = {
    "name": "nightshift_ack",
    "description": "Mark Nightshift incidents as seen (for one run, or all), so they stop being mentioned.",
    "parameters": {
        "type": "object",
        "properties": {"run_id": {"type": "string"}},
        "additionalProperties": False,
    },
}

ALL = [LAUNCH, WATCH, STATUS, RUN, REPORT, ACK]
