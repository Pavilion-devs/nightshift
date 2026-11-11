# Security

Nightshift starts, stops and restarts processes on your machine. This page covers what it can and can't do, and
how each limit is enforced.

## Reporting a vulnerability

Please open a [private security advisory](https://github.com/Pavilion-devs/nightshift/security/advisories/new)
rather than a public issue. We aim to reply within 72 hours.

## Threat model

| Who | Could try to | What stops it |
|---|---|---|
| **The agent** (or a prompt injection reaching it through a web page, an email or a file) | start an arbitrary command | `nightshift_launch` only *proposes*. No tool, hook or blueprint approves a run. Approval comes only from a human surface: `/nightshift approve` in chat, or the Approve button in the Desktop page / pane / dashboard tab. There is deliberately no `hermes nightshift approve`: the agent can run shell commands through its terminal tool, so a CLI approve would let it approve its own proposal. The approval card shows the exact rendered command, directory and playbook. |
| **The agent** | stop or restart someone else's process | `nightshift_watch` runs are notify-only: no signals, ever. Stop is a human command, not a tool. Nightshift signals only process groups it created (`start_new_session=True`), and it re-checks the PID's start time (`ps -o lstart=`) before every signal, so a reused PID is never signalled. A run re-attached without a recorded start time is never signalled or restarted: the run is marked failed and you are paged. |
| **The training command** (or a package it imports) | read Hermes's provider keys and tokens from its environment | The trainer gets a minimal environment, not Hermes's: `PATH`, `HOME`, user, locale, temp dirs, `PYTHONPATH` / `VIRTUAL_ENV` / `CONDA_*`, `CUDA_*` / `NVIDIA_*` / `TORCH_*` / `PYTORCH_*` / `NCCL_*`, Hugging Face cache paths, `LD_LIBRARY_PATH` and its own `NIGHTSHIFT_*` values (`core/process.py: minimal_env`). Any other variable must be named in the approved playbook's `env_passthrough`, which the approval card lists. |
| **A training log** (attacker-influenced text in stdout) | inject instructions into the agent | Log text never reaches the agent's context. `pre_llm_call` injects only Nightshift's own incident and action lines (detector kind, metric name, numbers), and a metric name is kept only if it's a short identifier (`[A-Za-z0-9_./:-[]]`, ≤ 64 characters). Log tails go only to the diagnosis call, which has no tools, and its output is shown as text, never acted on. |
| **A training log** | leak a secret to a model or chat | `nightshift.core.redact` scrubs API keys, bearer tokens, `*_KEY=` / `*TOKEN=` / `password=` values and URL credentials before a tail is stored, sent to your model or pushed to a platform. |
| **A restart** | escalate beyond what was approved | Restarts re-render the **approved argv template** with only the declared parameters (`lr`, `batch_size`, `grad_accum`, `resume`). Each is bounded: LR floor 1e-7, batch ≥ 1, intervention budget (default 3), 10-minute cooldown. A relapse inside the cooldown or an exhausted budget stops the run and pages you. `{resume}` is always the absolute path of a checkpoint matched by the run's approved checkpoint patterns. |
| **Shell metacharacters** in parameters | run a second command | Commands run as argv lists with `shell=False`. No string is ever handed to a shell. |
| **A second Hermes process** | double-supervise or double-restart a run | A lease row in SQLite (`owner`, `lease_until`) gives each run exactly one supervisor. Short-lived `hermes` invocations wait out `start_grace_seconds` and never touch a run. |

## Catalog compliance

- No core overrides, no monkeypatching, and no private Hermes imports. Hermes is reached only through
  `register_*`, `ctx.llm.complete`, `plugins.plugin_storage` and Hermes's documented `hermes send` CLI for alerts
  (`ctx.dispatch_tool("send_message", …)` is tried first, for Hermes versions that expose it as a tool).
- No credentials: Nightshift reads no `.env`, `auth.json`, keychain or token file, and needs no secrets.
- No self-updating code, no binaries, no `.pyc` files, and zero runtime dependencies.
- The Desktop plugin imports only `@hermes/plugin-sdk`, `react` and `react/jsx-runtime`. It has no `eval`, and it
  doesn't touch the DOM outside its own components.
- `hermes plugins validate nightshift --install-deps` passes, with the security scan rated `safe`.

## What the code does with your system

| Operation | Where | Why |
|---|---|---|
| `subprocess.Popen(argv, shell=False, start_new_session=True, env=minimal_env(...))` | `core/process.py` | launch the run you approved, without Hermes's environment |
| `os.killpg(pgid, SIGTERM)`, then `SIGKILL` after a grace period | `core/process.py` | stop or restart that run's process group |
| `ps -o lstart= -p <pid>` | `core/process.py` | PID-reuse guard before any signal |
| `hermes send --to <target> --file - --json` (argv, message on stdin; Hermes reads the platform credentials) | `adapter/notify.py` | push a critical alert to your messaging platform |
| `nvidia-smi --query-gpu=utilization.gpu` (if installed) | `core/process.py` | GPU-idle detection |
| read files | the run's directory, log, metric files, checkpoint globs | metrics and checkpoint discovery |
| write files | `~/.hermes/plugin-data/nightshift/`, `~/.hermes/image_cache/nightshift/` | run database, logs, alert charts |
