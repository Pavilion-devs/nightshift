# Nightshift: implementation plan

> **Hermes takes the night shift on your training runs.**
> Start a run, go to sleep. Nightshift watches every metric, catches NaN, loss spikes, divergence, OOM crashes and
> stalls the moment they happen, recovers the run within limits you approved, and wakes you only when it matters,
> with the chart.

Entry for the **Hermes Agent Plugin Hackathon** (Nous Research). Deadline: **Nov 11, 2026, 5 PM EST**.

---

## 1. Why this wins

| Judging criterion | How Nightshift answers it |
|---|---|
| **Usefulness** | Every ML practitioner has lost a night of GPU time to a run that died at 3 AM. Nous trains models for a living, so the judges feel this personally. |
| **Reliability** | The watch loop is deterministic Python: no model calls while a run is healthy. Detectors are unit-tested against recorded and synthetic curves. Every action is bounded by an approved playbook and an intervention budget. |
| **Polish** | Live loss charts in Hermes Desktop with incident markers, a status-bar chip, phone alerts carrying a rendered chart, and a morning report. Zero runtime dependencies. |
| **Creativity** | No plugin like it exists. We found none among ~420 hackathon PRs and ~1,000 catalog entries (no hits for TensorBoard, training run, loss curve). |
| **"More than a prompt or skill"** | Needs a long-running supervisor, metric parsing, process control, checkpoint rollback, charts and push alerts. A prompt cannot watch a process for eight hours. |

## 2. Scope

### In scope (v1, ships for the hackathon)

1. **Launch and supervise** a training command (`nightshift_launch`), or **attach** to a run already in progress
   (`nightshift_watch`, notify-only).
2. **Metric sources**, all stdlib:
   - stdout/log lines (built-in regex presets for common trainers, e.g. `loss=…`, `step …`, HF Trainer dicts, Lightning);
   - JSONL and CSV metric files;
   - TensorBoard event files (our own TFRecord + protobuf reader, no `tensorboard` dependency).
3. **Detectors**: NaN/Inf, loss spike, divergence, plateau, overfitting (val up while train goes down), throughput drop,
   stall (alive but silent), crash with OOM classification, optional GPU idle (`nvidia-smi` when present).
4. **Playbook actions** (per incident type, approved at launch): `notify`, `stop`, `rollback_restart` (newest healthy
   checkpoint, with bounded overrides such as `lr *= 0.5`), `oom_restart` (halve batch size, double grad accumulation).
   Guarded by a max-interventions budget and a cooldown.
5. **Diagnosis**: on an incident only, one `ctx.llm` call summarizes the root cause from the metric window, log tail and
   config. Capped per run.
6. **Alerts**: in-conversation notices, a push to the user's messaging platform (Telegram/Discord) with a PNG chart
   rendered by a stdlib rasterizer, and native Desktop notifications.
7. **Morning report**: an automation blueprint ("While you slept: 2 incidents, 2 recovered, 4.1 GPU-hours saved").
8. **Desktop**: a right pane with live multi-metric charts, incident markers and an action timeline, a run list, and a
   status-bar chip (`🌙 2 runs · healthy`).
9. **Demo kit**: a tiny char-level GPT trainer with fault-injection flags, plus a dependency-free synthetic trainer for
   tests and CI.

### Out of scope (roadmap only)

W&B/MLflow API sources, multi-node and Slurm, cloud GPU provisioning, hyperparameter search, auto-tuning beyond the
playbook's bounded overrides.

## 3. Architecture

```
                  ┌─────────────────────── Hermes process ───────────────────────┐
 user / agent ──▶ │ tools: nightshift_launch / watch / status / act / report      │
                  │ /nightshift slash cmd · `hermes nightshift` CLI · blueprint   │
                  │                                                               │
                  │  Supervisor (1 thread per run)                                │
                  │    ├─ ProcessHandle  (subprocess, own process group, log tee) │
                  │    ├─ Sources ──▶ MetricStream ──▶ Detectors ──▶ Incident     │
                  │    ├─ Playbook ──▶ Action (stop / rollback / oom_restart)     │
                  │    ├─ Diagnoser (ctx.llm, incident-only)                      │
                  │    └─ Notifier (post_message · platform push · desktop)       │
                  │  Store: SQLite in plugin data dir (runs, metrics, incidents)  │
                  │  RPC/REST ◀──────────────── Desktop pane (desktop/plugin.js)  │
                  └───────────────────────────────────────────────────────────────┘
```

Core rule: **the engine (`nightshift/core/`) never imports Hermes.** Everything Hermes-specific lives in the adapter
layer (`nightshift/adapter/`), so the engine is fully testable with plain pytest.

## 4. Repository layout

The plugin itself is the `nightshift/` package (catalog `subdir: nightshift`), the same layout Kata validated with.
Install locally with `ln -sfn "$PWD/nightshift" ~/.hermes/plugins/nightshift`.

```
neus/                                  # repo root (published as github.com/Pavilion-devs/nightshift)
├── pyproject.toml                     # no runtime deps; dev: pytest, ruff, coverage
├── nightshift/                        # ← the Hermes plugin directory
│   ├── plugin.yaml                    # manifest v2: tools, hooks, config_schema, requires_hermes
│   ├── __init__.py                    # register(ctx): thin, delegates to nightshift.adapter.register
│   ├── desktop/plugin.js              # Desktop pane + status chip (ESM, jsx() calls, no JSX)
│   ├── core/                          # pure engine, NO hermes imports
│   │   ├── models.py                  # Run, MetricPoint, Incident, Action, Playbook dataclasses
│   │   ├── sources/
│   │   │   ├── logtail.py             # follow a file, regex presets
│   │   │   ├── jsonl.py  csvfile.py
│   │   │   └── tfevents.py            # TFRecord framing + minimal protobuf decode (scalars)
│   │   ├── detectors.py               # pure functions over a MetricWindow → list[Incident]
│   │   ├── checkpoints.py             # discover + rank checkpoints by step, mark healthy
│   │   ├── playbook.py                # incident → action, budget, cooldown, overrides
│   │   ├── process.py                 # launch/stop/restart, process groups, exit classification
│   │   ├── supervisor.py              # the watch loop (thread), wires everything
│   │   ├── store.py                   # SQLite (WAL), downsampling, retention
│   │   ├── chart_png.py               # stdlib line-chart rasterizer (zlib PNG)
│   │   └── report.py                  # incident cards, morning report, GPU-hours-saved math
│   ├── adapter/                       # the only place that touches Hermes APIs
│   │   ├── register.py                # tools, hooks, commands, cli, blueprint, rpc
│   │   ├── tools.py                   # handlers: (args, **kw) -> JSON str, never raise
│   │   ├── schemas.py                 # tool schemas the model sees
│   │   ├── notify.py                  # post_message / send_message push / desktop event
│   │   ├── diagnose.py                # ctx.llm prompt + budget
│   │   └── approvals.py               # pending launches + /nightshift approve
│   └── skills/nightshift/SKILL.md     # bundled skill: how the agent should use the tools
├── examples/
│   ├── tinygpt/train.py               # demo trainer (torch, MPS/CUDA/CPU), --inject-spike-at, --oom-at-batch
│   └── synthetic/train.py             # stdlib fake trainer: realistic curves + checkpoints, for tests/CI
├── tests/                             # pytest; engine tests need neither Hermes nor torch
├── docs/                              # screenshots, demo assets, architecture svg
├── README.md  SECURITY.md  PRIVACY.md  CHANGELOG.md  LICENSE (MIT)
└── .github/workflows/ci.yml           # ruff + pytest (3.11–3.13, linux/macos)
```

## 5. Hermes surfaces used

| Surface | Use |
|---|---|
| `ctx.register_tool` | `nightshift_launch`, `nightshift_watch`, `nightshift_status`, `nightshift_incidents`, `nightshift_act`, `nightshift_report` |
| `ctx.register_command` | `/nightshift` (status, ack, stop, report) |
| `ctx.register_cli_command` | `hermes nightshift status|runs|report|stop` |
| `ctx.register_hook("pre_llm_call")` | inject a 1–3 line status **only** when an unacknowledged incident exists |
| `ctx.register_hook("on_session_start")` | resume supervisors for runs still alive after a Hermes restart |
| `ctx.llm` | incident diagnosis (capped) |
| `ctx.post_message` / platform push | alerts with chart |
| `ctx.register_automation_blueprint` | "Morning training report" |
| `ctx.register_rpc` + `ctx.emit_to_session` (or `plugin_api.py`) | live data for the Desktop pane |
| `ctx.register_skill` | bundled usage skill |
| `config_schema` | thresholds, playbook defaults, alert channel, LLM budget → Desktop settings form |
| Desktop SDK | `PANES_AREA` (right), `STATUSBAR_AREAS.right`, `ctx.os` notifications, `ctx.storage` |

## 6. Detector specs (v1 defaults, all configurable)

| Detector | Rule | Default severity → action |
|---|---|---|
| `nan_inf` | any tracked loss is NaN/Inf | critical → `rollback_restart` (lr × 0.5) |
| `spike` | robust z-score (median/MAD over last 50 points) > 6 **and** value > 2× EMA | warn → `notify`; if it repeats within 3 points → `rollback_restart` |
| `divergence` | EMA(loss) > 1.5× best EMA for 30 consecutive points | critical → `rollback_restart` |
| `plateau` | best value improved < 0.5% over the last 15% of steps (min 500 steps) | info → `notify` |
| `overfit` | val ↑ for 3 consecutive evals while train ↓ | warn → `notify` |
| `throughput` | steps/sec < 50% of the rolling median for 2 min | warn → `notify` |
| `stall` | process alive, no new metric for max(5 min, 10× median interval) | critical → `notify` |
| `crash` | non-zero exit; OOM if the log matches CUDA/MPS OOM patterns | critical → `oom_restart` or `notify` |
| `gpu_idle` | `nvidia-smi` util < 5% for 5 min while alive (skipped if unavailable) | warn → `notify` |

Intervention guard rails: at most **3 automatic interventions per run**, **10 min cooldown**, never restart from a
checkpoint written after the first anomalous point, LR overrides floored at `lr_min`. When the budget is exhausted,
the run is stopped and the user is paged.

## 7. Safety and catalog compliance (rules 9, 11, 12, 13, 17)

- **Consent first.** `nightshift_launch` shows the exact command, working directory, the playbook and the action
  budget, and needs explicit user approval. In unattended contexts (cron, hooks) it refuses to launch.
  Restarts reuse only the approved command template, with only the overrides the playbook declares.
- **No core overrides, no auto-approval, no YOLO.** Agent-issued `terminal` calls keep their normal approvals.
- **Disclosure** (README, PR body and `plugin.yaml` description): launches background processes the user approved,
  sends signals to process groups it launched, reads the run's log/metric/checkpoint paths, writes SQLite under the
  plugin data dir, makes optional LLM calls on incidents, pushes alerts to the user's configured platform.
  No telemetry. No network calls of its own.
- **Secrets**: none needed. If ever needed, only via `agent.secret_scope.get_secret`.
- **Payloads**: no binaries, no `.pyc`, zero runtime dependencies.
- `hermes plugins validate . --install-deps` clean, security scan `safe` (or `caution` with every finding explained).

## 8. Build phases

| Phase | When | Deliverable | Exit check |
|---|---|---|---|
| **0. Spike** | Nov 8 AM | Install Hermes (CLI + Desktop) and set up a model. Scaffold the plugin. Answer the open questions in §10 against the **installed source**. `git init`, repo created on GitHub. | Empty plugin loads, `validate` passes, one dummy tool callable, Desktop shows a dummy pane |
| **1. Engine** | Nov 8 | `core/`: models, sources (logtail, jsonl, csv, tfevents), detectors, store, checkpoints, synthetic trainer | `pytest` green, ≥ 90% engine coverage, every detector fires on its fixture and stays silent on clean curves |
| **2. Supervisor + Hermes wiring** | Nov 9 | process control, playbook, supervisor loop, tools/command/cli/hooks, diagnosis, notifier, PNG charts, restart-resume | End-to-end in real Hermes: the synthetic trainer diverges and is auto-recovered; a Telegram alert arrives with the chart |
| **3. Desktop + report + demo kit** | Nov 10 | pane, status chip, morning-report blueprint, `examples/tinygpt` with fault injection | Full demo script runs start to finish **3 times in a row** without manual fixes |
| **4. Ship** | Nov 10 PM – Nov 11 AM | README, SECURITY/PRIVACY, screenshots, ≤ 90 s video, catalog PR, X post, entry form | PR open with CI green, video posted, form submitted **by Nov 11, 12 PM EST** (5 h buffer) |

## 9. Demo script (≤ 90 s)

| Time | Shot |
|---|---|
| 0–8 s | Cold open: a loss curve goes vertical at 03:12. Caption: "Your run died at 3 AM. You found out at 9." |
| 8–22 s | Hermes Desktop: "Train tinygpt on shakespeare, lr 3e-3, watch it overnight." The consent card shows the command and playbook → Approve. The Nightshift pane opens with a live curve. |
| 22–45 s | The injected spike hits → NaN. A red marker appears; the pane reads "nan_inf detected in 1.8 s". Rollback to step 1200, lr 3e-3 → 1.5e-3, restart; the curve continues downward. The phone buzzes with a Telegram card and chart. |
| 45–62 s | Second run hits OOM → `oom_restart` (batch 64 → 32, grad-accum 1 → 2), recovered. A plateau notice is info only, so no page. |
| 62–80 s | Morning: the report blueprint fires: "2 incidents, 2 recovered, 4.1 GPU-hours saved." Ask "why did it diverge?" → the diagnosis answers. |
| 80–90 s | Title card: **Nightshift: Hermes takes the night shift.** `hermes plugins install nightshift` |

## 10. Open questions settled in Phase 0

Read from `NousResearch/hermes-agent` main source on Nov 7 (the installed v0.23.0 copy is re-checked once the
install finishes). File and function names are given so each answer can be re-verified.

1. **Alerts from a background thread → `ctx.dispatch_tool("send_message", {"action": "send", "target":
   "telegram", "message": "..."})`.** `hermes_cli/plugin_messages.py` (`post`) refuses outside a bound session
   ("a timer, a plugin thread, cron … is refused"), so `post_message` only works inside a tool/command turn.
   `ctx.inject_message` (`hermes_cli/plugins_injection.py`) triggers an agent turn and needs
   `plugins.entries.nightshift.allow_gateway_injection: true`, so we don't use it for alerts.
   `tools/send_message_tool.py` (`_handle_send`, `_resolve_tool_target`): a bare platform target (`"telegram"`)
   resolves to that platform's home channel; media attachments go through `BasePlatformAdapter.extract_media`
   (`MEDIA:<path>` syntax, to be confirmed end-to-end in Phase 2).
2. **Consent → a two-step pending launch.** `ctx.ask_choice` (`hermes_cli/plugin_choices.py`) only works from a
   sync slash-command handler, and no plugin API exists for approving a tool call. So `nightshift_launch` only
   *proposes* a run (command, cwd, playbook, budget) and returns a pending id. A human starts it with
   `/nightshift approve <id>` (the slash handler confirms with `ctx.ask_choice`) or the Approve button in the
   Desktop pane. The agent can never start a process on its own.
3. **Desktop data → `ctx.register_rpc(name, handler)`** (`hermes_cli/plugin_rpc.py`), served as
   `plugin.nightshift.<name>` over the gateway socket and called from the pane with
   `host.request('plugin.nightshift.status', {})`. It doesn't need the web dashboard. Live pushes:
   `ctx.emit_to_session(session_id, event, payload)`. v1 polls every 2 s, which is simpler and good enough.
4. **Background work → our own daemon `threading.Thread`, stopped via `ctx.on_unload(callback)`**
   (`hermes_cli/plugins_context.py`). `ctx.spawn_task` needs a running asyncio loop, which plugin load doesn't
   guarantee. Training processes start with `start_new_session=True`, so they outlive a Hermes restart.
   Supervisors re-attach on load from the SQLite state.
5. **LLM → `ctx.llm.complete(messages, max_tokens=…, timeout=…)` or `ctx.llm.complete_structured(instructions=…,
   input=[…], json_schema=…)`** (`agent/plugin_llm.py`, `PluginLlm`). It runs on the user's active model and auth.
   The result has `.text`, `.usage`, `.audit`. Each call is audit-logged by the host, and we cap the number of calls.
6. **`requires_hermes: ">=0.23.0"`**: the installer reports v0.23.0 as the latest stable release. The Kata entry
   validated on 0.23.0 with `>=0.22.0`; we keep 0.23.0 until we test lower.
7. **`register_tool(name, toolset, schema, handler, check_fn=None, requires_env=None, is_async=False,
   description="", emoji="", override=False)`**: there is no timeout parameter, so every handler must return
   immediately and all long work runs on the supervisor thread.

Other confirmed surfaces: `register_command(name, handler, description, args_hint)`,
`register_cli_command(name, help, setup_fn, handler_fn)`, `register_automation_blueprint(key, *, title,
description, schedule_template, prompt_template, …)` and `register_skill(name, path, description)`
(`hermes_cli/plugins_content.py`). The durable data dir is `plugins.plugin_storage.plugin_db("nightshift",
"nightshift.db")`, as Kata uses. Desktop SDK: `STATUSBAR_AREAS`, `Tip`, `host` come from `@hermes/plugin-sdk`; panes
use `area: 'panes'` with `data: { placement: 'right', width }` (reference:
`NousResearch/hermes-example-plugins/example-desktop-statusbar/plugin.js`).

## 11. Risks

| Risk | Mitigation |
|---|---|
| Platform push from a thread isn't supported cleanly | Fall back to the Desktop notification + a pending-alert queue delivered on the next turn via `pre_llm_call`; demo Telegram via the documented path from Phase 0 |
| torch/MPS flakiness in the demo | Fault injection is deterministic; the synthetic trainer is a backup that needs no torch |
| Security scan flags `subprocess`/`os.killpg` | Expected `caution`; explain each call in the PR; no shell=True; argv lists only |
| Desktop SDK has no charts | Hand-built SVG via `jsx('svg', …)`; it's only polylines plus markers |
| Running out of time | Phases are cut-safe: after Phase 2 we already have a shippable plugin; Desktop and the report are polish |
| Overlap with the user's Kata entry | Different category (ML ops vs routine compilation); both can be submitted |

## 12. Definition of done

- [ ] `hermes plugins validate . --install-deps` passes on the pinned SHA; scan result documented
- [ ] `pytest` green on 3.11–3.13, linux + macos CI; ruff clean
- [ ] Demo script reproducible 3× in a row in real Hermes with real Telegram delivery
- [ ] README: pitch, GIF, quick start, surfaces, safety model, disclosure, config table, limitations
- [ ] Catalog PR adding exactly `plugin-catalog/nightshift.yaml` (full 40-char SHA, quoted version, capabilities match)
- [ ] Demo video ≤ 90 s posted on X tagging @NousResearch
- [ ] Entry form submitted with both links
