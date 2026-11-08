# Privacy

Nightshift has **no telemetry and makes no network calls of its own**. Everything it stores stays on your machine.

## What is stored, and where

| Data | Location | Kept |
|---|---|---|
| Runs: name, command template, directory, parameters, playbook, status | `~/.hermes/plugin-data/nightshift/nightshift.db` (SQLite) | until you run `/nightshift forget <id\|finished>` |
| Metrics parsed from the run (step, name, value, time) | same database | per run |
| Incidents, actions taken, diagnoses | same database | per run |
| The run's stdout/stderr | `~/.hermes/plugin-data/nightshift/runs/<run>/attempt-N.log` | until you delete them |
| Alert charts (PNG) | `~/.hermes/image_cache/nightshift/` | until Hermes clears its image cache |

With a Hermes profile active, `~/.hermes` is that profile's home.

## What leaves your machine, and only when

| When | What | To |
|---|---|---|
| A **critical** incident, if `diagnose` is on (default), at most `max_diagnoses` (3) times per run | the run's command and parameters, the incident, the action taken, recent metric values and a **redacted** log tail (40 lines; ≤ 12,000 characters in all) | **your own configured model**, through Hermes's `ctx.llm` (audit-logged by Hermes) |
| A **critical** incident, if an alert target is set or the run was requested from a chat platform | a short alert, the chart and the diagnosis | **your own messaging platform** (Telegram, Discord, …), through Hermes's `send_message` |

Set `diagnose: false` to keep everything local. Set `alert_target: none` to turn off pushes.

Before anything is stored or sent, log tails are scrubbed of API keys, tokens, `KEY=value` secrets and URL
credentials. This is best-effort pattern matching, so don't print secrets in your training logs.
