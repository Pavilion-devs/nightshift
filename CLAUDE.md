# CLAUDE.md: Nightshift

Nightshift is a Hermes Agent plugin that babysits ML training runs: it watches metrics, detects failures, recovers runs
within an approved playbook, and alerts the user. It is our entry for the Nous Research **Hermes Agent Plugin
Hackathon** (deadline **Nov 11, 2026, 5 PM EST**). The full plan, scope, phases and demo script are in
[plan.md](plan.md). Read it before starting any phase.

## Ground rules

1. **The engine never imports Hermes.** `nightshift/core/` is pure stdlib Python. Only `nightshift/adapter/` and
   `nightshift/__init__.py` touch Hermes APIs. If you need Hermes behaviour in core, add an interface in core and implement
   it in the adapter.
2. **Zero runtime dependencies.** Stdlib only (sqlite3, zlib, struct, threading, subprocess, re, json, csv).
   torch is allowed only in `examples/tinygpt/`; pytest and ruff are dev-only.
3. **Verify Hermes APIs against the installed source, never against memory or docs summaries.** Before using any
   `ctx.*` method, read it in the installed package (find it with `python -c "import hermes_cli, os;
   print(os.path.dirname(hermes_cli.__file__))"`), plus `hermes_cli/data/plugin_api.json` for the import allowlist.
   Record each confirmed answer in plan.md §10.
4. **Tool handlers**: signature `(args: dict, **kwargs) -> str`, always return JSON, never raise (catch, log, return
   `{"ok": false, "error": ...}`), return in well under a second. Long work runs in the supervisor thread, never inside
   a tool call (plugin tools are capped at 420 s).
5. **Silent while healthy.** No LLM calls, no notifications and no `pre_llm_call` injection unless an incident is
   open. Diagnosis calls are budget-capped per run.
6. **Consent and bounded autonomy.** A run is launched only after the user approves the exact command, cwd, playbook
   and intervention budget. Restarts reuse that approved template with only the declared overrides. Never auto-approve,
   never disable guards, never run `shell=True`; use argv lists. Signal only process groups Nightshift itself started.
   Attached (`nightshift_watch`) runs are notify-only.
7. **Catalog compliance** (`plugin-catalog/README.md` in hermes-agent): no core overrides or monkeypatching, no reading
   `.env`/`auth.json`/keychains, secrets only via `agent.secret_scope.get_secret`, no self-updating code, no binaries
   or `.pyc` in the repo, capabilities in `plugin.yaml` must match exactly what `register()` registers, and every risky
   behaviour is disclosed in the README and PR body.
8. **Desktop plugin** (`nightshift/desktop/plugin.js`): one ESM file, imports only `@hermes/plugin-sdk`, `react`,
   `react/jsx-runtime`. No JSX syntax (use `jsx()`/`jsxs()`), no DOM reaching, no raw `<webview>`, no eval. Charts are
   hand-built SVG.
9. **Logging**: use `logging.getLogger("nightshift")`, never `print()` in plugin code (stdout goes to the gateway log).
10. **Secrets in logs**: run `agent.redact.redact_sensitive_text` (adapter) or our own masking (core) over log tails
    before storing them or sending them to an LLM or a platform.

## Commands

```bash
# dev setup
python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'

# tests + lint (must pass before every commit)
pytest -q
ruff check . && ruff format --check .

# run the engine standalone against the synthetic trainer (no Hermes needed)
python -m nightshift.core.cli_dev launch -- python examples/synthetic/train.py --diverge-at 600

# install into a local Hermes for real testing
ln -sfn "$PWD/nightshift" ~/.hermes/plugins/nightshift
hermes plugins enable nightshift
HERMES_PLUGINS_DEBUG=1 hermes
hermes plugins validate nightshift --install-deps
hermes logs --level WARNING
```

## Code style

- Python 3.11+, type hints everywhere, dataclasses for models, ruff (line length 110).
- Pure functions for detectors: `detect_*(window: MetricWindow, cfg: DetectorConfig) -> list[Incident]`.
- Every detector ships with a firing fixture and a clean-curve fixture in `tests/fixtures/`.
- Time comes through an injectable clock (`core/clock.py`), so tests never sleep.
- SQLite in WAL mode, one connection per thread, schema version table, idempotent migrations.
- Small modules; no file over ~400 lines.

## Testing priorities

1. Detectors: no false positives on clean real-world curves; this is our "reliability" score.
2. Playbook: budget, cooldown, never restoring a checkpoint written after the anomaly began.
3. Process control: stop/restart only touches our own process group; crash and OOM classification.
4. tfevents reader against a real file written by `torch.utils.tensorboard.SummaryWriter` (fixture committed as
   bytes generated once; no torch needed at test time).
5. Adapter: a fake `ctx` that records registrations, asserting they match `plugin.yaml` capabilities.

## Workflow

- Work phase by phase from plan.md §8 and tick off its definition of done. Don't start Desktop polish before Phase 2's
  end-to-end check passes in real Hermes.
- Commit small and often with clear messages. Never commit `.venv`, `__pycache__`, run outputs, checkpoints or
  `*.pyc`.
- When an open question in plan.md §10 is answered, write the answer there (file + function name from the Hermes
  source).
- Anything cut for time goes under "Roadmap" in the README, not silently dropped.
