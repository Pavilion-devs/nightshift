# Changelog

## Unreleased

### 0.1.1: catalog review (requested by the reviewer at 8bbbd6e4)
- The trainer no longer inherits Hermes's environment. It gets a minimal one (`PATH`, `HOME`, user, locale, temp
  dirs, Python/conda/venv paths, `CUDA_*`/`NVIDIA_*`/`TORCH_*`, Hugging Face cache paths, `NIGHTSHIFT_*`), so
  provider keys never reach the training command. Anything else must be named in the playbook's new
  `env_passthrough` (`--env NAME` on the CLI), which the approval card shows.
- Removed `hermes nightshift approve`: the agent can reach the CLI through its terminal tool, so it could have
  approved its own proposal. Approval is `/nightshift approve` in chat or the Approve button only.
- A run re-attached without a recorded process start time is never signalled or restarted; it is marked failed
  and you are paged. A new launch retries `ps` briefly so the start time is always recorded.

### Phase 4: ship
- Fixed: phone alerts never reached Telegram. Hermes v0.23 deliberately has no agent-callable `send_message`
  tool, so `ctx.dispatch_tool("send_message")` answered "Unknown tool". Alerts now go through Hermes's documented
  `hermes send` (argv, message on stdin, Hermes's own credentials); verified live: the chart and the model's root
  cause arrived in Telegram.
- New Nightshift page (dashboard tab and a full Desktop page with a sidebar entry; the pane is its compact
  form): incidents board with an attention banner, loss-over-steps chart that keeps the healthy curve in view while
  a blow-up runs off the top, the learning rate on a second axis, diagnostics that show how far each metric moved
  against its own median, a plain-language summary, what Nightshift did, an incidents table, a run log and an
  approval dialog with a review checkbox. Data comes from one pure function (`core/board.py`).
- Fixed (found in a demo round): the spike detector looked only at the last point of each poll, so a fast trainer
  that logs many lines between polls could hide a blow-up mid-batch; it was then caught 330 steps later as
  divergence. It now checks every new point. Also fixed: a warning that repeats into a critical incident is
  handled again instead of being skipped as already open.
- README with real demo results and screenshots, SECURITY.md (threat model, every system call) and PRIVACY.md
  (what is stored, what leaves the machine and when). `scripts/sync_readme.py` copies the README into the plugin
  folder for the catalog page; a test keeps them in sync.
- Hardening: metric names parsed from logs are kept only if they are short identifiers, since they reach the
  agent's context in incident lines; checkpoint paths handed to `{resume}` are absolute; storage no longer falls
  back to a hard-coded `~/.hermes` path outside Hermes (tests inject their own directory).
- Demo scenario run a 4th time in real Hermes v0.23.0: same results.
- Fixed: the pid-reuse guard compared the command line too, which changes when a launcher re-execs itself
  (python.org's macOS Python re-execs into `Python.app`), so a re-attached run could be mistaken for a stranger's
  process. It now compares the process start time only.
- Fixed: the web-dashboard bundle (`nightshift/dashboard/dist/`) was git-ignored and missing from the repo.

### Phase 3: Desktop pane, demo kit
- One UI (`ui/core.js`) built into the Desktop plugin and a web-dashboard tab, verified live in `hermes dashboard`:
  approval card with Approve/Decline, run cards, a step-based chart that shows the diverged branch next to the
  recovered attempt, incident and restore markers, action timeline, diagnosis, two-click Stop, ack, a status-bar
  chip, and a native notification plus toast on new critical incidents (Desktop).
- Phone charts plot loss against step too, with integer step labels, and a blow-up no longer flattens the curve.
- `examples/tinygpt`: a real character-level GPT on Shakespeare (CUDA / MPS / CPU) with an injected,
  lr-dependent instability and a simulated OOM; `scripts/demo.sh` proposes both demo runs.
- Demo scenario run 3 times in real Hermes v0.23.0 with identical results: tinygpt rolled back to step 600 with
  lr 0.003 → 0.0015 and finished; the big-batch run restarted with batch 128 → 64, grad_accum 1 → 2 and finished.
- Fixed: `--resume {resume}` written as two words left a dangling `--resume` on a fresh start.
- `/nightshift forget <id|finished>` and `hermes nightshift forget …` clear finished runs from history.

### Phase 2: supervisor and Hermes wiring
- Supervisor per run: poll → detect → decide → act → diagnose → notify. Act first, so a diverged run stops burning
  GPU before any LLM call. Per-run lock; leases so exactly one Hermes process supervises a run; a run left running
  by a crashed Hermes is adopted by the next one.
- Playbook: rollback with `lr ×0.5` and OOM restart with `batch_size ÷2, grad_accum ×2`, an intervention budget, a
  cooldown that stops a run which relapses right after recovery, and a stop when recovery is impossible (no
  `{resume}` slot, no checkpoint from before the anomaly, lr at its floor). Only critical incidents page you.
- Process control: own process group per run, pid-reuse guard before every signal, zombie-aware, SIGTERM then
  SIGKILL. The whole group (dataloader workers included) is stopped.
- Consent: the agent can only propose; a human approves with `/nightshift approve`, `hermes nightshift approve` or
  the Desktop pane. One-shot Hermes processes never launch, adopt or act on a run (start grace).
- Hermes surfaces: 6 tools, `/nightshift`, `hermes nightshift` (incl. `launch -- <cmd>` and `watch`),
  `pre_llm_call` (each incident once per conversation), `on_session_start`, a bundled skill, a morning-report
  automation blueprint, dashboard routes for the Desktop pane.
- Alerts through Hermes's `send_message` with a stdlib-rendered PNG chart; capped `ctx.llm` diagnosis; secrets
  redacted before anything leaves the machine.
- Verified inside real Hermes v0.23.0 (`hermes gateway run`, plugin host isolation): a diverging run was detected,
  rolled back to the last checkpoint before the onset with lr halved, and trained to completion.

### Phase 1: engine
- Metric sources: log tailing (JSON lines, HF dict reprs, key=value, nanoGPT, tqdm/Lightning bars, custom regex),
  JSONL, CSV and TensorBoard event files (own TFRecord + protobuf reader, CRC-checked, tested on real
  tensorboardX output including TF2 tensor scalars).
- Detectors: NaN/Inf, spike, divergence, plateau, overfit, throughput drop, stall, GPU idle, crash/OOM
  classification. Onsets are traced back to where the loss left its healthy trend, so rollback never restores a
  checkpoint saved mid-climb.
- Checkpoint discovery and rollback selection (both step and time bounds must hold).
- SQLite store (WAL, per-thread connections, versioned migrations, spike-preserving downsampling).
- Synthetic trainer with lr-dependent divergence, OOM, crash, stall and overfit modes.
- 390 tests; engine coverage 95%; clean-curve sweep (6 shapes x 25 seeds x 2 lengths) raises nothing actionable.
