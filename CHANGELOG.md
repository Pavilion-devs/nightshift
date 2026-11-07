# Changelog

## Unreleased

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
