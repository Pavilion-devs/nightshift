# Changelog

## Unreleased

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
