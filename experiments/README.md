# Experiments

Experiment definitions live in `configs/experiment/`. This directory holds what surrounds
them: sweep launchers, result tables, and analysis of completed runs.

Each experiment must document four things in its config header:

- **Objective** -- the question the run answers
- **Configuration** -- what it changes relative to defaults
- **Expected output** -- what artefacts appear, and roughly what values
- **Interpretation** -- how to read the result, and what it does *not* show

## Available now

| Experiment | Command |
| --- | --- |
| Phase 1 scaffold check | `tinyearth-config +experiment=smoke` |
| Phase 2 data pipeline check | `tinyearth-data +experiment=data_smoke` |
| Phase 3 training stack check | `tinyearth-train +experiment=baseline_smoke` |

All three run on synthetic data in seconds. Their metric *values* are meaningless; they
verify that the stack is correct and reproducible.

## Planned (Phase 4)

| Study | Axis |
| --- | --- |
| Scaling | 2M / 5M / 10M / 20M parameters |
| History length | 2 / 4 / 6 / 8 frames |
| Forecast horizon | 1 / 2 / 4 / 8 steps |
| Hidden dimension | 64 / 128 / 256 / 512 |
| State dimension | SSM-specific sweep |
| Encoder | CNN vs. frozen foundation encoder |

In every study, only the temporal backbone and the swept parameter vary. Everything else is
held fixed -- that control is what makes the efficiency comparison meaningful, and it is
enforced by tests rather than by convention.

Each run writes a flat `summary.json`, so collating a sweep is a script over those files.
