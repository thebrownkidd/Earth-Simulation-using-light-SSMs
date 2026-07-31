# Experiments

Experiment definitions live in `configs/experiment/`. This directory holds what surrounds
them: sweep launchers, result tables, and analysis of completed runs.

Each experiment must document four things, in its config header:

- **Objective** — the question the run answers
- **Configuration** — what it changes relative to defaults
- **Expected output** — what artefacts appear, and roughly what values
- **Interpretation** — how to read the result, and what it does *not* show

## Planned (Phases 3-4)

| Study | Axis |
| --- | --- |
| Scaling | 2M / 5M / 10M / 20M parameters |
| History length | 2 / 4 / 6 / 8 frames |
| Forecast horizon | 1 / 2 / 4 / 8 steps |
| Hidden dimension | 64 / 128 / 256 / 512 |
| State dimension | SSM-specific sweep |
| Encoder | CNN vs. frozen foundation encoder |

In every study, only the temporal backbone and the swept parameter vary. Everything else
is held fixed — that control is what makes the efficiency comparison meaningful.
