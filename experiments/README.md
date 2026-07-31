# Experiments

Experiment definitions live in `configs/experiment/`. This directory holds what surrounds
them: sweep launchers, result tables, and analysis of completed runs.

Each experiment documents four things in its config header: **objective, configuration,
expected output, interpretation** -- including what the result does *not* show.

## Verification experiments

| Experiment | Command |
| --- | --- |
| Phase 1 scaffold | `tinyearth-config +experiment=smoke` |
| Phase 2 data pipeline | `tinyearth-data +experiment=data_smoke` |
| Phase 3 training stack | `tinyearth-train +experiment=baseline_smoke` |
| Phase 4 SSM stack | `tinyearth-train +experiment=ssm_smoke` |

All run on synthetic data in seconds. Their metric *values* are meaningless; they verify
that the stack is correct and reproducible.

## Research sweeps

| Study | Command |
| --- | --- |
| Scaling | `tinyearth-train --multirun +experiment=scaling model=s4d,mamba,convlstm,transformer model.backbone.size=tiny,small,base,large` |
| History length | `tinyearth-train --multirun +experiment=history_length data.history_length=2,4,6,8` |
| Forecast horizon | `tinyearth-train --multirun +experiment=horizon data.horizon=1,2,4,8` |
| Hidden dimension | `tinyearth-train --multirun +experiment=hidden_dim model.backbone.kwargs.hidden_dim=64,128,256,512` |
| State dimension | `tinyearth-train --multirun +experiment=state_dim model.backbone.kwargs.state_dim=8,16,32,64,128` |

Collate with `python scripts/collate_results.py --group <group>`.

In every study only the temporal backbone and the swept parameter vary. That control is
enforced by tests, not by convention.

## Before reporting anything

**These sweeps have not been run on real data.** They default to `data=synthetic`, where
quality numbers are meaningless. Add `data=earthnet2021` and train to convergence before
treating any output as a result.

Read `docs/phase-4.md` first: parameter efficiency and FLOP efficiency disagree here, and a
result quoting only one of them would be misleading.
