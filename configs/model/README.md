# Model configs

| Config | Backbone | Notes |
| --- | --- | --- |
| `s4d.yaml` | Diagonal SSM | **The backbone this project studies.** Parallel in time. |
| `mamba.yaml` | Selective SSM | Input-dependent dynamics; sequential scan. |
| `convlstm.yaml` | ConvLSTM | Baseline. **Default.** The only autoregressive one. |
| `transformer.yaml` | Temporal transformer | Baseline. Attention over time only. |

Select with `model=s4d`, `model=mamba`, `model=convlstm` or `model=transformer`. That single
override is the *only* change needed -- the encoder and decoder are held fixed, which is
what makes a difference in results attributable to the backbone.

## Sizing

Prefer the calibrated tiers over raw widths, so comparisons happen at matched budgets:

```bash
tinyearth-train model=s4d model.backbone.size=base     # ~10M parameters
```

| Backbone | tiny (2M) | small (5M) | base (10M) | large (20M) |
| --- | ---: | ---: | ---: | ---: |
| `s4d` | 272 | 448 | 672 | 960 |
| `mamba` | 256 | 416 | 608 | 880 |
| `convlstm` | 80 | 128 | 192 | 272 |
| `transformer` | 128 | 208 | 288 | 416 |

Values are `hidden_dim`; every tier fixes `n_layers=4` so width is the only axis that
varies. An explicit `kwargs.hidden_dim` overrides the tier.

Check any configuration without training:

```bash
tinyearth-model --compare
tinyearth-model model=s4d model.backbone.kwargs.hidden_dim=256
```

Recalibrate after changing a block structure: `python scripts/calibrate_sizes.py`.
`tests/test_sizes.py` fails if the stored table drifts.
