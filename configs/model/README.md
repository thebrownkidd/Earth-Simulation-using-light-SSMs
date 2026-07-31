# Model configs

| Config | Backbone | Notes |
| --- | --- | --- |
| `convlstm.yaml` | ConvLSTM (Shi et al., 2015) | **Default.** Strictly sequential in time. |
| `transformer.yaml` | Temporal transformer | Attention over time only; non-autoregressive. |

Select with `model=convlstm` or `model=transformer`. That single override is the *only*
change needed -- the encoder and decoder are held fixed, which is what makes a difference in
results attributable to the backbone.

Capacity is scaled with `model.backbone.kwargs.hidden_dim`. Use `tinyearth-model` to check
the resulting parameter count and cost without training:

```bash
tinyearth-model --compare
tinyearth-model model.backbone.kwargs.hidden_dim=256
```

Measured counts are tabulated in `docs/models.md`.

Phase 4 adds the State Space Model configs and the size tiers (tiny ~2M, small ~5M,
base ~10M, large ~20M).
