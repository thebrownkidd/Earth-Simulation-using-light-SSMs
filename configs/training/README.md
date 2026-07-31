# Training configs

| Config | Purpose |
| --- | --- |
| `default.yaml` | 20 epochs, AdamW + cosine with warmup, checkpointing, metrics. |
| `smoke.yaml` | 2 epochs x 4 steps. Verification only; results are meaningless. |

Losses are configured under `model.loss`, not here -- they belong to the model. The loss
system is registry-backed, so adding one is a new module plus a config entry, never an edit
to the training loop.

Efficiency profiling is **on by default** (`evaluation.efficiency: true`). Efficiency is
this project's primary result and should never depend on remembering a flag.

Note `scheduler.warmup_epochs: 1` in the default: the transformer baseline is markedly less
stable without warmup.
