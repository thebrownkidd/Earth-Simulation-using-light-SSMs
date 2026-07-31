# Training configs

Populated in **Phase 3**: optimiser, learning-rate schedule, batch size, epochs, gradient
clipping, mixed precision, checkpointing and early stopping.

Losses are configured here too. The loss system is registry-backed, so adding a loss is a
new module plus a config entry — never an edit to the training loop.
