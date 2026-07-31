# Model configs

Populated in **Phases 3-4**.

Phase 3 adds the ConvLSTM and temporal-transformer baselines. Phase 4 adds the State Space
Model backbones and the size tiers:

| Tier | Target parameters |
| --- | --- |
| tiny | ~2M |
| small | ~5M |
| base | ~10M |
| large | ~20M |

Scaling happens primarily inside the temporal backbone; encoder and decoder stay fixed so
that size comparisons isolate the component under study.
