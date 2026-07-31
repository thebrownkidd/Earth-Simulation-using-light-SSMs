# Configuration

Every TinyEarth experiment is reproducible from a single [Hydra](https://hydra.cc) config.

## Composition

`configs/config.yaml` is the root. Its `defaults` list controls composition order:

```yaml
defaults:
  - tinyearth_schema   # registered structured schema -- first, so it validates everything
  - _self_             # this file's values -- last, so they win over groups
```

`tinyearth_schema` is the dataclass tree in `src/tinyearth/config/schema.py`, registered
with Hydra's `ConfigStore` by `tinyearth.config.store.register_configs()`. Registering it
*before* composition is what makes configs type-checked.

## Why structured configs

Without a schema, a typo composes silently and fails three hours into a sweep:

```bash
$ tinyearth-config seed.valu=7        # note the typo
# unstructured: silently adds a new key, uses the default seed
# structured:   Could not override 'seed.valu'. No match in the config.
```

Types are enforced too:

```bash
$ tinyearth-config seed.value=not-an-int
Error merging override seed.value=not-an-int
Value 'not-an-int' of type 'str' could not be converted to Integer
```

## Schema

| Group | Owner | Contents |
| --- | --- | --- |
| `run` | Phase 1 | `name`, `group`, `notes`, `tags`, `device` |
| `seed` | Phase 1 | `value`, `deterministic`, `cudnn_benchmark` |
| `paths` | Phase 1 | `data`, `outputs`, `cache` |
| `logging` | Phase 1 | `level`, `rich`, `log_file`, `tensorboard.*`, `wandb.*` |
| `data` | Phase 2 | typed `Any` for now |
| `model` | Phases 3-4 | typed `Any` for now |
| `training` | Phase 3 | typed `Any` for now |

The three placeholders are deliberately untyped. Their schemas belong to the phases that
build them; inventing fields now would only guarantee a rewrite. They compose normally in
the meantime, just without static validation.

## Overrides

```bash
tinyearth-config run.name=ablation seed.value=7   # change values
tinyearth-config +experiment=smoke                # add a config group
tinyearth-config --multirun seed.value=1,2,3      # sweep
```

## Experiments

A file in `configs/experiment/` is a complete, named experiment. It carries the
`# @package _global_` header so its keys apply at the config root rather than nested under
`experiment.`:

```yaml
# @package _global_
run:
  name: phase1
  group: smoke
seed:
  value: 1234
```

Every experiment config must document four things in its header comment:

- **Objective** — the question the run answers
- **Configuration** — what it changes relative to defaults
- **Expected output** — what artefacts appear, and roughly what values
- **Interpretation** — how to read the result, and what it does *not* show

`configs/experiment/smoke.yaml` is the template.

## Output layout

Runs write to `outputs/<run.group>/<run.name>/`, so a sweep's runs sit adjacent on disk
and glob cleanly. Hydra's own job metadata goes to `outputs/.hydra/` and is kept only for
its override record.

```
outputs/smoke/phase1/
├── resolved_config.yaml   # fully resolved, interpolations expanded
└── run.log
```

Note `hydra.job.chdir: false` — TinyEarth manages its own run directories, keyed on
experiment *identity* rather than wall-clock time, so that re-running an experiment
overwrites its own outputs rather than accumulating timestamped copies.

## Config fingerprints

`tinyearth.config.config_fingerprint(cfg)` returns a short hash of the resolved config
with `run.*` **excluded**. Two runs differing only in name, notes or tags share a
fingerprint, which is what makes it useful for catching accidental duplicate experiments
in a sweep.

```
config fingerprint: a361b176
```

## Programmatic access

```python
from hydra import compose, initialize_config_dir

from tinyearth.config import register_configs, resolve_paths, to_dataclass
from tinyearth.utils.paths import configs_dir

register_configs()
with initialize_config_dir(version_base=None, config_dir=str(configs_dir())):
    cfg = compose(config_name="config", overrides=["seed.value=7"])

typed = to_dataclass(cfg)     # -> TinyEarthConfig, with attribute completion
paths = resolve_paths(cfg)    # -> absolute ResolvedPaths
```

## Adding a config group

1. Add the dataclass to `src/tinyearth/config/schema.py` and reference it from
   `TinyEarthConfig`.
2. Create `configs/<group>/<option>.yaml`.
3. Add `- <group>: <default-option>` to the `defaults` list in `configs/config.yaml`.
4. Add a composition test to `tests/test_config.py`.
