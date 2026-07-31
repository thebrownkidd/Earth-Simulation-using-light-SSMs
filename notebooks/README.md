# Notebooks

Exploratory analysis and visualisation.

Install the extra first:

```bash
pip install -e ".[notebooks]"
```

Conventions:

- Notebooks **explore and visualise**; they do not define reusable logic. Anything worth
  keeping moves into `src/tinyearth/` where it can be tested.
- Clear outputs before committing — they bloat diffs and leak absolute paths.
- Load configs through `tinyearth.config` rather than hardcoding values, so a notebook
  and a run agree on what was configured.

Phase 2 adds a dataset visualisation notebook.
