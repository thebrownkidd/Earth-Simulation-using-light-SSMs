# Notebooks

Exploratory analysis and visualisation.

```bash
pip install -e ".[notebooks]"
```

| Notebook | Contents |
| --- | --- |
| `01_dataset_visualisation.ipynb` | Sample structure, temporal sequences, spectral channels, cloud masking, fill policies, channel statistics, the train/val partition, and a reproducibility check. |

Runs on synthetic data by default — no download required.

Conventions:

- Notebooks **explore and visualise**; they do not define reusable logic. Anything worth
  keeping moves into `src/tinyearth/` where it can be tested.
- **Clear outputs before committing.** `tests/test_notebooks.py` enforces this, and also
  executes every notebook so a broken example fails the suite.
- Load configs through `tinyearth.config` rather than hardcoding values, so a notebook and
  a run agree on what was configured.
