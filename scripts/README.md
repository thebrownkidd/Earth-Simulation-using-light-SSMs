# Scripts

Standalone maintenance and data-preparation scripts.

Scripts here are for one-off or operational tasks. They are **not** experiment entry
points: those are console commands defined in `src/tinyearth/cli/` and registered in
`pyproject.toml`.

| Script | Purpose |
| --- | --- |
| `download_earthnet2021.py` | Wraps the official downloader; prints manual instructions if it is unavailable. |
| `compute_dataset_statistics.py` | Per-channel normalisation statistics over the training split, excluding masked pixels. |

TinyEarth does not redistribute any dataset. See `docs/datasets.md`.
