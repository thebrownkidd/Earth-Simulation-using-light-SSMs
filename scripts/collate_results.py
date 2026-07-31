#!/usr/bin/env python
"""Collate sweep results into a comparison table.

Every run writes a flat ``summary.json``, so gathering a sweep is a walk over
those files rather than a UI task. That is the point of the flat format.

Usage:
    python scripts/collate_results.py --group scaling
    python scripts/collate_results.py --group scaling --format csv > scaling.csv
    python scripts/collate_results.py --group scaling --sort params/total
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from tinyearth.utils.logging import get_logger, setup_logging
from tinyearth.utils.paths import outputs_dir

logger = get_logger("tinyearth.scripts.collate")

DEFAULT_COLUMNS = (
    "run",
    "params/total_millions",
    "params/backbone_fraction",
    "efficiency/gflops_per_sample",
    "efficiency/latency_ms",
    "efficiency/throughput_samples_per_s",
    "val/mae",
    "val/rmse",
    "val/psnr",
    "val/ssim",
    "val/sam",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. ``None`` uses :data:`sys.argv`.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--group",
        default=None,
        help="Run group to collate, e.g. 'scaling'. Omit to collate everything.",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=None,
        help="Output root (default: <project>/outputs).",
    )
    parser.add_argument("--format", choices=["table", "csv", "json"], default="table")
    parser.add_argument(
        "--sort",
        default="params/total_millions",
        help="Column to sort by (default: %(default)s).",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="Columns to show. Defaults to the standard quality/efficiency set.",
    )
    return parser.parse_args(argv)


def discover_runs(root: Path, group: str | None) -> list[tuple[str, dict[str, Any]]]:
    """Find and load every ``summary.json`` under ``root``.

    Args:
        root: Output root to scan.
        group: Restrict to this run group, or ``None`` for all.

    Returns:
        ``(run_label, summary)`` pairs, sorted by label. Unreadable summaries are
        skipped with a warning rather than aborting the collation -- one failed
        run in a sweep should not hide the rest.
    """
    base = root / group if group else root
    if not base.is_dir():
        logger.error("no such directory: %s", base)
        return []

    rows: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(base.rglob("summary.json")):
        # Hydra's own metadata lives under outputs/.hydra; skip it.
        if ".hydra" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("skipping %s: %s", path, error)
            continue
        label = "/".join(path.parent.relative_to(root).parts)
        rows.append((label, payload))
    return rows


def build_records(
    runs: list[tuple[str, dict[str, Any]]], columns: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Project runs onto the requested columns.

    Args:
        runs: ``(label, summary)`` pairs.
        columns: Column names; ``"run"`` is the label.

    Returns:
        One record per run.
    """
    records = []
    for label, summary in runs:
        record: dict[str, Any] = {}
        for column in columns:
            record[column] = label if column == "run" else summary.get(column)
        records.append(record)
    return records


def render_table(records: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    """Render records as a fixed-width table.

    Args:
        records: Records to render.
        columns: Column order.

    Returns:
        The rendered table.
    """
    headers = [column.split("/")[-1] for column in columns]

    def cell(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    widths = [
        max(len(header), *(len(cell(record[column])) for record in records))
        for header, column in zip(headers, columns, strict=True)
    ]
    lines = [
        "  ".join(header.rjust(width) for header, width in zip(headers, widths, strict=True)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(
            cell(record[column]).rjust(width) for column, width in zip(columns, widths, strict=True)
        )
        for record in records
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list. ``None`` uses :data:`sys.argv`.

    Returns:
        A process exit code; 1 when no runs were found.
    """
    args = parse_args(argv)
    setup_logging(rich=True, force=True)

    root = args.outputs or outputs_dir()
    runs = discover_runs(root, args.group)
    if not runs:
        logger.error(
            "no runs found under %s. Run a sweep first, e.g.\n"
            "    tinyearth-train --multirun +experiment=scaling "
            "model.backbone.size=tiny,small,base,large",
            root / (args.group or ""),
        )
        return 1

    columns = tuple(args.columns) if args.columns else DEFAULT_COLUMNS
    records = build_records(runs, columns)

    if args.sort in columns:
        records.sort(key=lambda record: (record[args.sort] is None, record[args.sort]))

    if args.format == "json":
        print(json.dumps(records, indent=2))
    elif args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(records)
    else:
        print(render_table(records, columns))

    logger.info("collated %d run(s) from %s", len(records), root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
