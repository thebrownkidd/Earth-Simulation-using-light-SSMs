#!/usr/bin/env python
"""Run the EarthNet2021 comparison end to end.

Trains all four backbones on real data at a matched ~2M parameter budget, then
leaves the run directories ready for ``evaluate_earthnet.py`` and
``visualize_forecasts.py``.

Why this runs the models one at a time
--------------------------------------
It did not, at first. A thread sweep showed PyTorch barely scaling past four
threads for models this small -- four threads reach ~95% of sixteen (s4d 3.96 vs
4.09 samples/s) -- which looked like an invitation to run four single-model
processes at four threads each and finish in a quarter of the wall clock.

Measured, that was wrong. Under four concurrent jobs a ConvLSTM epoch took
1188 s against a 265 s solo estimate: **4.5x slower each**, so the aggregate was
*below* sequential throughput. The thread sweep had measured one process against
an idle machine, which says nothing about four processes competing for memory
bandwidth -- and these convolutional workloads are bandwidth-bound on a mobile
CPU, not thread-bound. Threads were never the scarce resource.

So ``--jobs`` defaults to 1. Raising it is supported and may pay off on a
machine with more memory bandwidth, but measure the aggregate before trusting
it: per-process throughput under contention is the number that matters, and it
is not predictable from a single-process sweep.

The long pole is Mamba, at roughly one sixth of S4D's training throughput.
Expect it to dominate the total.

Usage:
    python scripts/download_earthnet2021.py --splits train --max-tarballs 16
    python scripts/run_earthnet_study.py
    python scripts/run_earthnet_study.py --backbones s4d convlstm --epochs 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from tinyearth.utils.logging import get_logger, setup_logging
from tinyearth.utils.paths import project_root

logger = get_logger("tinyearth.scripts.study")

BACKBONES = ("s4d", "mamba", "convlstm", "transformer")

THREADS_PER_JOB = 8
"""Threads each training process is pinned to.

Throughput is nearly flat in thread count for these model sizes (s4d 3.96
samples/s at four threads, 4.09 at sixteen), so this is not a sensitive knob
when jobs run one at a time. It is pinned rather than left to default so that
raising ``--jobs`` does not silently oversubscribe the machine.
"""


@dataclass(frozen=True)
class RunOutcome:
    """The result of one training process.

    Attributes:
        backbone: Which backbone was trained.
        returncode: Process exit code.
        seconds: Wall-clock duration.
        log: Path to the captured output.
    """

    backbone: str
    returncode: int
    seconds: float
    log: Path

    @property
    def ok(self) -> bool:
        """Whether the run succeeded."""
        return self.returncode == 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=list(BACKBONES),
        choices=list(BACKBONES),
        help="Backbones to train (default: all four).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Concurrent training processes (default: %(default)s). Concurrency measured "
            "slower than sequential on the reference CPU -- these workloads are "
            "memory-bandwidth bound. Raise it only after measuring aggregate throughput."
        ),
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    parser.add_argument("--experiment", default="earthnet", help="Experiment config to use.")
    parser.add_argument("--outputs", type=Path, default=None, help="Run output root.")
    parser.add_argument(
        "--threads",
        type=int,
        default=THREADS_PER_JOB,
        help="Torch threads per process (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the commands without running them."
    )
    return parser.parse_args(argv)


def check_dataset(root: Path) -> bool:
    """Verify the dataset is present before committing hours to a run.

    Args:
        root: Dataset root.

    Returns:
        Whether cubes were found.
    """
    train = root / "train"
    if not train.is_dir():
        logger.error("No training cubes at %s.", train)
        logger.error("Download some first:")
        logger.error("    python scripts/download_earthnet2021.py --splits train --max-tarballs 16")
        return False

    cubes = sum(1 for _ in train.rglob("*.npz"))
    if cubes == 0:
        logger.error("%s exists but holds no .npz cubes.", train)
        return False

    logger.info("dataset: %d cubes under %s", cubes, train)
    if cubes < 200:
        logger.warning(
            "Only %d cubes. The comparison will run, but a 2M-parameter model on so few "
            "samples will overfit and the quality numbers should not be reported.",
            cubes,
        )
    return True


def build_command(backbone: str, args: argparse.Namespace, outputs: Path) -> list[str]:
    """Assemble the training command for one backbone."""
    command = [
        sys.executable,
        "-m",
        "tinyearth.cli.train",
        f"+experiment={args.experiment}",
        f"model={backbone}",
        "logging.rich=false",
        f"paths.outputs={outputs.as_posix()}",
    ]
    if args.epochs is not None:
        command.append(f"training.epochs={args.epochs}")
    return command


def train_one(backbone: str, args: argparse.Namespace, outputs: Path) -> RunOutcome:
    """Train one backbone in its own process.

    Output is captured to a per-backbone log rather than interleaved onto the
    console, which would be unreadable with four concurrent runs.

    Args:
        backbone: Backbone to train.
        args: Parsed arguments.
        outputs: Run output root.

    Returns:
        The outcome.
    """
    log_path = outputs / f"{backbone}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    environment = dict(os.environ)
    # Pin the thread count in the child. Without this each of the four
    # processes would spin up sixteen threads and spend its time contending.
    environment["OMP_NUM_THREADS"] = str(args.threads)
    environment["MKL_NUM_THREADS"] = str(args.threads)

    command = build_command(backbone, args, outputs)
    logger.info("[%s] starting (%d threads)", backbone, args.threads)
    start = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=project_root(),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    elapsed = time.perf_counter() - start
    outcome = RunOutcome(backbone, process.returncode, elapsed, log_path)
    if outcome.ok:
        logger.info("[%s] done in %s", backbone, _duration(elapsed))
    else:
        logger.error("[%s] FAILED (exit %d) -- see %s", backbone, process.returncode, log_path)
    return outcome


def _duration(seconds: float) -> str:
    """Format a duration as hours and minutes."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes = remainder // 60
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{int(seconds) % 60:02d}s"


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list. ``None`` uses :data:`sys.argv`.

    Returns:
        A process exit code.
    """
    args = parse_args(argv)
    setup_logging(rich=True, force=True)

    outputs = args.outputs or project_root() / "outputs"
    if not check_dataset(project_root() / "data" / "earthnet2021"):
        return 1

    jobs = max(args.jobs, 1)
    logger.info(
        "training %s | %d concurrent job(s) x %d threads",
        ", ".join(args.backbones),
        jobs,
        args.threads,
    )

    if args.dry_run:
        for backbone in args.backbones:
            logger.info("would run: %s", " ".join(build_command(backbone, args, outputs)))
        return 0

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        # Mamba last. It takes roughly as long as the other three combined, so
        # finishing them first means the evaluation and figure scripts can be
        # run against real checkpoints while it is still going, rather than
        # every downstream step waiting on the slowest model.
        ordered = sorted(args.backbones, key=lambda name: name == "mamba")
        outcomes = list(pool.map(lambda name: train_one(name, args, outputs), ordered))

    logger.info("--- summary (%s total) ---", _duration(time.perf_counter() - started))
    for outcome in sorted(outcomes, key=lambda item: item.backbone):
        status = "ok" if outcome.ok else f"FAILED ({outcome.returncode})"
        logger.info("  %-12s %-18s %s", outcome.backbone, _duration(outcome.seconds), status)

    failed = [outcome for outcome in outcomes if not outcome.ok]
    if failed:
        return 1

    logger.info("next:")
    logger.info("    python scripts/evaluate_earthnet.py")
    logger.info("    python scripts/visualize_forecasts.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
