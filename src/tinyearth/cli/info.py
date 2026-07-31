"""``tinyearth-info``: report the installed environment and hardware.

Efficiency claims are only interpretable against the hardware that produced
them, so this report is the first thing to paste into an issue or attach to a
result table.
"""

from __future__ import annotations

from tinyearth import __version__
from tinyearth.utils.device import describe_device, resolve_device
from tinyearth.utils.paths import ROOT_ENV_VAR, project_root

__all__ = ["main", "render_report"]


def render_report(device_spec: str = "auto") -> str:
    """Build the environment report.

    Args:
        device_spec: Device specification to describe.

    Returns:
        A formatted multi-line report.
    """
    info = describe_device(resolve_device(device_spec))
    try:
        root = str(project_root())
    except RuntimeError as exc:  # pragma: no cover - only in exotic installs
        root = f"<unresolved: {exc}>"

    lines = [
        f"TinyEarth {__version__}",
        "-" * 52,
        f"{'project root':<18} {root}",
        f"{'root env var':<18} {ROOT_ENV_VAR}",
    ]
    lines.extend(f"{key.replace('_', ' '):<18} {value}" for key, value in info.as_dict().items())
    return "\n".join(lines)


def main() -> None:
    """Console-script entry point for ``tinyearth-info``."""
    print(render_report())  # noqa: T201 - this command's entire purpose is stdout


if __name__ == "__main__":
    main()
