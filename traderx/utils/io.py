"""IO helpers for safe file operations.

This module now also exposes a tiny command-line interface so the project
archive helper can be invoked straight from the shell::

    $ python -m traderx.utils.io --output exports/project.zip

The CLI simply wraps :func:`create_project_archive` and reports the location of
the generated archive.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


class AtomicWriter:
    """Context manager for atomic file writes using temporary files."""

    def __init__(self, path: str | Path, suffix: str = ".tmp") -> None:
        self.path = Path(path)
        self.tmp_path = self.path.with_suffix(self.path.suffix + suffix)

    def __enter__(self):
        self.tmp_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.tmp_path.open("w", encoding="utf-8")
        return self.handle

    def __exit__(self, exc_type, exc, tb) -> None:
        self.handle.close()
        if exc is None:
            self.tmp_path.replace(self.path)
        else:
            self.tmp_path.unlink(missing_ok=True)


def atomic_to_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to CSV atomically."""
    writer = AtomicWriter(path)
    with writer as fh:
        df.to_csv(fh, index=False)


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    """Append a JSON record to a JSONL file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


_DEFAULT_EXCLUDES = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
}


def _should_exclude(path: Path, root: Path, excludes: set[str]) -> bool:
    """Return ``True`` if the relative ``path`` should be excluded."""

    relative = path.relative_to(root)
    return any(part in excludes for part in relative.parts)


def _resolve_destination(destination: str | Path | None, root: Path) -> Path:
    """Return a resolved destination path, defaulting under ``root``.

    When ``destination`` is ``None`` a timestamped archive is placed inside an
    ``exports`` directory directly under ``root``.
    """

    if destination is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        destination = root / "exports" / f"project-{timestamp}.zip"

    archive_path = Path(destination).expanduser().resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    return archive_path


def create_project_archive(
    destination: str | Path | None = None,
    project_root: str | Path | None = None,
    excludes: Iterable[str] | None = None,
) -> Path:
    """Create a ZIP archive of the project and return its path.

    Parameters
    ----------
    destination:
        Optional path to the ZIP file to create. When omitted a file named
        ``project-<timestamp>.zip`` is written below ``<project_root>/exports``.
    project_root:
        Optional project root to archive. Defaults to the repository root.
    excludes:
        Optional iterable of directory names to exclude from the archive.

    Notes
    -----
    README and requirements files are always included by virtue of archiving the
    repository root. Callers can use this helper whenever they need a
    distributable snapshot of the project.
    """

    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Project root '{root}' does not exist")

    archive_path = _resolve_destination(destination, root)

    exclude_set = _DEFAULT_EXCLUDES.copy()
    if excludes:
        exclude_set.update(excludes)

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            if path == archive_path:
                continue
            if _should_exclude(path, root, exclude_set):
                continue
            archive.write(path, path.relative_to(root).as_posix())

    return archive_path


def _parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a ZIP export of the project")
    parser.add_argument(
        "--output",
        "-o",
        dest="destination",
        help="Destination ZIP file. Defaults to <project_root>/exports/project-<timestamp>.zip",
    )
    parser.add_argument(
        "--project-root",
        "-r",
        dest="project_root",
        help="Root directory to archive. Defaults to the TraderX repository root",
    )
    parser.add_argument(
        "--exclude",
        "-x",
        dest="excludes",
        action="append",
        default=None,
        help="Additional directory or file names to exclude (can be repeated)",
    )
    return parser.parse_args(argv)


def _run_cli(argv: Sequence[str] | None = None) -> Path:
    args = _parse_cli_args(argv)
    excludes = args.excludes if args.excludes else None
    archive_path = create_project_archive(
        destination=args.destination,
        project_root=args.project_root,
        excludes=excludes,
    )
    print(f"Project archive created at {archive_path}")
    return archive_path


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m traderx.utils.io``."""

    _run_cli(argv)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via integration test
    raise SystemExit(main())
