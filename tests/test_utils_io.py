"""Tests for :mod:`traderx.utils.io` helpers."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

from traderx.utils.io import create_project_archive


def test_create_project_archive_contains_expected_files(tmp_path):
    archive_path = create_project_archive(tmp_path / "project_export.zip")

    assert archive_path.exists()

    with zipfile.ZipFile(archive_path, "r") as archive:
        entries = set(archive.namelist())

    # README and requirements should always be present at the project root.
    assert "README.md" in entries
    assert "requirements.txt" in entries

    # Some representative source file should also be present.
    assert any(entry.startswith("traderx/") for entry in entries)


def test_create_project_archive_uses_default_destination(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()

    (project_root / "README.md").write_text("hello", encoding="utf-8")
    (project_root / "requirements.txt").write_text("deps", encoding="utf-8")
    (project_root / "module.py").write_text("print('hi')\n", encoding="utf-8")

    archive_path = create_project_archive(project_root=project_root)
    try:
        assert archive_path.exists()
        assert archive_path.parent == project_root / "exports"
    finally:
        archive_path.unlink(missing_ok=True)


def test_cli_wrapper_creates_archive(tmp_path):
    project_root = tmp_path / "cli-project"
    project_root.mkdir()

    (project_root / "README.md").write_text("ciao", encoding="utf-8")
    (project_root / "requirements.txt").write_text("pandas", encoding="utf-8")
    (project_root / "data.txt").write_text("42", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "traderx.utils.io",
            "--project-root",
            str(project_root),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = result.stdout.strip().splitlines()
    assert stdout, "CLI should print the created archive path"
    last_line = stdout[-1]
    _, _, path_str = last_line.partition(" at ")
    archive_path = Path(path_str.strip())

    try:
        assert archive_path.exists()
    finally:
        archive_path.unlink(missing_ok=True)
