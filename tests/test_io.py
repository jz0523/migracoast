from pathlib import Path

import pandas as pd
import pytest

from migracoast_ml.sources._io import atomic_write_bytes, atomic_write_csv


def test_writes_bytes_to_the_destination(tmp_path):
    target = tmp_path / "archive.zip"
    atomic_write_bytes(target, b"payload")
    assert target.read_bytes() == b"payload"


def test_leaves_no_partial_file_behind(tmp_path):
    target = tmp_path / "archive.zip"
    atomic_write_bytes(target, b"payload")
    assert list(tmp_path.iterdir()) == [target]


def test_overwrites_an_existing_file(tmp_path):
    target = tmp_path / "archive.zip"
    target.write_bytes(b"old")
    atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"new"


def test_a_failed_write_never_corrupts_the_destination(tmp_path, monkeypatch):
    """The property that makes skip-if-exists safe."""
    target = tmp_path / "archive.zip"
    target.write_bytes(b"complete old file")

    def interrupted(self, other):
        raise KeyboardInterrupt("killed mid-rename")

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"complete old file"


def test_writes_a_frame_as_csv_without_index(tmp_path):
    target = tmp_path / "claims.csv"
    atomic_write_csv(pd.DataFrame({"a": [1, 2]}), target)
    assert target.read_text().splitlines() == ["a", "1", "2"]
    assert list(tmp_path.iterdir()) == [target]
