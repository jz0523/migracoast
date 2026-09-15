"""Crash-safe file writes for the data fetchers.

Fetchers skip a download when the destination already exists. Writing
straight to that path would let a killed process leave a truncated file
that every later run silently reuses. Writing to a sibling .part file and
renaming it into place means the destination only ever holds a complete
file. Path.replace is used rather than rename because it overwrites an
existing target on Windows too.
"""

from pathlib import Path


def _part_path(destination: Path) -> Path:
    return destination.with_name(destination.name + ".part")


def atomic_write_bytes(destination: Path, data: bytes) -> Path:
    part = _part_path(destination)
    part.write_bytes(data)
    part.replace(destination)
    return destination


def atomic_write_csv(frame, destination: Path) -> Path:
    part = _part_path(destination)
    frame.to_csv(part, index=False)
    part.replace(destination)
    return destination
