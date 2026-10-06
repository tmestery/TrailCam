"""Repository and data path helpers."""

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[3]


def data_path(*parts: str) -> Path:
    """Build a path under the repo ``data/`` directory."""
    return repo_root() / "data" / Path(*parts)
