"""Path utilities."""
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath("data", *parts)


def artifact_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath("artifacts", *parts)
