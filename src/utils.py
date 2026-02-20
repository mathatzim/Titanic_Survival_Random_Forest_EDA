from __future__ import annotations
from pathlib import Path

def project_root() -> Path:
    """Return the repository root (folder that contains src/, data/, docs/)."""
    return Path(__file__).resolve().parents[1]
