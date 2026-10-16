"""Small I/O helpers: JSON with numpy support, stable hashing, git provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def dump_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, cls=_NumpyEncoder, sort_keys=False) + "\n")


def to_json_line(obj: Any) -> str:
    return json.dumps(obj, cls=_NumpyEncoder)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def stable_hash(obj: Any, n: int = 12) -> str:
    """Order-independent short hash of a JSON-serialisable object (used as cache keys)."""
    blob = json.dumps(obj, sort_keys=True, cls=_NumpyEncoder).encode()
    return hashlib.sha256(blob).hexdigest()[:n]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
