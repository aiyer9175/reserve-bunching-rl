"""Run directories, config snapshots and provenance."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_config(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_dir(run_id: str) -> Path:
    d = ROOT / "results" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def save_run(d: Path, config: dict, metrics: dict) -> None:
    commit = git_commit()  # before writing: rewriting tracked results would mark it dirty
    with open(d / "config.yaml", "w") as f:
        yaml.safe_dump(config | {"git_commit": commit}, f, sort_keys=False)
    with open(d / "metrics.json", "w") as f:
        json.dump(_jsonable(metrics), f, indent=2)
