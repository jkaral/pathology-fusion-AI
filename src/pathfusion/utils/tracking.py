"""Experiment tracking: Weights & Biases when available, always mirrored to local CSV.

``mode=auto`` logs online only if ``WANDB_API_KEY`` is set (keys stay in the environment,
never in the repo); otherwise tracking is a no-op apart from the local CSV, so every stage
runs offline and in tests.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Literal, cast

from pathfusion.config import TrackingCfg
from pathfusion.utils.logging import get_logger

log = get_logger("tracking")
WandbMode = Literal["online", "offline", "disabled"]


class Tracker:
    def __init__(self, cfg: TrackingCfg, name: str, group: str, config: dict[str, Any],
                 local_csv: Path | None = None, job_type: str = "train"):
        self.local_csv = local_csv
        self._rows: list[dict[str, Any]] = []
        self._run = None
        mode = cfg.mode
        if mode == "auto":
            mode = "online" if os.environ.get("WANDB_API_KEY") else "disabled"
        if mode not in ("online", "offline", "disabled"):
            raise ValueError(f"tracking.mode must be auto|online|offline|disabled, got {mode!r}")
        if mode != "disabled":
            try:
                import wandb

                self._run = wandb.init(
                    project=cfg.wandb_project, entity=cfg.wandb_entity, name=name, group=group,
                    job_type=job_type, config=config, mode=cast(WandbMode, mode), reinit="finish_previous",
                )
            except Exception as exc:  # tracking must never break a run
                log.warning("wandb disabled (%s)", exc)
                self._run = None

    @property
    def url(self) -> str | None:
        return getattr(self._run, "url", None) if self._run is not None else None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._rows.append({"step": step, **metrics})
        if self._run is not None:
            self._run.log(metrics, step=step)

    def summary(self, metrics: dict[str, Any]) -> None:
        if self._run is not None:
            for k, v in metrics.items():
                self._run.summary[k] = v

    def finish(self) -> None:
        if self.local_csv is not None and self._rows:
            self.local_csv.parent.mkdir(parents=True, exist_ok=True)
            keys: list[str] = []
            for r in self._rows:
                keys.extend(k for k in r if k not in keys)
            with open(self.local_csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(self._rows)
        if self._run is not None:
            self._run.finish()
            self._run = None


class NullTracker(Tracker):
    def __init__(self) -> None:  # noqa: D107 - intentionally no wandb
        self.local_csv = None
        self._rows = []
        self._run = None
