"""Console logging and an optional MLflow wrapper (no-op when disabled/not installed)."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str = "iad", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger


class Tracker:
    """Thin MLflow wrapper. Everything is a no-op unless enabled and mlflow is importable."""

    def __init__(self, enabled: bool, experiment: str, run_name: str):
        self.mlflow = None
        if enabled:
            try:
                import mlflow

                mlflow.set_experiment(experiment)
                mlflow.start_run(run_name=run_name)
                self.mlflow = mlflow
            except Exception as e:  # pragma: no cover
                get_logger().warning(f"MLflow disabled ({e})")

    def log_params(self, params: dict) -> None:
        if self.mlflow:
            self.mlflow.log_params({k: str(v)[:250] for k, v in params.items()})

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if self.mlflow:
            clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v == v}
            self.mlflow.log_metrics(clean, step=step)

    def end(self) -> None:
        if self.mlflow:
            self.mlflow.end_run()
