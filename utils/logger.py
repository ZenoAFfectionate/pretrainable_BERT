"""Structured training logger.

`TrainingLogger` owns a per-run output directory under ``result/`` and records:

* ``config.json``      — full training configuration snapshot
* ``training.log``     — human-readable text log (same content as stderr)
* ``metrics.jsonl``    — one JSON line per logged event (step / epoch / test)
* ``checkpoints/``     — all epoch checkpoints + ``best_model.pt`` symlink/copy

This makes every run self-contained and easy to diff across experiments.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


def _default_run_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class TrainingLogger:
    """File + console logger for one training run.

    Parameters
    ----------
    result_root : str | Path
        The root directory under which a run subdirectory will be created.
        Typically ``result/``.
    run_name : str, optional
        Name of this run. If None, uses ``{prefix}_{timestamp}``.
    prefix : str
        Used when auto-generating ``run_name`` (e.g. "bert" / "deberta").
    level : int
        Logging level for the text logger.
    """

    def __init__(self,
                 result_root: Union[str, Path] = "result",
                 run_name: Optional[str] = None,
                 prefix: str = "run",
                 level: int = logging.INFO):
        self.result_root = Path(result_root)
        self.run_name = run_name or _default_run_name(prefix)
        self.run_dir = self.result_root / self.run_name
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.log_file = self.run_dir / "training.log"
        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.config_file = self.run_dir / "config.json"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._logger = self._configure_text_logger(level)
        # Open metrics file in append mode so re-entry (resume) does not clobber.
        self._metrics_fp = open(self.metrics_file, "a", encoding="utf-8")

        self.info("run_dir=%s", self.run_dir.resolve())

    # ------------------------------------------------------------------ setup

    def _configure_text_logger(self, level: int) -> logging.Logger:
        logger = logging.getLogger(f"training.{self.run_name}")
        logger.setLevel(level)
        # Avoid duplicate handlers if TrainingLogger is constructed twice in-process.
        logger.propagate = False
        logger.handlers.clear()

        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        fh = logging.FileHandler(self.log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        return logger

    # --------------------------------------------------------------- text log

    def info(self, msg: str, *args: Any) -> None:
        self._logger.info(msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        self._logger.warning(msg, *args)

    def error(self, msg: str, *args: Any) -> None:
        self._logger.error(msg, *args)

    # ------------------------------------------------------------- structured

    def save_config(self, config: Dict[str, Any]) -> Path:
        """Write the full run config to config.json. Also echo to text log."""
        payload = {"timestamp": datetime.now().isoformat(), **config}
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str, ensure_ascii=False)
        self.info("Saved config to %s", self.config_file)
        return self.config_file

    def log_step(self, epoch: int, step: int, **metrics: Any) -> None:
        self._write_metric({"event": "step", "epoch": epoch, "step": step, **metrics})

    def log_epoch(self, epoch: int, split: str, **metrics: Any) -> None:
        """Emit an epoch-level summary for a given split ("train" or "test")."""
        record = {"event": "epoch", "epoch": epoch, "split": split, **metrics}
        self._write_metric(record)
        # Human-readable echo
        rendered = " | ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        self.info("%s Epoch %d | %s", split, epoch, rendered)

    def _write_metric(self, record: Dict[str, Any]) -> None:
        record["_ts"] = datetime.now().isoformat(timespec="seconds")
        self._metrics_fp.write(json.dumps(record, default=str) + "\n")
        self._metrics_fp.flush()

    # -------------------------------------------------------------- lifecycle

    def checkpoint_path(self, filename: str) -> Path:
        return self.checkpoint_dir / filename

    def close(self) -> None:
        try:
            self._metrics_fp.close()
        except Exception:
            pass
        for h in list(self._logger.handlers):
            self._logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
