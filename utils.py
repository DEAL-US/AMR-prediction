"""
Utility helpers: logging, device selection, per-seed results I/O (with resume),
aggregation across seeds and model saving.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from training import METRICS


class _TqdmHandler(logging.Handler):
    """Writes log records above the progress bars instead of breaking them."""

    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record))


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("amr")
    if not logger.handlers:
        handler = _TqdmHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_device(preference: str = "auto") -> torch.device:
    """``"auto"`` uses CUDA when available, otherwise the CPU."""
    if preference == "auto":
        preference = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(preference)


# ---------------------------------------------------------------------------
# Per-seed results
# ---------------------------------------------------------------------------
def completed_runs(detailed_csv: Path, key_cols: List[str]) -> Set[Tuple]:
    """Keys of the runs already present in ``detailed_csv`` (used to resume)."""
    if not detailed_csv.exists():
        return set()
    done = pd.read_csv(detailed_csv, usecols=key_cols, dtype=str, keep_default_na=False)
    return set(map(tuple, done[key_cols].itertuples(index=False, name=None)))


def append_row(detailed_csv: Path, row: Dict[str, object]) -> None:
    """Append one run to ``detailed_csv`` (written after every run, so a crash loses nothing)."""
    pd.DataFrame([row]).to_csv(detailed_csv, mode="a", index=False,
                               header=not detailed_csv.exists())


def aggregate(detailed_csv: Path, out_csv: Path, group_cols: Iterable[str]) -> pd.DataFrame:
    """Mean and standard deviation across seeds of every metric."""
    d = pd.read_csv(detailed_csv, keep_default_na=False, na_values=[""])
    group_cols = list(group_cols)
    agg = {"n_seeds": ("seed", "nunique"), "n_total": ("n_total", "first"),
           "n_pos_total": ("n_pos_total", "first")}
    for m in METRICS:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, "std")
    out = d.groupby(group_cols, sort=False).agg(**agg).reset_index()
    out.to_csv(out_csv, index=False)
    return out


def save_model_state(output_dir: Path, dataset: str, antibiotic: str, seed: int, state: dict) -> Path:
    """Save a ``state_dict`` to ``<output_dir>/models/<dataset>/<antibiotic>_seed<seed>.pt``."""
    clean = lambda s: re.sub(r"[^\w\-]+", "_", s).strip("_")
    path = output_dir / "models" / clean(dataset) / f"{clean(antibiotic)}_seed{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return path
