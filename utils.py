"""
Utility helpers: metric aggregation, results I/O, model saving, logging setup,
and device selection.
"""
import logging
import math
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(name: str = "baseline", level: int = logging.INFO) -> logging.Logger:
    """Create and configure the project-wide logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------
def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Average classification metrics across multiple runs."""
    metric_keys = ["f1", "accuracy", "precision", "recall"]
    count_keys = ["tp", "tn", "fp", "fn"]
    agg: Dict[str, float] = {}
    for k in metric_keys:
        vals = [m.get(k, math.nan) for m in metrics_list
                if not math.isnan(m.get(k, math.nan))]
        agg[k] = float(np.mean(vals)) if vals else math.nan
    for k in count_keys:
        vals = [m.get(k, 0) for m in metrics_list]
        agg[k] = int(round(float(np.mean(vals)))) if vals else 0
    return agg


# ---------------------------------------------------------------------------
# Results persistence (with resume support)
# ---------------------------------------------------------------------------
def ensure_results_dir(output_dir: Path) -> Path:
    """Create the output directory (if needed) and return the results CSV path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "results.csv"


def load_existing_results(csv_path: Path) -> pd.DataFrame:
    """Load previously saved results or return an empty DataFrame."""
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass
    return pd.DataFrame()


def result_row_exists(df: pd.DataFrame, dataset_name: str, antibiotic: str) -> bool:
    """Check whether a (dataset, antibiotic) pair has already been computed."""
    if df.empty:
        return False
    mask = (df["dataset"] == dataset_name) & (df["antibiotic"] == antibiotic)
    return bool(mask.any())


def write_results(
    csv_path: Path,
    existing_df: pd.DataFrame,
    new_rows: List[Dict[str, object]],
) -> pd.DataFrame:
    """Append *new_rows* to *existing_df*, deduplicate, and persist to CSV."""
    if not new_rows:
        return existing_df
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["dataset", "antibiotic"], keep="first")
    combined.to_csv(csv_path, index=False)
    logger = logging.getLogger("baseline")
    logger.info("Wrote results to %s (rows=%d)", csv_path, len(combined))
    return combined


# ---------------------------------------------------------------------------
# Detailed (per-run) results persistence
# ---------------------------------------------------------------------------
def write_detailed_results(
    csv_path: Path,
    new_rows: List[Dict[str, object]],
) -> None:
    """Append per-run metric rows to the detailed results CSV."""
    if not new_rows:
        return
    new_df = pd.DataFrame(new_rows)
    if csv_path.exists():
        new_df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        new_df.to_csv(csv_path, index=False)
    logger = logging.getLogger("baseline")
    logger.info("Appended %d rows to %s", len(new_rows), csv_path)


# ---------------------------------------------------------------------------
# Model saving
# ---------------------------------------------------------------------------
def _sanitize_name(name: str) -> str:
    """Make a string safe for use as a file/directory name."""
    return re.sub(r'[^\w\-]', '_', name).strip('_')


def save_model_state(
    output_dir: Path,
    dataset_name: str,
    antibiotic_name: str,
    state_dict: dict,
) -> Path:
    """Save a model's ``state_dict`` to ``<output_dir>/models/<dataset>/<antibiotic>.pt``."""
    model_dir = output_dir / "models" / _sanitize_name(dataset_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{_sanitize_name(antibiotic_name)}.pt"
    torch.save(state_dict, path)
    logger = logging.getLogger("baseline")
    logger.info("Saved model to %s", path)
    return path


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    logger = logging.getLogger("baseline")
    if torch.cuda.is_available():
        logger.info("Using CUDA")
        return torch.device("cuda")
    logger.info("Using CPU")
    return torch.device("cpu")
