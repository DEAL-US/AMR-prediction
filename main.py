"""
Antibiotic Resistance Prediction – Baseline MLP Experiments
===========================================================

Entry-point script.  Edit the ``config`` dictionary below to configure
dataset paths, hyperparameters, and which antibiotics to evaluate.

Run with::

    python main.py
"""
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import ModelConfig, RunConfig, DatasetConfig
from data import (
    generate_dataset_configs,
    read_dataset,
    get_antibiotic_columns,
    get_feature_columns,
    prepare_xy,
)
from training import (
    train_and_eval_once,
    train_and_eval_once_two_tower,
)
from utils import (
    setup_logging,
    aggregate_metrics,
    ensure_results_dir,
    load_existing_results,
    result_row_exists,
    write_results,
    get_device,
)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_for_dataset(
    dcfg: DatasetConfig,
    mcfg: ModelConfig,
    rcfg: RunConfig,
    existing_results: pd.DataFrame,
    device: torch.device,
    ndaro_path: Path,
    sparse_datasets: Dict[str, Path],
    limit_antibiotics: Optional[List[str]] = None,
    results_csv: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """
    Run all antibiotic experiments for a single dataset configuration.

    For every antibiotic column found in the loaded data (optionally filtered
    by *limit_antibiotics*), trains ``rcfg.runs_per_antibiotic`` models with
    different random seeds and records the aggregated metrics.
    """
    logger = setup_logging()

    df = read_dataset(dcfg, ndaro_path, sparse_datasets)
    if df is None:
        return []

    # Determine feature columns ------------------------------------------
    if dcfg.is_combined:
        g_cols = [c for c in df.columns if c.startswith("g_")]
        u_cols = [c for c in df.columns if c.startswith("U_")]
        if not g_cols or not u_cols:
            logger.warning("Combined dataset missing features (g_=%d, U_=%d); skipping",
                           len(g_cols), len(u_cols))
            return []
    else:
        feature_cols = get_feature_columns(df, dcfg.feature_prefix)
        if not feature_cols:
            logger.warning("No feature columns with prefix '%s' in %s; skipping",
                           dcfg.feature_prefix, dcfg.name)
            return []

    # Determine antibiotic columns ----------------------------------------
    antibiotic_cols = get_antibiotic_columns(df, dcfg.antibiotic_prefix)
    if not antibiotic_cols:
        logger.warning("No antibiotic columns in %s; skipping", dcfg.name)
        return []

    if limit_antibiotics is not None:
        antibiotic_cols = [
            c for c in antibiotic_cols
            if c[len(dcfg.antibiotic_prefix):] in limit_antibiotics
        ]
        if not antibiotic_cols:
            logger.warning("No matching antibiotics in %s for the specified filter", dcfg.name)
            return []
        logger.info("Filtered to %d antibiotics", len(antibiotic_cols))

    # Per-antibiotic loop -------------------------------------------------
    results: List[Dict[str, object]] = []
    pbar = tqdm(antibiotic_cols, desc=f"{dcfg.name} antibiotics")
    for a_col in pbar:
        antibiotic_name = a_col[len(dcfg.antibiotic_prefix):]
        pbar.set_postfix({"ab": antibiotic_name[:20]})

        # Resume support
        if result_row_exists(existing_results, dcfg.name, antibiotic_name):
            logger.info("Skipping already computed: %s / %s", dcfg.name, antibiotic_name)
            continue

        # Build feature matrices & labels ---------------------------------
        if dcfg.is_combined:
            valid = df[a_col].isin(rcfg.label_map.keys())
            filtered = df[valid].copy()
            if filtered.empty:
                continue
            Xg = filtered[g_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            Xu = filtered[u_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            y = filtered[a_col].map(rcfg.label_map).to_numpy(dtype=np.int64)
            n_pos, n_neg, n_total = int((y == 1).sum()), int((y == 0).sum()), len(y)
        else:
            X, y, counts = prepare_xy(df, a_col, feature_cols, rcfg.label_map)
            n_total, n_pos, n_neg = counts["n_total"], counts["n_pos"], counts["n_neg"]

        if n_total == 0 or n_pos == 0 or n_neg == 0:
            logger.info("Skipping %s / %s: insufficient labels (total=%d, pos=%d, neg=%d)",
                        dcfg.name, antibiotic_name, n_total, n_pos, n_neg)
            continue

        # Multiple random runs -------------------------------------------
        metrics_runs: List[Dict[str, float]] = []
        for run_idx in tqdm(range(rcfg.runs_per_antibiotic),
                            desc=f"runs {antibiotic_name}", leave=False):
            seed = rcfg.base_seed + run_idx
            torch.manual_seed(seed)
            np.random.seed(seed)
            if dcfg.is_combined:
                metrics = train_and_eval_once_two_tower(Xg, Xu, y, mcfg, seed, device)
            else:
                metrics = train_and_eval_once(X, y, mcfg, seed, device)
            metrics_runs.append(metrics)

        agg = aggregate_metrics(metrics_runs)
        row = {
            "dataset": dcfg.name,
            "antibiotic": antibiotic_name,
            "n_samples": n_total,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "runs": rcfg.runs_per_antibiotic,
            "f1": None if math.isnan(agg["f1"]) else round(agg["f1"], 2),
            "accuracy": None if math.isnan(agg["accuracy"]) else round(agg["accuracy"], 2),
            "precision": None if math.isnan(agg["precision"]) else round(agg["precision"], 2),
            "recall": None if math.isnan(agg["recall"]) else round(agg["recall"], 2),
            "tp": agg.get("tp", 0),
            "tn": agg.get("tn", 0),
            "fp": agg.get("fp", 0),
            "fn": agg.get("fn", 0),
            "hidden_dims": json.dumps(mcfg.hidden_dims),
            "dropout": mcfg.dropout,
            "learning_rate": mcfg.learning_rate,
            "batch_size": mcfg.batch_size,
            "epochs": mcfg.epochs,
        }
        results.append(row)

        # Persist incrementally for crash resilience
        if results_csv is not None:
            existing_results = write_results(results_csv, existing_results, [row])

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ===================================================================
    #  CONFIGURATION – edit the values below to match your setup
    # ===================================================================
    config = {
        # -- Dataset paths ------------------------------------------------
        # NDARO baseline CSV (columns: assembly, organism, g_* features, a_* labels)
        "ndaro_path": "ndaro_baseline.csv",

        # Sparse BAKTA datasets – each needs a triplet of files:
        #   <path>.npz, <path>_assemblies.pkl, <path>_columns.pkl
        "sparse_datasets": {
            "BAKTA50":     "bakta50.npz",
            "BAKTA50 AMR": "bakta50_amr.npz",
            "BAKTA90":     "bakta90.npz",
            "BAKTA90 AMR": "bakta90_amr.npz",
        },

        # -- Output -------------------------------------------------------
        "output_dir": "results",        # directory where results.csv will be written

        # -- Model hyperparameters ----------------------------------------
        "hidden_dims": [512, 256],
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 512,
        "epochs": 200,

        # -- Early stopping -----------------------------------------------
        "val_fraction": 0.1,
        "use_early_stopping": True,
        "es_metric": "f1",              # one of: f1, accuracy, precision, recall
        "es_patience": 10,
        "es_min_delta": 0.0,

        # -- Experiment protocol ------------------------------------------
        "runs": 30,                     # independent seeds per antibiotic

        # -- Antibiotic filter (None = evaluate all) ----------------------
        "limit_antibiotics": None,
        # Example: uncomment the list below to restrict to specific antibiotics
        # "limit_antibiotics": [
        #     "amoxicillin-clavulanic acid", "piperacillin",
        #     "piperacillin-tazobactam", "cefotaxime", "cefepime",
        #     "gentamicin", "tobramycin", "amikacin",
        #     "trimethoprim-sulfamethoxazole", "fosfomycin",
        #     "ciprofloxacin", "ertapenem", "meropenem",
        # ],
    }

    # ===================================================================
    logger = setup_logging()
    logger.info("Config: %s", json.dumps(config, default=str))

    # Build typed configs
    ndaro_path = Path(config["ndaro_path"])
    sparse_datasets: Dict[str, Path] = {
        k: Path(v) for k, v in config["sparse_datasets"].items()
    }
    output_dir = Path(config["output_dir"])

    mcfg = ModelConfig(
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        val_fraction=config["val_fraction"],
        use_early_stopping=config["use_early_stopping"],
        es_metric=config["es_metric"],
        es_patience=config["es_patience"],
        es_min_delta=config["es_min_delta"],
    )
    rcfg = RunConfig(runs_per_antibiotic=config["runs"])

    device = get_device()

    # Results CSV (with resume support)
    results_csv = ensure_results_dir(output_dir)
    existing_df = load_existing_results(results_csv)

    # Generate the list of dataset configurations
    dataset_configs = generate_dataset_configs(ndaro_path, sparse_datasets)

    logger.info("Datasets to evaluate: %s",
                ", ".join(dc.name for dc in dataset_configs))

    # Main loop – one pass per dataset config
    for dcfg in dataset_configs:
        logger.info("=" * 60)
        logger.info("Processing dataset: %s", dcfg.name)
        logger.info("=" * 60)
        run_for_dataset(
            dcfg=dcfg,
            mcfg=mcfg,
            rcfg=rcfg,
            existing_results=existing_df,
            device=device,
            ndaro_path=ndaro_path,
            sparse_datasets=sparse_datasets,
            limit_antibiotics=config["limit_antibiotics"],
            results_csv=results_csv,
        )
        # Refresh from disk after each dataset (picks up newly written rows)
        existing_df = load_existing_results(results_csv)

    logger.info("All datasets processed! Results saved to %s", results_csv)


if __name__ == "__main__":
    main()
