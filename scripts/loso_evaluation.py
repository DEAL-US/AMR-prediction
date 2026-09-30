"""
Leave-one-species-out (LOSO) evaluation
=======================================

For each held-out species, models are trained on the isolates of all remaining species
(split 90/10 into training and validation) and evaluated on every isolate of the held-out
species. Training is identical to ``main.py``.

Configure ``CONFIG`` below (or override from the command line) and run from the
repository root::

    python scripts/loso_evaluation.py

Outputs (in ``output_dir``):
    results_detailed.csv     one row per (dataset, held-out species, antibiotic, seed)
    loso_per_antibiotic.csv  mean over seeds, with the reliability flag of each fold
    loso_per_species.csv     mean over reliable antibiotics, per held-out species x dataset
    loso_summary.csv         AMRFinderPlus vs NDARO vs best combined representation
                             (highest geometric-mean accuracy) per held-out species

A fold (held-out species, antibiotic) is reliable when its test set has at least
``min_minority_test`` isolates of the minority class; the summaries use reliable folds only.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ModelConfig, RunConfig  # noqa: E402
from data import DATASETS, SourceCache, build_xy, eligible_antibiotics, load_dataset  # noqa: E402
from training import METRICS, set_seed, train_one  # noqa: E402
from utils import append_row, completed_runs, get_device, setup_logging  # noqa: E402

# ===========================================================================
#  CONFIGURATION
# ===========================================================================
CONFIG = {
    # -- Inputs -------------------------------------------------------------
    "ndaro_csv": "data/ndaro_baseline.csv",
    "bakta_dir": "data",
    # Per-antibiotic output of amrfinderplus_baseline.py (must include the held-out
    # species in its CONFIG["species"]). None -> no AMRFinderPlus column in the summary.
    "amrfinder_csv": "results/amrfinderplus/amrfinder_baseline_per_antibiotic.csv",
    "output_dir": "results/loso",

    # -- Experiment ---------------------------------------------------------
    "datasets": ["NDARO", "NDARO + BAKTA50", "NDARO + BAKTA50 AMR",
                 "NDARO + BAKTA90", "NDARO + BAKTA90 AMR"],
    "holdout_species": ["Salmonella enterica", "E.coli and Shigella",
                        "Acinetobacter baumannii", "Campylobacter jejuni"],
    "antibiotics": None,             # None = every eligible antibiotic
    "n_seeds": 5,
    "seed_offset": 0,
    "min_samples": 50,               # eligibility over all species (as in main.py)
    "min_minority": 5,
    "min_minority_test": 5,          # reliability of a held-out fold

    # -- Model and training (as in main.py) ---------------------------------
    "hidden_dims": [512, 256],
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 512,
    "max_epochs": 200,
    "patience": 10,
    "device": "auto",
}
# ===========================================================================

KEY = ["dataset", "holdout", "antibiotic", "seed"]


def split_loso(y: np.ndarray, organism: np.ndarray, held_out: str, val_fraction: float = 0.1,
               seed: int = 0):
    """Held-out species -> test; the other species are split into training and validation.
    Returns None when the held-out species or the training/validation parts lack a class."""
    test_mask = organism == held_out
    test_idx = np.where(test_mask)[0]
    pool = np.where(~test_mask)[0]
    if len(test_idx) == 0 or len(set(y[test_idx])) < 2:
        return None
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        tr, va = next(sss.split(pool, y[pool]))
        train_idx, val_idx = pool[tr], pool[va]
    except ValueError:
        rng = np.random.RandomState(seed)
        rng.shuffle(pool)
        n_val = max(1, int(len(pool) * val_fraction))
        val_idx, train_idx = pool[:n_val], pool[n_val:]
    if len(set(y[train_idx])) < 2 or len(set(y[val_idx])) < 2:
        return None
    return train_idx, val_idx, test_idx


def summarise(detailed_csv: Path, out_dir: Path, min_minority_test: int, amrfinder_csv) -> None:
    d = pd.read_csv(detailed_csv)
    per_ab = (d.groupby(["dataset", "holdout", "antibiotic"])
                .agg(n_seeds=("seed", "nunique"), n_test=("n_test", "first"),
                     n_pos_test=("n_pos_test", "first"),
                     **{m: (m, "mean") for m in METRICS})
                .reset_index())
    per_ab["minority_test"] = np.minimum(per_ab.n_pos_test, per_ab.n_test - per_ab.n_pos_test)
    per_ab["reliable"] = per_ab.minority_test >= min_minority_test
    per_ab.to_csv(out_dir / "loso_per_antibiotic.csv", index=False)

    rel = per_ab[per_ab.reliable]
    per_sp = (rel.groupby(["holdout", "dataset"])
                 .agg(n_reliable=("antibiotic", "nunique"), f1=("f1", "mean"),
                      roc_auc=("roc_auc", "mean"), gmean=("gmean", "mean"))
                 .reset_index())
    per_sp.to_csv(out_dir / "loso_per_species.csv", index=False)

    amr = None
    if amrfinder_csv is not None:
        if Path(amrfinder_csv).exists():
            amr = pd.read_csv(amrfinder_csv)
        else:
            print(f"AMRFinderPlus CSV not found ({amrfinder_csv}); summary without it.")

    rows = []
    for species, g in per_sp.groupby("holdout", sort=False):
        g = g.set_index("dataset")
        row = {"held_out_species": species}
        if amr is not None:
            # rule-based reference on the same held-out isolates and reliable antibiotics
            # (those with at least one gene mapped to the drug class, as in Table 2)
            abs_sp = sorted(rel[rel.holdout == species].antibiotic.unique())
            a = amr[(amr.organism == species) & amr.antibiotic.isin(abs_sp) & (amr.n_rule_genes > 0)]
            if a.empty:
                print(f"No AMRFinderPlus rows for {species}; add it to its CONFIG['species'].")
            else:
                row.update(AMRFinderPlus_n=len(a), AMRFinderPlus_f1=a.f1.mean(),
                           AMRFinderPlus_roc_auc=a.roc_auc.mean(), AMRFinderPlus_gmean=a.gmean.mean())
        if "NDARO" in g.index:
            row.update(NDARO_n=int(g.loc["NDARO", "n_reliable"]), NDARO_f1=g.loc["NDARO", "f1"],
                       NDARO_roc_auc=g.loc["NDARO", "roc_auc"], NDARO_gmean=g.loc["NDARO", "gmean"])
        combined = g[g.index != "NDARO"]
        if not combined.empty:
            best = combined.gmean.idxmax()
            row.update(best_combined=best, best_combined_f1=combined.loc[best, "f1"],
                       best_combined_roc_auc=combined.loc[best, "roc_auc"],
                       best_combined_gmean=combined.loc[best, "gmean"])
        rows.append(row)
    summary = pd.DataFrame(rows).round(3)
    summary.to_csv(out_dir / "loso_summary.csv", index=False)
    print(f"\nLOSO summary (reliable folds: test minority >= {min_minority_test}):")
    print(summary.to_string(index=False))


def parse_overrides(cfg: dict) -> dict:
    p = argparse.ArgumentParser(description="Leave-one-species-out evaluation (overrides CONFIG).")
    p.add_argument("--ndaro-csv")
    p.add_argument("--bakta-dir")
    p.add_argument("--amrfinder-csv")
    p.add_argument("--output-dir")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS))
    p.add_argument("--holdout-species", nargs="+")
    p.add_argument("--antibiotics", nargs="+")
    p.add_argument("--n-seeds", type=int)
    p.add_argument("--seed-offset", type=int)
    p.add_argument("--device")
    args = vars(p.parse_args())
    return {**cfg, **{k: v for k, v in args.items() if v is not None}}


def main(cfg: dict) -> None:
    logger = setup_logging()
    mcfg = ModelConfig(hidden_dims=cfg["hidden_dims"], dropout=cfg["dropout"],
                       learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
                       batch_size=cfg["batch_size"], max_epochs=cfg["max_epochs"],
                       patience=cfg["patience"])
    rcfg = RunConfig(n_seeds=cfg["n_seeds"], seed_offset=cfg["seed_offset"],
                     min_samples=cfg["min_samples"], min_minority=cfg["min_minority"])
    for name in cfg["datasets"]:
        if DATASETS[name][0] != "NDARO":
            raise ValueError(f"{name}: LOSO needs NDARO (it provides the organism of each isolate)")
    device = get_device(cfg["device"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv = out_dir / "results_detailed.csv"
    done = completed_runs(detailed_csv, KEY)

    cache = SourceCache(cfg["ndaro_csv"], cfg["bakta_dir"])
    seeds = range(rcfg.seed_offset, rcfg.seed_offset + rcfg.n_seeds)
    for dataset in tqdm(cfg["datasets"], desc="datasets"):
        data = load_dataset(dataset, cache)
        antibiotics = eligible_antibiotics(data, rcfg.min_samples, rcfg.min_minority, cfg["antibiotics"])
        logger.info("=== %s | %d assemblies | %d eligible antibiotics ===",
                    dataset, len(data.assemblies), len(antibiotics))
        for held_out in tqdm(cfg["holdout_species"], desc="held-out species", leave=False):
            for a_col in tqdm(antibiotics, desc=held_out[:25], leave=False):
                antibiotic = a_col[2:]
                X_blocks, y, organism = build_xy(data, a_col)
                seed_bar = tqdm(seeds, desc=antibiotic[:25], leave=False)
                for seed in seed_bar:
                    if (dataset, held_out, antibiotic, str(seed)) in done:
                        continue
                    t0 = time.time()
                    set_seed(seed)
                    split = split_loso(y, organism, held_out, rcfg.val_fraction, seed)
                    if split is None:
                        continue
                    m, _ = train_one(X_blocks, y, split, mcfg, device, rcfg.threshold)
                    append_row(detailed_csv, {
                        "dataset": dataset, "holdout": held_out, "antibiotic": antibiotic, "seed": seed,
                        "n_total": len(y), "n_pos_total": int(y.sum()),
                        "n_train": len(split[0]), "n_val": len(split[1]), "n_test": m["n_test"],
                        "n_pos_test": m["n_pos_test"],
                        **{k: m[k] for k in METRICS + ["tp", "tn", "fp", "fn"]},
                        "time_s": round(time.time() - t0, 2),
                    })
                    seed_bar.set_postfix(seed=seed, f1=f"{m['f1']:.3f}", roc=f"{m['roc_auc']:.3f}")
                    logger.debug("[%s | %s held out | %s | seed %d] F1 %.3f  ROC-AUC %.3f  G-mean %.3f",
                                 dataset, held_out, antibiotic, seed, m["f1"], m["roc_auc"], m["gmean"])

    if detailed_csv.exists():
        summarise(detailed_csv, out_dir, cfg["min_minority_test"], cfg["amrfinder_csv"])


if __name__ == "__main__":
    main(parse_overrides(CONFIG))
