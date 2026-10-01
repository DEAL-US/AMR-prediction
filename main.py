"""
Antimicrobial resistance prediction: one binary model per antibiotic
=====================================================================

Trains and evaluates every feature representation of Table 2 with the protocol of the
paper (30 seeds, stratified 72/8/20 split, early stopping on validation F1). Setting
``organism`` restricts training and evaluation to one species (per-species analysis).

Edit ``CONFIG`` below, or override its main entries from the command line, e.g.::

    python main.py
    python main.py --organism "Campylobacter jejuni" --output-dir results/campylobacter_jejuni
    python main.py --datasets NDARO "NDARO + BAKTA50" --antibiotics ciprofloxacin --n-seeds 2

Outputs (in ``output_dir``):
    results_detailed.csv   one row per (dataset, antibiotic, seed)
    results.csv            mean and std across seeds per (dataset, antibiotic)
Interrupted runs resume where they stopped (completed seeds are skipped).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from config import ModelConfig, RunConfig
from data import DATASETS, SourceCache, build_xy, eligible_antibiotics, load_dataset
from training import METRICS, set_seed, split_random, train_one
from utils import aggregate, append_row, completed_runs, get_device, save_model_state, setup_logging

# ===========================================================================
#  CONFIGURATION - edit the values below to match your setup
# ===========================================================================
CONFIG = {
    # -- Data (BioStudies S-BSST2698) ---------------------------------------
    "ndaro_csv": "data/ndaro_baseline.csv",
    "bakta_dir": "data",                 # bakta50.npz, bakta50_amr.npz, bakta90.npz, bakta90_amr.npz
                                         # (+ their _columns.pkl / _assemblies.pkl)
    "output_dir": "results/random_split",

    # -- Experiment ---------------------------------------------------------
    "datasets": list(DATASETS),          # the nine representations of Table 2
    "organism": None,                    # e.g. "Campylobacter jejuni" for within-species training
    "antibiotics": None,                 # None = every eligible antibiotic, or a list of names
    "n_seeds": 30,
    "seed_offset": 0,                    # seeds 0..29
    "min_samples": 50,                   # eligibility: >= 50 labelled isolates
    "min_minority": 5,                   #              and >= 5 in the minority class

    # -- Model and training -------------------------------------------------
    "hidden_dims": [512, 256],
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 512,
    "max_epochs": 200,
    "patience": 10,
    "device": "auto",                    # "auto", "cuda" or "cpu"

    # -- Optional output ----------------------------------------------------
    "save_models": False,                # save the first seed's model of every (dataset, antibiotic)
}
# ===========================================================================

KEY = ["dataset", "organism", "antibiotic", "seed"]


def parse_overrides(cfg: dict) -> dict:
    p = argparse.ArgumentParser(description="Per-antibiotic AMR prediction (overrides CONFIG).")
    p.add_argument("--ndaro-csv")
    p.add_argument("--bakta-dir")
    p.add_argument("--output-dir")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS))
    p.add_argument("--organism")
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
    device = get_device(cfg["device"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv = out_dir / "results_detailed.csv"
    done = completed_runs(detailed_csv, KEY)
    organism_label = cfg["organism"] or "all"
    logger.info("Device: %s | organism: %s | output: %s", device, organism_label, out_dir)

    cache = SourceCache(cfg["ndaro_csv"], cfg["bakta_dir"])
    seeds = range(rcfg.seed_offset, rcfg.seed_offset + rcfg.n_seeds)
    for dataset in tqdm(cfg["datasets"], desc="datasets"):
        if cfg["antibiotics"] and all((dataset, organism_label, a, str(s)) in done
                                      for a in cfg["antibiotics"] for s in seeds):
            logger.info("%s: all requested runs already done; skipped", dataset)
            continue
        data = load_dataset(dataset, cache, cfg["organism"])
        eligible = eligible_antibiotics(data, rcfg.min_samples, rcfg.min_minority)
        antibiotics = eligible_antibiotics(data, rcfg.min_samples, rcfg.min_minority, cfg["antibiotics"])
        logger.info("=== %s | %d assemblies | %d features | %d eligible antibiotics, %d selected ===",
                    dataset, len(data.assemblies), sum(b.shape[1] for b in data.blocks),
                    len(eligible), len(antibiotics))

        for a_col in tqdm(antibiotics, desc=dataset, leave=False):
            antibiotic = a_col[2:]
            X_blocks, y, _ = build_xy(data, a_col)
            seed_bar = tqdm(seeds, desc=antibiotic[:25], leave=False)
            for seed in seed_bar:
                if (dataset, organism_label, antibiotic, str(seed)) in done:
                    continue
                t0 = time.time()
                set_seed(seed)
                split = split_random(y, rcfg.test_size, rcfg.val_fraction, seed)
                save = cfg["save_models"] and seed == rcfg.seed_offset
                m, state = train_one(X_blocks, y, split, mcfg, device, rcfg.threshold, return_state=save)
                if save:
                    save_model_state(out_dir, dataset, antibiotic, seed, state)
                append_row(detailed_csv, {
                    "dataset": dataset, "organism": organism_label, "antibiotic": antibiotic, "seed": seed,
                    "n_total": len(y), "n_pos_total": int(y.sum()),
                    "n_train": len(split[0]), "n_val": len(split[1]), "n_test": m["n_test"],
                    "n_pos_test": m["n_pos_test"],
                    **{k: m[k] for k in METRICS + ["tp", "tn", "fp", "fn"]},
                    "time_s": round(time.time() - t0, 2),
                })
                seed_bar.set_postfix(seed=seed, f1=f"{m['f1']:.3f}", gmean=f"{m['gmean']:.3f}")
                logger.debug("[%s | %s | seed %d] F1 %.3f  G-mean %.3f  PR-AUC %.3f (%.1fs)",
                             dataset, antibiotic, seed, m["f1"], m["gmean"], m["pr_auc"], time.time() - t0)

    if detailed_csv.exists():
        aggregate(detailed_csv, out_dir / "results.csv", ["dataset", "organism", "antibiotic"])
        logger.info("Done. Per-seed: %s | aggregated: %s", detailed_csv, out_dir / "results.csv")


if __name__ == "__main__":
    main(parse_overrides(CONFIG))
