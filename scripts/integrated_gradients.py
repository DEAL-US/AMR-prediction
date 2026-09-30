"""
Integrated Gradients on non-AMR UniRef50 clusters (Table 4)
===========================================================

For the antibiotics where NDARO + BAKTA50 outperforms NDARO + BAKTA50 AMR (mean F1 in
``main.py``'s ``results.csv``), the gain must come from UniRef50 clusters that are not
AMR-related. For each of those antibiotics this script

  1. trains one NDARO + BAKTA50 two-tower model,
  2. computes Integrated Gradients (zero baseline) for the UniRef50 features only, keeping
     each isolate's NDARO genes fixed at their observed values (conditional IG),
  3. ranks the clusters by mean |IG| over the labelled isolates,

and then aggregates the ranking of the non-AMR clusters (those absent from BAKTA50 AMR)
across antibiotics.

The model used for attribution follows the settings of the original IG analysis, which
differ from ``main.py``: one model per antibiotic (seed 1337, scikit-learn
``train_test_split`` 80/20 and 90/10), class weights n / (2 n_class), at most 120 epochs
with early stopping on validation accuracy (patience 10). IG uses 50 interpolation steps.

Run from the repository root after ``main.py``::

    python scripts/integrated_gradients.py

Outputs (in ``output_dir``):
    ig/ig_summary_<antibiotic>.csv     mean |IG| and mean signed IG of every UniRef50 cluster
                                       (reused if present, so interrupted runs resume)
    ig_selected_antibiotics.csv        antibiotics analysed and their F1 gain
    ig_amr_vs_non_amr.csv              per antibiotic: top feature, AMR/non-AMR composition
    non_amr_ig_per_antibiotic.csv      top-k non-AMR clusters of every antibiotic
    non_amr_ig_combined_summary.csv    non-AMR clusters ranked by mean |IG| across antibiotics
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import SourceCache  # noqa: E402
from models import TwoTowerMLP  # noqa: E402
from utils import get_device, setup_logging  # noqa: E402

# ===========================================================================
#  CONFIGURATION
# ===========================================================================
CONFIG = {
    # -- Inputs -------------------------------------------------------------
    "ndaro_csv": "data/ndaro_baseline.csv",
    "bakta_dir": "data",
    "results_csv": "results/random_split/results.csv",   # main.py output (antibiotic selection)
    "antibiotics": None,             # None = NDARO + BAKTA50 better than NDARO + BAKTA50 AMR
    "delta_threshold": 0.0,          # minimum mean-F1 gain for the selection
    "output_dir": "results/integrated_gradients",
    "top_k": 100,                    # non-AMR clusters kept per antibiotic for the aggregation

    # -- Attribution model (settings of the original IG analysis) -----------
    "hidden_dims": [512, 256],
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 512,
    "max_epochs": 120,
    "patience": 10,                  # early stopping on validation accuracy
    "val_fraction": 0.1,
    "seed": 1337,

    # -- Integrated Gradients -----------------------------------------------
    "ig_steps": 50,
    "ig_batch": 8,                   # isolates per IG batch (memory: ig_batch x ig_steps rows)
    "device": "auto",
}
# ===========================================================================


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def select_antibiotics(results_csv: Path, threshold: float) -> pd.DataFrame:
    """Antibiotics where NDARO + BAKTA50 beats NDARO + BAKTA50 AMR in mean F1."""
    r = pd.read_csv(results_csv)
    r = r[r.organism == "all"]
    f1 = r.pivot_table(index="antibiotic", columns="dataset", values="f1_mean")
    delta = (f1["NDARO + BAKTA50"] - f1["NDARO + BAKTA50 AMR"]).dropna()
    return (delta[delta > threshold].sort_values(ascending=False)
            .rename("delta_f1").reset_index())


def build_combined(cache: SourceCache):
    """NDARO genes + BAKTA50 clusters on their shared assemblies (in NDARO order).
    A phenotype is taken from BAKTA50 when present there, otherwise from NDARO."""
    nd, b50 = cache.get("NDARO"), cache.get("BAKTA50")
    b_row = {}
    for i, a in enumerate(b50.assemblies):
        b_row.setdefault(a, i)
    nd_rows = np.array([i for i, a in enumerate(nd.assemblies) if a in b_row], dtype=np.int64)
    b_rows = np.array([b_row[nd.assemblies[i]] for i in nd_rows], dtype=np.int64)

    names = sorted(set(nd.label_names) | set(b50.label_names))
    labels = np.full((len(nd_rows), len(names)), -1, dtype=np.int8)
    for j, name in enumerate(names):
        from_nd = nd.labels[nd_rows, nd.label_names.index(name)] if name in nd.label_names else -1
        from_b = b50.labels[b_rows, b50.label_names.index(name)] if name in b50.label_names else -1
        labels[:, j] = np.where(np.asarray(from_b) >= 0, from_b, from_nd)
    return nd.features[nd_rows], b50.features[b_rows], labels, names, b50.feature_names


# ---------------------------------------------------------------------------
# Attribution model
# ---------------------------------------------------------------------------
def train_attribution_model(Xg: np.ndarray, Xu: np.ndarray, y: np.ndarray, cfg: dict,
                            device: torch.device) -> TwoTowerMLP:
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    strat = y if len(np.unique(y)) > 1 else None
    try:
        Xg_tr, _, Xu_tr, _, y_tr, _ = train_test_split(Xg, Xu, y, test_size=0.2, random_state=seed, stratify=strat)
    except ValueError:
        Xg_tr, _, Xu_tr, _, y_tr, _ = train_test_split(Xg, Xu, y, test_size=0.2, random_state=seed)
    idx = np.arange(len(Xg_tr))
    strat = y_tr if len(np.unique(y_tr)) > 1 else None
    try:
        tr, va = train_test_split(idx, test_size=cfg["val_fraction"], random_state=seed, stratify=strat)
    except ValueError:
        tr, va = train_test_split(idx, test_size=cfg["val_fraction"], random_state=seed)

    y_train = y_tr[tr]
    n, n_pos, n_neg = float(len(y_train)), float((y_train == 1).sum()), float((y_train == 0).sum())
    if n_pos > 0 and n_neg > 0:
        weights = np.where(y_train == 1, n / (2.0 * n_pos), n / (2.0 * n_neg)).astype(np.float32)
    else:
        weights = np.ones(len(y_train), dtype=np.float32)
    to = lambda a: torch.from_numpy(a).to(device)
    train_loader = DataLoader(TensorDataset(to(Xg_tr[tr]), to(Xu_tr[tr]), to(y_train.astype(np.float32)),
                                            to(weights)), batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(to(Xg_tr[va]), to(Xu_tr[va]), to(y_tr[va].astype(np.float32))),
                            batch_size=cfg["batch_size"], shuffle=False)

    model = TwoTowerMLP(Xg.shape[1], Xu.shape[1], cfg["hidden_dims"], cfg["dropout"]).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    def val_accuracy() -> float:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xg, xu, yb in val_loader:
                pred = (torch.sigmoid(model(xg, xu)) > 0.5).float()
                correct += int((pred == yb).sum()); total += len(yb)
        return correct / total if total else float("nan")

    best, best_state, bad = -math.inf, None, 0
    epochs = tqdm(range(cfg["max_epochs"]), desc="epochs", leave=False)
    for _ in epochs:
        model.train()
        for xg, xu, yb, wb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = (criterion(model(xg, xu), yb) * wb).mean()
            loss.backward()
            optimizer.step()
        acc = val_accuracy()
        epochs.set_postfix(val_acc=f"{acc:.3f}")
        if not math.isnan(acc) and acc > best:
            best, bad = acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def conditional_ig(model: TwoTowerMLP, Xg: np.ndarray, Xu: np.ndarray, device: torch.device,
                   steps: int, batch: int) -> np.ndarray:
    """IG of the UniRef block (zero baseline) with NDARO genes fixed; returns (n_isolates, n_clusters)."""
    n, u_dim = Xu.shape
    alphas = torch.linspace(0.0, 1.0, steps=steps, device=device).view(1, steps, 1)
    ig = np.zeros((n, u_dim), dtype=np.float32)
    for start in tqdm(range(0, n, batch), desc="IG", leave=False):
        end = min(start + batch, n)
        xg = torch.from_numpy(Xg[start:end]).to(device)
        xu = torch.from_numpy(Xu[start:end]).to(device)
        path_u = (alphas * xu.unsqueeze(1)).reshape(-1, u_dim).requires_grad_(True)
        path_g = xg.repeat_interleave(steps, dim=0)
        grads = torch.autograd.grad(model(path_g, path_u).sum(), path_u)[0]
        ig[start:end] = (xu * grads.view(end - start, steps, u_dim).mean(dim=1)).detach().cpu().numpy()
    return ig


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(summaries: List[Tuple[str, pd.DataFrame]], amr_clusters: set, top_k: int, out_dir: Path) -> None:
    stats, top_rows = [], []
    for ab, s in summaries:
        s = s.sort_values("mean_abs_ig", ascending=False).reset_index(drop=True)
        is_amr = s.cluster_id.isin(amr_clusters)
        stats.append({
            "antibiotic": ab,
            "top_cluster": s.cluster_id[0], "top_cluster_is_amr": bool(is_amr[0]),
            "non_amr_in_top20": int((~is_amr[:20]).sum()),
            "max_abs_ig_amr": s.mean_abs_ig[is_amr].max(),
            "max_abs_ig_non_amr": s.mean_abs_ig[~is_amr].max(),
        })
        top = s[~is_amr].head(top_k).copy()
        top.insert(0, "rank_in_antibiotic", range(1, len(top) + 1))
        top.insert(0, "antibiotic", ab)
        top_rows.append(top)
    stats = pd.DataFrame(stats)
    stats.to_csv(out_dir / "ig_amr_vs_non_amr.csv", index=False)
    detail = pd.concat(top_rows, ignore_index=True)
    detail.to_csv(out_dir / "non_amr_ig_per_antibiotic.csv", index=False)

    combined = (detail.groupby("cluster_id")
                .agg(mean_abs_ig=("mean_abs_ig", "mean"), n_antibiotics=("antibiotic", "nunique"),
                     antibiotics=("antibiotic", lambda x: "; ".join(sorted(set(x)))),
                     best_rank=("rank_in_antibiotic", "min"))
                .sort_values("mean_abs_ig", ascending=False).reset_index())
    combined.insert(0, "rank", range(1, len(combined) + 1))
    combined.to_csv(out_dir / "non_amr_ig_combined_summary.csv", index=False)

    print(f"\n{len(stats)} antibiotics | top feature non-AMR in {int((~stats.top_cluster_is_amr).sum())} | "
          f"mean non-AMR clusters in top 20: {stats.non_amr_in_top20.mean():.1f} | "
          f"mean max |IG| non-AMR / AMR: {stats.max_abs_ig_non_amr.mean() / stats.max_abs_ig_amr.mean():.1f}x")
    print("\nTop non-AMR clusters across antibiotics:")
    print(combined.head(15).to_string(index=False))


def parse_overrides(cfg: dict) -> dict:
    p = argparse.ArgumentParser(description="Conditional Integrated Gradients (overrides CONFIG).")
    p.add_argument("--ndaro-csv")
    p.add_argument("--bakta-dir")
    p.add_argument("--results-csv")
    p.add_argument("--antibiotics", nargs="+")
    p.add_argument("--output-dir")
    p.add_argument("--device")
    args = vars(p.parse_args())
    return {**cfg, **{k: v for k, v in args.items() if v is not None}}


def main(cfg: dict) -> None:
    logger = setup_logging()
    device = get_device(cfg["device"])
    out_dir = Path(cfg["output_dir"])
    (out_dir / "ig").mkdir(parents=True, exist_ok=True)

    if cfg["antibiotics"] is None:
        selected = select_antibiotics(Path(cfg["results_csv"]), cfg["delta_threshold"])
    else:
        selected = pd.DataFrame({"antibiotic": cfg["antibiotics"], "delta_f1": np.nan})
    selected.to_csv(out_dir / "ig_selected_antibiotics.csv", index=False)
    logger.info("%d antibiotics selected: %s", len(selected), ", ".join(selected.antibiotic))

    cache = SourceCache(cfg["ndaro_csv"], cfg["bakta_dir"])
    Xg_all, Xu_all, labels, label_names, clusters = build_combined(cache)
    amr_clusters = set(cache.get("BAKTA50 AMR").feature_names)
    logger.info("NDARO + BAKTA50: %d assemblies, %d genes, %d clusters (%d AMR-related)",
                Xg_all.shape[0], Xg_all.shape[1], Xu_all.shape[1], len(amr_clusters & set(clusters)))

    summaries = []
    for ab in tqdm(selected.antibiotic, desc="antibiotics"):
        path = out_dir / "ig" / f"ig_summary_{ab}.csv"
        if path.exists():
            summaries.append((ab, pd.read_csv(path)))
            continue
        y_raw = labels[:, label_names.index(f"a_{ab}")]
        rows = y_raw >= 0
        Xg = Xg_all[rows].toarray().astype(np.float32)
        Xu = Xu_all[rows].toarray().astype(np.float32)
        y = y_raw[rows].astype(np.int64)
        logger.info("%s: %d isolates (%d resistant)", ab, len(y), int(y.sum()))
        model = train_attribution_model(Xg, Xu, y, cfg, device)
        ig = conditional_ig(model, Xg, Xu, device, cfg["ig_steps"], cfg["ig_batch"])
        mean_abs = np.abs(ig).mean(axis=0)
        s = pd.DataFrame({"cluster_id": clusters, "mean_abs_ig": mean_abs,
                          "mean_signed_ig": ig.mean(axis=0),
                          "pct_of_total": 100 * mean_abs / mean_abs.sum() if mean_abs.sum() > 0 else 0.0})
        s = s.sort_values("mean_abs_ig", ascending=False)
        s.to_csv(path, index=False)
        summaries.append((ab, s))

    aggregate(summaries, amr_clusters, cfg["top_k"], out_dir)


if __name__ == "__main__":
    main(parse_overrides(CONFIG))
