"""
Cross-species (leave-one-species-out) evaluation.

Configure the run by editing the CONFIG block below, then execute:

    python loso_evaluation.py

CONFIG["split"] selects the evaluation protocol:

    "loso"     : leave-one-species-out: trains on all but the held-out species
                 and evaluates on the held-out one, iterating over
                 CONFIG["holdout_species"].

It computes F1, accuracy, precision, recall, PR-AUC and ROC-AUC for every seed
and writes, under CONFIG["output_dir"]:
    - results_detailed.csv    (one row per seed)
    - results_aggregated.csv  (mean +/- std across seeds)
    - loso_per_species.csv    (loso only: mean over reliable antibiotics,
                               per held-out species x dataset)
    - loso_summary.csv         (loso only: NDARO vs the best combined
                               representation per held-out species)

Inputs: the NDARO CSV (CONFIG["ndaro_csv"]) is read and converted to a sparse
matrix in memory, so no intermediate files are produced. The BAKTA `.npz`
matrices (CONFIG["bakta_dir"]) are available on BioStudies (accession
S-BSST2698).

Two architectures are used, mirroring the main pipeline:
    - Single source (one feature block)  -> BaselineMLP
    - Combined source (NDARO + BAKTA)    -> TwoTowerMLP

Training uses early stopping on a held-out validation split (patience 10,
max 200 epochs), identical to the main pipeline, so results are reproducible.
"""
from __future__ import annotations
import math, pickle, time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    average_precision_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit


# ===========================================================================
#  CONFIGURATION
# ===========================================================================
CONFIG = {
    # -- Inputs -------------------------------------------------------------
    # NDARO CSV (columns: assembly, organism, g_* features, a_* labels).
    "ndaro_csv": "data/processed.csv",
    # Directory holding the BAKTA sparse matrices and their companion pickles:
    #   bakta50.npz / bakta50_columns.pkl / bakta50_assemblies.pkl  (and _amr, 90 variants)
    "bakta_dir": "data",
    # Optional (loso): per-antibiotic AMRFinderPlus CSV from amrfinderplus_baseline.py.
    # When set, its F1/ROC-AUC are averaged over each species' reliable antibiotics
    # and added as a rule-based reference column to the per-species summary.
    # None -> the summary contains only the trained models.
    "amrfinder_csv": None,            # e.g. "results/amrfinderplus/amrfinder_baseline_per_antibiotic.csv"

    # -- Output -------------------------------------------------------------
    "output_dir": "results/loso",

    # -- Experiment ---------------------------------------------------------
    # Datasets to evaluate (see DATASET_DEFS for the available names); the
    # curated NDARO set plus its combinations with BAKTA features.
    "datasets": ["NDARO", "NDARO+BAKTA50", "NDARO+BAKTA50_AMR",
                 "NDARO+BAKTA90", "NDARO+BAKTA90_AMR"],
    "split": "loso",
    # (loso) species held out one at a time; must match the `organism` strings.
    "holdout_species": ["Salmonella enterica", "E.coli and Shigella",
                        "Campylobacter jejuni", "Acinetobacter baumannii"],
    "antibiotics": None,             # None = all antibiotics with >= min_samples
    "min_samples": 50,
    "seeds": 5,                      # independent seeds per (dataset, antibiotic, holdout)
    "seed_offset": 0,
    # (loso) a held-out fold is "reliable" for the per-species summary when its
    # test set has at least this many minority-class isolates; unreliable folds
    # give near-random F1 and are excluded from the per-species aggregation.
    "min_minority_test": 5,

    # -- Model / training (matches the main pipeline) -----------------------
    "max_epochs": 200,
    "patience": 10,                  # early-stopping patience (monitors validation F1)
    "batch_size": 512,
    "device": "cuda",                # "cuda" or "cpu"
    "tag": "loso",
}
# ===========================================================================


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_ndaro_csv(csv_path: Path):
    """Read the NDARO CSV and build the sparse representation in memory.

    Returns the same tuple shape as the BAKTA loader expects downstream:
    (features, feature_names, assemblies, organism, labels, label_names),
    where labels is an int8 matrix with -1=missing, 0=S, 1=R.
    """
    csv_path = str(csv_path)
    header = pd.read_csv(csv_path, nrows=0).columns
    g_cols = [c for c in header if c.startswith("g_")]
    a_cols = [c for c in header if c.startswith("a_")]
    keep = ["assembly", "organism"] + g_cols + a_cols
    dtype_map = {c: "Int8" for c in g_cols}

    # Read in chunks and keep the first row per assembly (the CSV repeats rows).
    seen: set[str] = set()
    parts = []
    for ch in pd.read_csv(csv_path, usecols=keep, chunksize=100_000, dtype=dtype_map):
        mask = ~ch["assembly"].isin(seen)
        ch = ch[mask].drop_duplicates(subset=["assembly"])
        seen.update(ch["assembly"].tolist())
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)

    assemblies = df["assembly"].tolist()
    organism = df["organism"].tolist()
    feats = sp.csr_matrix(df[g_cols].fillna(0).to_numpy(dtype=np.int8).astype(np.float32))
    labels = np.full((len(df), len(a_cols)), -1, dtype=np.int8)
    for j, c in enumerate(a_cols):
        col = df[c].astype("string")
        labels[col == "S", j] = 0
        labels[col == "R", j] = 1
    return feats, g_cols, assemblies, organism, labels, a_cols


def load_bakta_sparse(prefix: Path):
    mat = sp.load_npz(str(prefix) + ".npz")
    columns = pickle.load(open(str(prefix) + "_columns.pkl", "rb"))
    assemblies = pickle.load(open(str(prefix) + "_assemblies.pkl", "rb"))
    ab_idx = [i for i, c in enumerate(columns) if str(c).startswith("a_")]
    feat_idx = [i for i in range(len(columns)) if i not in set(ab_idx)]
    feats = mat[:, feat_idx].tocsr()
    labels_dense = mat[:, ab_idx].toarray()
    ab_names = [columns[i] for i in ab_idx]
    feat_names = [columns[i] for i in feat_idx]
    return feats, feat_names, assemblies, labels_dense, ab_names


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class BaselineMLP(nn.Module):
    def __init__(self, in_dim, hidden=(512, 256), dropout=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TwoTowerMLP(nn.Module):
    def __init__(self, in_g, in_u, hidden=(512, 256), dropout=0.2):
        super().__init__()
        def tower(d):
            layers = []
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
                d = h
            return nn.Sequential(*layers), d
        self.tow_g, out_g = tower(in_g)
        self.tow_u, out_u = tower(in_u)
        self.head = nn.Linear(out_g + out_u, 1)

    def forward(self, xg, xu):
        z = torch.cat([self.tow_g(xg), self.tow_u(xu)], dim=-1)
        return self.head(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Train + eval
# ---------------------------------------------------------------------------
def build_xy(features_blocks, labels, organism, antibiotic_idx, antibiotic_filter=None):
    y_raw = labels[:, antibiotic_idx]
    mask = y_raw >= 0
    y = y_raw[mask].astype(np.int64)
    if organism is None:
        org = np.array(["Unknown"] * int(mask.sum()))
    else:
        org = np.asarray(organism)[mask]
    feat_block = [block[mask].toarray().astype(np.float32) if sp.issparse(block) else block[mask].astype(np.float32) for block in features_blocks]
    return feat_block, y, org


def split_random(y, org, test_size=0.2, val_fraction=0.1, seed=0):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y))
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(sss.split(idx, y))
    except ValueError:
        rng.shuffle(idx)
        n_test = int(len(y) * test_size)
        test_idx, train_idx = idx[:n_test], idx[n_test:]
    # carve val from train
    try:
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx2, val_idx = next(sss2.split(train_idx, y[train_idx]))
        train_idx, val_idx = train_idx[train_idx2], train_idx[val_idx]
    except ValueError:
        rng.shuffle(train_idx)
        n_val = max(1, int(len(train_idx) * val_fraction))
        val_idx, train_idx = train_idx[:n_val], train_idx[n_val:]
    return train_idx, val_idx, test_idx


def split_by_species(y, org, test_size=0.2, val_fraction=0.1, seed=0):
    """Group-stratified split: each species (organism) is wholly train, val or test.
    Aim for ~test_size of samples in the test set. Species are assigned greedily
    largest-first to whichever bucket is most under-allocated, while preserving
    that each split has at least one positive and one negative sample.
    """
    rng = np.random.RandomState(seed)
    species, counts = np.unique(org, return_counts=True)
    # Shuffle order of equally-sized species so seeds give different splits
    order = np.argsort(-counts + rng.uniform(-0.49, 0.49, size=counts.shape))
    species = species[order]
    counts = counts[order]
    n_total = len(y)
    target_test = test_size * n_total
    target_val = val_fraction * (1 - test_size) * n_total
    target_train = n_total - target_test - target_val
    bucket = {"train": [], "val": [], "test": []}
    sizes = {"train": 0.0, "val": 0.0, "test": 0.0}
    targets = {"train": target_train, "val": target_val, "test": target_test}
    for sp_name, sp_n in zip(species, counts):
        # Choose bucket with most remaining capacity (target - current)
        b = max(sizes.keys(), key=lambda k: (targets[k] - sizes[k]))
        bucket[b].append(sp_name)
        sizes[b] += sp_n
    # Convert species sets to indices
    def idx(sp_set):
        s = set(sp_set)
        return np.where(np.isin(org, list(s)))[0]
    train_idx = idx(bucket["train"])
    val_idx = idx(bucket["val"])
    test_idx = idx(bucket["test"])
    # If any split has no positives or no negatives, fall back to random
    for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        if len(split_idx) == 0 or len(set(y[split_idx])) < 2:
            return split_random(y, org, test_size, val_fraction, seed)
    return train_idx, val_idx, test_idx


def split_loso(y, org, held_out_species, val_fraction=0.1, seed=0):
    """Leave-one-species-out: held-out species -> test; remainder split train/val."""
    org = np.asarray(org)
    test_mask = org == held_out_species
    test_idx = np.where(test_mask)[0]
    train_pool = np.where(~test_mask)[0]
    if len(test_idx) == 0 or len(set(y[test_idx])) < 2:
        return None
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx2, val_idx = next(sss.split(train_pool, y[train_pool]))
        train_idx = train_pool[train_idx2]
        val_idx = train_pool[val_idx]
    except ValueError:
        rng = np.random.RandomState(seed)
        rng.shuffle(train_pool)
        n_val = max(1, int(len(train_pool) * val_fraction))
        val_idx, train_idx = train_pool[:n_val], train_pool[n_val:]
    if len(set(y[train_idx])) < 2 or len(set(y[val_idx])) < 2:
        return None
    return train_idx, val_idx, test_idx


def train_one(X_blocks, y, train_idx, val_idx, test_idx, *,
              hidden=(512, 256), dropout=0.2, lr=1e-3, weight_decay=0.0,
              batch_size=512, max_epochs=200, patience=10, device="cuda"):
    multi = len(X_blocks) > 1
    if multi:
        Xg = torch.from_numpy(X_blocks[0])
        Xu = torch.from_numpy(X_blocks[1])
        model = TwoTowerMLP(Xg.shape[1], Xu.shape[1], hidden, dropout).to(device)
    else:
        X = torch.from_numpy(X_blocks[0])
        model = BaselineMLP(X.shape[1], hidden, dropout).to(device)
    y_t = torch.from_numpy(y).float()

    # Per-sample class weights inversely proportional to class frequency
    n_pos = max(1, int((y[train_idx] == 1).sum()))
    n_neg = max(1, int((y[train_idx] == 0).sum()))
    w_pos = 1.0 / n_pos
    w_neg = 1.0 / n_neg
    s = w_pos + w_neg
    w_pos /= s; w_neg /= s
    sample_w = np.where(y == 1, w_pos, w_neg).astype(np.float32)
    w_t = torch.from_numpy(sample_w)

    def make_loader(idx, shuffle):
        idx_t = torch.from_numpy(idx.astype(np.int64))
        if multi:
            ds = TensorDataset(Xg[idx_t], Xu[idx_t], y_t[idx_t], w_t[idx_t])
        else:
            ds = TensorDataset(X[idx_t], y_t[idx_t], w_t[idx_t])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    best_f1, bad = -math.inf, 0
    best_state = None
    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            if multi:
                xg, xu, yy, ww = [b.to(device, non_blocking=True) for b in batch]
                logit = model(xg, xu)
            else:
                xx, yy, ww = [b.to(device, non_blocking=True) for b in batch]
                logit = model(xx)
            loss = (bce(logit, yy) * ww).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        # Validation
        model.eval()
        with torch.no_grad():
            preds, ys = [], []
            for batch in val_loader:
                if multi:
                    xg, xu, yy, _ = [b.to(device, non_blocking=True) for b in batch]
                    logit = model(xg, xu)
                else:
                    xx, yy, _ = [b.to(device, non_blocking=True) for b in batch]
                    logit = model(xx)
                preds.append(torch.sigmoid(logit).cpu().numpy())
                ys.append(yy.cpu().numpy())
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(ys)
        f1 = f1_score(y_true, (y_pred >= 0.5).astype(int), zero_division=0)
        if f1 > best_f1 + 1e-6:
            best_f1 = f1
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on test
    model.eval()
    test_loader = make_loader(test_idx, False)
    with torch.no_grad():
        preds, ys = [], []
        for batch in test_loader:
            if multi:
                xg, xu, yy, _ = [b.to(device, non_blocking=True) for b in batch]
                logit = model(xg, xu)
            else:
                xx, yy, _ = [b.to(device, non_blocking=True) for b in batch]
                logit = model(xx)
            preds.append(torch.sigmoid(logit).cpu().numpy())
            ys.append(yy.cpu().numpy())
    y_pred_p = np.concatenate(preds)
    y_test = np.concatenate(ys)
    y_pred = (y_pred_p >= 0.5).astype(int)
    out = {
        "n_test": int(len(y_test)),
        "n_pos_test": int(y_test.sum()),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }
    if 0 < y_test.sum() < len(y_test):
        out["pr_auc"] = average_precision_score(y_test, y_pred_p)
        out["roc_auc"] = roc_auc_score(y_test, y_pred_p)
    else:
        out["pr_auc"] = float("nan")
        out["roc_auc"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
DATASET_DEFS = {
    "NDARO": {"sources": ["ndaro"]},
    "BAKTA50":     {"sources": ["bakta50"]},
    "BAKTA50_AMR": {"sources": ["bakta50_amr"]},
    "BAKTA90":     {"sources": ["bakta90"]},
    "BAKTA90_AMR": {"sources": ["bakta90_amr"]},
    "NDARO+BAKTA50":     {"sources": ["ndaro", "bakta50"]},
    "NDARO+BAKTA50_AMR": {"sources": ["ndaro", "bakta50_amr"]},
    "NDARO+BAKTA90":     {"sources": ["ndaro", "bakta90"]},
    "NDARO+BAKTA90_AMR": {"sources": ["ndaro", "bakta90_amr"]},
}


def load_source(name, ndaro_csv, bakta_dir):
    if name == "ndaro":
        feats, _, asm, org, labels, lbl = load_ndaro_csv(ndaro_csv)
        return feats, asm, org, labels, lbl
    else:
        # BAKTA: returns sparse and the assemblies aligned with NDARO via shared assembly ids
        mat_feats, _, asm, labels_dense, ab_names = load_bakta_sparse(bakta_dir / name)
        return mat_feats, asm, None, labels_dense, ab_names


def align(sources):
    """sources : list of (feats, asm, org, labels, lbl). Returns aligned blocks
    using the intersection of assemblies, plus organism (from NDARO if present)
    and the labels matrix aligned to the first source's label set."""
    # Filter out any non-string assemblies (NaN/None) before set intersection
    asm_sets = [set(a for a in s[1] if isinstance(a, str) and a) for s in sources]
    common = sorted(set.intersection(*asm_sets))
    common_idx = {a: i for i, a in enumerate(common)}
    aligned_blocks = []
    aligned_labels = None
    aligned_label_names = None
    organism_aligned = None
    for src_feats, asm, org, labels, lbl in sources:
        idx_in_src = [None] * len(common)
        for i, a in enumerate(asm):
            if a in common_idx:
                idx_in_src[common_idx[a]] = i
        row_idx = np.array(idx_in_src, dtype=np.int64)
        if sp.issparse(src_feats):
            aligned_blocks.append(src_feats[row_idx])
        else:
            aligned_blocks.append(src_feats[row_idx])
        if org is not None and organism_aligned is None:
            organism_aligned = [org[i] for i in row_idx]
        if aligned_label_names is None and lbl is not None:
            aligned_label_names = lbl
            aligned_labels = labels[row_idx]
    return aligned_blocks, common, organism_aligned, aligned_labels, aligned_label_names


# ---------------------------------------------------------------------------
# Per-held-out-species aggregation of the raw LOSO results
# ---------------------------------------------------------------------------
def summarise_loso_by_species(detailed_csv: Path, out_dir: Path, min_minority: int = 5,
                              amrfinder_csv=None) -> None:
    """Aggregate the raw per-seed LOSO results into a per-held-out-species summary.

    A (held-out species, antibiotic) fold is *reliable* when its held-out test
    set contains at least ``min_minority`` minority-class isolates; unreliable
    folds give near-random F1 and are excluded. Metrics are averaged over seeds,
    then over reliable antibiotics, per held-out species. F1 / ROC-AUC are
    reported for NDARO and the best combined representation (the NDARO+BAKTA*
    variant with the highest mean ROC-AUC).

    If ``amrfinder_csv`` is given (the per-antibiotic output of
    amrfinderplus_baseline.py), the rule-based AMRFinderPlus F1 / ROC-AUC are
    averaged over each species' reliable antibiotics and added as extra columns,
    for a like-for-like comparison on the same antibiotics.

    Writes:
      - loso_per_species.csv    (every dataset x held-out species)
      - loso_summary.csv        (compact NDARO vs best-combined view)
    """
    d = pd.read_csv(detailed_csv)
    d = d[d["split"] == "loso"]
    if d.empty or "holdout" not in d.columns or "n_pos_test" not in d.columns:
        return

    # Mean over seeds per (dataset, holdout, antibiotic). The held-out test set
    # is identical across seeds, so 'first' is exact for the fold counts.
    per_ab = (d.groupby(["dataset", "holdout", "antibiotic"], dropna=False)
                .agg(f1=("f1", "mean"), roc_auc=("roc_auc", "mean"),
                     n_test=("n_test", "first"), n_pos_test=("n_pos_test", "first"))
                .reset_index())
    minority = np.minimum(per_ab["n_pos_test"], per_ab["n_test"] - per_ab["n_pos_test"])
    per_ab["reliable"] = minority >= min_minority
    rel = per_ab[per_ab["reliable"]]
    if rel.empty:
        print("No reliable LOSO folds; skipping per-species summary.")
        return

    # Per (held-out species, dataset): mean over reliable antibiotics.
    per_sp = (rel.groupby(["holdout", "dataset"])
                 .agg(n_reliable=("antibiotic", "nunique"),
                      f1=("f1", "mean"), roc_auc=("roc_auc", "mean"))
                 .reset_index())
    per_sp.to_csv(out_dir / "loso_per_species.csv", index=False)

    # Reliable-antibiotic set per held-out species (NDARO is the reference; the
    # set is essentially dataset-independent as it depends on test composition).
    base_rel = rel[rel["dataset"] == "NDARO"] if (rel["dataset"] == "NDARO").any() else rel
    reliable_abs = base_rel.groupby("holdout")["antibiotic"].agg(set)

    # Optional rule-based AMRFinderPlus reference, averaged over the same
    # reliable antibiotics (it is deterministic, so no per-species training).
    amr = None
    if amrfinder_csv is not None:
        amr_path = Path(amrfinder_csv)
        if amr_path.exists():
            amr = pd.read_csv(amr_path).set_index("antibiotic")
        else:
            print(f"AMRFinderPlus CSV not found ({amr_path}); skipping that column.")

    # Compact view: NDARO vs best combined (by ROC-AUC) per held-out species.
    rows = []
    for sp_name, g in per_sp.groupby("holdout"):
        gi = g.set_index("dataset")
        row = {"held_out_species": sp_name}
        row["n_reliable"] = int(gi.loc["NDARO", "n_reliable"]) if "NDARO" in gi.index \
            else int(g["n_reliable"].max())
        if amr is not None:
            common = [a for a in reliable_abs.get(sp_name, set()) if a in amr.index]
            if common:
                row["AMRFinderPlus_f1"] = round(float(amr.loc[common, "f1"].mean()), 3)
                row["AMRFinderPlus_roc_auc"] = round(float(amr.loc[common, "roc_auc"].mean()), 3)
        if "NDARO" in gi.index:
            row["NDARO_f1"] = round(float(gi.loc["NDARO", "f1"]), 3)
            row["NDARO_roc_auc"] = round(float(gi.loc["NDARO", "roc_auc"]), 3)
        combined = gi[gi.index.str.startswith("NDARO+")]
        if not combined.empty:
            best = combined["roc_auc"].idxmax()
            row["best_combined"] = best
            row["best_combined_f1"] = round(float(combined.loc[best, "f1"]), 3)
            row["best_combined_roc_auc"] = round(float(combined.loc[best, "roc_auc"]), 3)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "loso_summary.csv", index=False)
    print(f"\nPer-species LOSO summary (reliable folds: test minority >= {min_minority}):")
    print(summary.to_string(index=False))


def main(cfg=CONFIG):
    ndaro_csv = Path(cfg["ndaro_csv"])
    bakta_dir = Path(cfg["bakta_dir"])
    amrfinder_csv = cfg.get("amrfinder_csv")
    out = Path(cfg["output_dir"])
    datasets = cfg["datasets"]
    antibiotics = cfg["antibiotics"]
    split = cfg["split"]
    seeds = cfg["seeds"]
    seed_offset = cfg["seed_offset"]
    holdout_species = cfg["holdout_species"]
    min_samples = cfg["min_samples"]
    min_minority = cfg["min_minority_test"]
    max_epochs = cfg["max_epochs"]
    patience = cfg["patience"]
    batch_size = cfg["batch_size"]
    device = cfg["device"]
    tag = cfg["tag"]

    unknown = [d for d in datasets if d not in DATASET_DEFS]
    if unknown:
        raise ValueError(f"Unknown dataset(s) {unknown}; choose from {list(DATASET_DEFS)}")
    if split not in ("random", "species", "loso"):
        raise ValueError(f"split must be 'random', 'species' or 'loso', got {split!r}")

    out.mkdir(parents=True, exist_ok=True)
    detailed_csv = out / "results_detailed.csv"
    agg_csv = out / "results_aggregated.csv"

    print(f"Loading sources... ndaro_csv={ndaro_csv} bakta_dir={bakta_dir}")
    cache = {}
    def get_source(name):
        if name not in cache:
            cache[name] = load_source(name, ndaro_csv, bakta_dir)
        return cache[name]

    for dname in datasets:
        sources_to_load = DATASET_DEFS[dname]["sources"]
        srcs = [get_source(s) for s in sources_to_load]
        aligned_blocks, common, organism, labels, label_names = align(srcs)
        n = len(common)
        n_feat = sum(b.shape[1] for b in aligned_blocks)
        print(f"\n=== {dname} | aligned to {n} assemblies | {n_feat} features ===")

        if antibiotics:
            target_names = [f"a_{a}" for a in antibiotics]
        else:
            target_names = label_names

        # Pick which antibiotics meet min_samples
        viable = []
        for ab_name in target_names:
            if ab_name not in label_names:
                continue
            j = label_names.index(ab_name)
            y_raw = labels[:, j]
            mask = y_raw >= 0
            y = y_raw[mask]
            if len(y) < min_samples or len(np.unique(y)) < 2:
                continue
            viable.append(ab_name)
        print(f"  {len(viable)} antibiotics meeting min_samples={min_samples}")

        if split == "loso":
            counts = pd.Series(organism).value_counts()
            held_out_list = holdout_species or counts.head(4).index.tolist()
            print(f"  LOSO species: {held_out_list}")
        else:
            held_out_list = [None]

        for held_out in held_out_list:
            for ab_name in viable:
                ab = ab_name[2:]
                j = label_names.index(ab_name)
                X_blocks_full, y_full, org_full = build_xy(aligned_blocks, labels, organism, j)
                for run_idx in range(seeds):
                    seed = seed_offset + run_idx
                    t0 = time.time()
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    if split == "random":
                        split_idx = split_random(y_full, org_full, seed=seed)
                    elif split == "species":
                        split_idx = split_by_species(y_full, org_full, seed=seed)
                    else:
                        split_idx = split_loso(y_full, org_full, held_out_species=held_out, seed=seed)
                        if split_idx is None:
                            # Skip if held-out species doesn't have both classes
                            continue
                    train_idx, val_idx, test_idx = split_idx
                    try:
                        m = train_one(X_blocks_full, y_full, train_idx, val_idx, test_idx,
                                      max_epochs=max_epochs, patience=patience,
                                      batch_size=batch_size, device=device)
                    except RuntimeError as e:
                        print(f"  {ab_name} seed {seed}: training failed: {e}")
                        continue
                    row = {
                        "dataset": dname,
                        "antibiotic": ab,
                        "split": split,
                        "holdout": held_out or "",
                        "seed": seed,
                        "n_total": int(len(y_full)),
                        "n_pos_total": int((y_full == 1).sum()),
                        "n_train": int(len(train_idx)),
                        "n_val": int(len(val_idx)),
                        "n_test": int(len(test_idx)),
                        "n_pos_test": int(m["n_pos_test"]),
                        **{k: m[k] for k in ("f1","accuracy","precision","recall","pr_auc","roc_auc")},
                        "time_s": round(time.time() - t0, 2),
                        "tag": tag,
                    }
                    df_row = pd.DataFrame([row])
                    df_row.to_csv(detailed_csv, mode="a", index=False,
                                  header=not detailed_csv.exists())
                    print(f"  [{dname}|{ab}|{split}|seed={seed}|holdout={held_out}] "
                          f"f1={m['f1']:.3f} prauc={m['pr_auc']:.3f} ({row['time_s']}s)")

    # Build aggregated CSV from detailed
    if detailed_csv.exists():
        d = pd.read_csv(detailed_csv)
        agg = (d.groupby(["dataset", "antibiotic", "split", "holdout"], dropna=False)
                 .agg(n_seeds=("seed", "count"),
                      n_total=("n_total", "first"),
                      n_pos_total=("n_pos_total", "first"),
                      f1_mean=("f1", "mean"), f1_std=("f1", "std"),
                      accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                      precision_mean=("precision", "mean"), precision_std=("precision", "std"),
                      recall_mean=("recall", "mean"), recall_std=("recall", "std"),
                      pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                      roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"))
                 .reset_index())
        agg.to_csv(agg_csv, index=False)

    # Per-held-out-species summary (reliable folds only)
    if split == "loso":
        summarise_loso_by_species(detailed_csv, out, min_minority=min_minority,
                                  amrfinder_csv=amrfinder_csv)
    print(f"\nDone. Detailed: {detailed_csv}\nAggregated: {agg_csv}")


if __name__ == "__main__":
    main()
