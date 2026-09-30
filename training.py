"""
Splitting, training and evaluation of one model (one antibiotic, one seed).

Protocol (identical for every representation):
  - stratified 72 / 8 / 20 train / validation / test split (``split_random``);
  - per-sample loss weights inversely proportional to class frequency in the training part;
  - AdamW + binary cross-entropy, early stopping on validation F1 (best checkpoint restored);
  - test predictions thresholded at 0.5.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import ModelConfig
from models import BaselineMLP, TwoTowerMLP

Split = Tuple[np.ndarray, np.ndarray, np.ndarray]   # train, validation, test indices

METRICS = ["f1", "pr_auc", "gmean", "accuracy", "precision", "recall",
           "specificity", "roc_auc"]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def split_random(y: np.ndarray, test_size: float = 0.2, val_fraction: float = 0.1,
                 seed: int = 0) -> Split:
    """Stratified test split, then a stratified validation split of the remainder."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y))
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(sss.split(idx, y))
    except ValueError:              # too few minority isolates to stratify
        rng.shuffle(idx)
        n_test = int(len(y) * test_size)
        test_idx, train_idx = idx[:n_test], idx[n_test:]
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        tr, va = next(sss.split(train_idx, y[train_idx]))
        train_idx, val_idx = train_idx[tr], train_idx[va]
    except ValueError:
        rng.shuffle(train_idx)
        n_val = max(1, int(len(train_idx) * val_fraction))
        val_idx, train_idx = train_idx[:n_val], train_idx[n_val:]
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Threshold-based metrics on the resistant (positive) class, specificity, geometric-mean
    accuracy sqrt(sensitivity x specificity), and the threshold-free PR-AUC and ROC-AUC."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if tn + fp > 0 else float("nan")
    both_classes = 0 < y_true.sum() < len(y_true)
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall,
        "specificity": specificity,
        "gmean": math.sqrt(recall * specificity) if not math.isnan(specificity) else float("nan"),
        "pr_auc": average_precision_score(y_true, y_prob) if both_classes else float("nan"),
        "roc_auc": roc_auc_score(y_true, y_prob) if both_classes else float("nan"),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_test": int(len(y_true)), "n_pos_test": int(y_true.sum()),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def build_model(input_dims: Sequence[int], mcfg: ModelConfig) -> nn.Module:
    """MLP for one feature block, two-tower MLP for two."""
    if len(input_dims) == 1:
        return BaselineMLP(input_dims[0], mcfg.hidden_dims, mcfg.dropout)
    if len(input_dims) == 2:
        return TwoTowerMLP(input_dims[0], input_dims[1], mcfg.hidden_dims, mcfg.dropout)
    raise ValueError("Only one or two feature blocks are supported")


def train_one(X_blocks: List[np.ndarray], y: np.ndarray, split: Split, mcfg: ModelConfig,
              device: torch.device, threshold: float = 0.5, return_state: bool = False,
              progress: bool = True) -> Tuple[Dict[str, float], Optional[dict]]:
    """Train on ``split[0]``, early-stop on ``split[1]``, evaluate on ``split[2]``.

    The caller seeds the random generators (:func:`set_seed`) beforehand; model
    initialisation, batch shuffling and dropout then follow deterministically.
    ``progress`` shows a per-epoch bar with the training loss and validation F1.
    """
    train_idx, val_idx, test_idx = split
    X_t = [torch.from_numpy(X) for X in X_blocks]
    model = build_model([X.shape[1] for X in X_blocks], mcfg).to(device)
    y_t = torch.from_numpy(y).float()

    # Per-sample weights inversely proportional to class frequency in the training part
    n_pos = max(1, int((y[train_idx] == 1).sum()))
    n_neg = max(1, int((y[train_idx] == 0).sum()))
    w_pos, w_neg = 1.0 / n_pos, 1.0 / n_neg
    w_pos, w_neg = w_pos / (w_pos + w_neg), w_neg / (w_pos + w_neg)
    w_t = torch.from_numpy(np.where(y == 1, w_pos, w_neg).astype(np.float32))

    pin = device.type == "cuda"

    def loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        i = torch.from_numpy(idx.astype(np.int64))
        tensors = [X[i] for X in X_t] + [y_t[i], w_t[i]]
        return DataLoader(TensorDataset(*tensors), batch_size=mcfg.batch_size,
                          shuffle=shuffle, num_workers=0, pin_memory=pin)

    def predict(dl: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        probs, ys = [], []
        with torch.no_grad():
            for batch in dl:
                *xs, yy, _ = [b.to(device, non_blocking=True) for b in batch]
                probs.append(torch.sigmoid(model(*xs)).cpu().numpy())
                ys.append(yy.cpu().numpy())
        return np.concatenate(ys), np.concatenate(probs)

    train_loader = loader(train_idx, True)
    val_loader = loader(val_idx, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=mcfg.learning_rate,
                                  weight_decay=mcfg.weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    best_f1, epochs_without_improvement, best_state = -math.inf, 0, None
    epochs = tqdm(range(mcfg.max_epochs), desc="epochs", leave=False, disable=not progress)
    for _ in epochs:
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            *xs, yy, ww = [b.to(device, non_blocking=True) for b in batch]
            loss = (bce(model(*xs), yy) * ww).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yy)
        y_val, p_val = predict(val_loader)
        val_f1 = f1_score(y_val, (p_val >= threshold).astype(int), zero_division=0)
        epochs.set_postfix(loss=f"{total_loss / len(train_idx):.4f}", val_f1=f"{val_f1:.3f}")
        if val_f1 > best_f1 + mcfg.min_delta:
            best_f1, epochs_without_improvement = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= mcfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    y_test, p_test = predict(loader(test_idx, False))
    metrics = compute_metrics(y_test.astype(int), p_test, threshold)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} if return_state else None
    return metrics, state
