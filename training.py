"""
Training loops, evaluation, data-loader construction, and single-run
train-and-evaluate pipelines for both single-tower and two-tower models.
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import ModelConfig
from models import BaselineMLP, TwoTowerMLP


# ---------------------------------------------------------------------------
# Data-loader helpers
# ---------------------------------------------------------------------------
def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    batch_size: int,
    device: torch.device,
    train_sample_weights: Optional[np.ndarray] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    X_t = torch.from_numpy(X_train).to(device)
    y_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    if train_sample_weights is not None:
        w_t = torch.from_numpy(train_sample_weights.astype(np.float32)).to(device)
        train_ds = TensorDataset(X_t, y_t, w_t)
    else:
        train_ds = TensorDataset(X_t, y_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None and y_val is not None and len(X_val) > 0:
        Xv = torch.from_numpy(X_val).to(device)
        yv = torch.from_numpy(y_val.astype(np.float32)).to(device)
        val_loader = DataLoader(TensorDataset(Xv, yv), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def make_loaders_two_tower(
    Xg_train: np.ndarray,
    Xu_train: np.ndarray,
    y_train: np.ndarray,
    Xg_val: Optional[np.ndarray],
    Xu_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    batch_size: int,
    device: torch.device,
    train_sample_weights: Optional[np.ndarray] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    Xg_t = torch.from_numpy(Xg_train).to(device)
    Xu_t = torch.from_numpy(Xu_train).to(device)
    y_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    if train_sample_weights is not None:
        w_t = torch.from_numpy(train_sample_weights.astype(np.float32)).to(device)
        train_ds = TensorDataset(Xg_t, Xu_t, y_t, w_t)
    else:
        train_ds = TensorDataset(Xg_t, Xu_t, y_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if Xg_val is not None and Xu_val is not None and y_val is not None and len(Xg_val) > 0:
        Xgv = torch.from_numpy(Xg_val).to(device)
        Xuv = torch.from_numpy(Xu_val).to(device)
        yv = torch.from_numpy(y_val.astype(np.float32)).to(device)
        val_loader = DataLoader(TensorDataset(Xgv, Xuv, yv), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Class-balance weighting
# ---------------------------------------------------------------------------
def compute_class_sample_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weights inversely proportional to class frequency."""
    n = float(len(y))
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.ones_like(y, dtype=np.float32)
    w0 = n / (2.0 * n_neg)
    w1 = n / (2.0 * n_pos)
    return np.where(y == 1, w1, w0).astype(np.float32)


# ---------------------------------------------------------------------------
# Single-tower training epoch
# ---------------------------------------------------------------------------
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for batch in loader:
        if len(batch) == 3:
            xb, yb, wb = batch
        else:
            xb, yb = batch
            wb = None

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        if wb is not None:
            loss = (criterion(logits, yb) * wb).mean()
        else:
            loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
            y_true_all.append(yb.long().cpu().numpy())
            y_pred_all.append(preds)

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([])
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    avg_loss = total_loss / max(1, len(loader.dataset))
    return avg_loss, acc


# ---------------------------------------------------------------------------
# Two-tower training epoch
# ---------------------------------------------------------------------------
def train_epoch_two_tower(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for batch in loader:
        if len(batch) == 4:
            xg, xu, yb, wb = batch
        else:
            xg, xu, yb = batch
            wb = None

        optimizer.zero_grad(set_to_none=True)
        logits = model(xg, xu)
        if wb is not None:
            loss = (criterion(logits, yb) * wb).mean()
        else:
            loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xg.size(0)

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
            y_true_all.append(yb.long().cpu().numpy())
            y_pred_all.append(preds)

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([])
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    avg_loss = total_loss / max(1, len(loader.dataset))
    return avg_loss, acc


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics from ground-truth and predicted arrays."""
    if len(y_true) == 0:
        return {
            "f1": math.nan, "accuracy": math.nan,
            "precision": math.nan, "recall": math.nan,
            "tp": 0, "tn": 0, "fp": 0, "fn": 0,
        }

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn = int(cm[0, 0]) if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        fp = int(cm[0, 1]) if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
        fn = int(cm[1, 0]) if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
        tp = int(cm[1, 1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    return {
        "f1": float(f1), "accuracy": float(acc),
        "precision": float(prec), "recall": float(rec),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate a single-tower model on a DataLoader."""
    model.eval()
    y_true_all, y_pred_all = [], []
    for xb, yb in loader:
        logits = model(xb)
        preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
        y_true_all.append(yb.long().cpu().numpy())
        y_pred_all.append(preds)
    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([])
    return _compute_metrics(y_true, y_pred)


@torch.no_grad()
def evaluate_two_tower(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate a two-tower model on a DataLoader."""
    model.eval()
    y_true_all, y_pred_all = [], []
    for batch in loader:
        xg, xu, yb = batch
        logits = model(xg, xu)
        preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
        y_true_all.append(yb.long().cpu().numpy())
        y_pred_all.append(preds)
    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([])
    return _compute_metrics(y_true, y_pred)


# ---------------------------------------------------------------------------
# Full single-run pipelines
# ---------------------------------------------------------------------------
def train_and_eval_once(
    X: np.ndarray,
    y: np.ndarray,
    mcfg: ModelConfig,
    rseed: int,
    device: torch.device,
) -> Tuple[Dict[str, float], dict]:
    """Train a BaselineMLP for one seed and return (test-set metrics, state_dict)."""
    test_size = mcfg.val_fraction  # reuse for outer split convention
    stratify = y if len(np.unique(y)) > 1 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rseed, stratify=stratify,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rseed, stratify=None,
        )

    # Validation split for early stopping
    X_tr, y_tr = X_train, y_train
    val_loader = None
    if mcfg.val_fraction and mcfg.val_fraction > 0.0 and len(X_train) > 10:
        stratify_tv = y_train if len(np.unique(y_train)) > 1 else None
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train,
                test_size=mcfg.val_fraction, random_state=rseed, stratify=stratify_tv,
            )
        except ValueError:
            X_tr, y_tr = X_train, y_train
            X_val, y_val = None, None

    # Loaders
    train_weights = compute_class_sample_weights(y_tr)
    train_loader, _ = make_loaders(X_tr, y_tr, None, None, mcfg.batch_size, device,
                                   train_sample_weights=train_weights)
    if mcfg.val_fraction and "X_val" in dir() and X_val is not None:
        _, val_loader = make_loaders(X_tr[:1], y_tr[:1], X_val, y_val, mcfg.batch_size, device)
    test_loader, _ = make_loaders(X_test, y_test, None, None, mcfg.batch_size, device)

    # Model
    model = BaselineMLP(X.shape[1], mcfg.hidden_dims, mcfg.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=mcfg.learning_rate,
                                  weight_decay=mcfg.weight_decay)

    best_score = float("-inf")
    best_state = None
    epochs_no_improve = 0
    monitor = mcfg.es_metric

    epoch_iter = tqdm(range(mcfg.epochs), desc="epochs", leave=False)
    for _ in epoch_iter:
        loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        postfix = {"loss": f"{loss:.4f}", "train_acc": f"{train_acc:.3f}"}
        if val_loader is not None:
            val_m = evaluate(model, val_loader, device)
            cur = val_m.get(monitor, float("nan"))
            postfix["val_" + monitor] = f"{cur:.3f}" if not math.isnan(cur) else "nan"
            improved = (not math.isnan(cur)) and (cur > best_score + mcfg.es_min_delta)
            if improved:
                best_score = cur
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if mcfg.use_early_stopping and epochs_no_improve >= mcfg.es_patience:
                epoch_iter.set_postfix(postfix)
                break
        epoch_iter.set_postfix(postfix)

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate(model, test_loader, device)
    # Return metrics + a CPU copy of the final state_dict
    state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return metrics, state


def train_and_eval_once_two_tower(
    Xg: np.ndarray,
    Xu: np.ndarray,
    y: np.ndarray,
    mcfg: ModelConfig,
    rseed: int,
    device: torch.device,
) -> Tuple[Dict[str, float], dict]:
    """Train a TwoTowerMLP for one seed and return (test-set metrics, state_dict)."""
    stratify = y if len(np.unique(y)) > 1 else None
    try:
        Xg_train, Xg_test, Xu_train, Xu_test, y_train, y_test = train_test_split(
            Xg, Xu, y, test_size=0.2, random_state=rseed, stratify=stratify,
        )
    except ValueError:
        Xg_train, Xg_test, Xu_train, Xu_test, y_train, y_test = train_test_split(
            Xg, Xu, y, test_size=0.2, random_state=rseed, stratify=None,
        )

    # Validation split via index-based approach
    Xg_tr, Xu_tr, y_tr = Xg_train, Xu_train, y_train
    val_loader = None
    if mcfg.val_fraction and mcfg.val_fraction > 0.0 and len(Xg_train) > 10:
        stratify_tv = y_train if len(np.unique(y_train)) > 1 else None
        try:
            idx = np.arange(len(Xg_train))
            try:
                tr_idx, val_idx = train_test_split(
                    idx, test_size=mcfg.val_fraction, random_state=rseed, stratify=stratify_tv,
                )
            except ValueError:
                tr_idx, val_idx = train_test_split(
                    idx, test_size=mcfg.val_fraction, random_state=rseed, stratify=None,
                )
            Xg_tr, Xu_tr, y_tr = Xg_train[tr_idx], Xu_train[tr_idx], y_train[tr_idx]
            Xg_val, Xu_val, y_val = Xg_train[val_idx], Xu_train[val_idx], y_train[val_idx]
            _, val_loader = make_loaders_two_tower(
                Xg_tr[:1], Xu_tr[:1], y_tr[:1],
                Xg_val, Xu_val, y_val,
                mcfg.batch_size, device,
            )
        except Exception:
            Xg_tr, Xu_tr, y_tr = Xg_train, Xu_train, y_train
            val_loader = None

    # Loaders
    train_weights = compute_class_sample_weights(y_tr)
    train_loader, _ = make_loaders_two_tower(
        Xg_tr, Xu_tr, y_tr, None, None, None, mcfg.batch_size, device,
        train_sample_weights=train_weights,
    )
    test_loader, _ = make_loaders_two_tower(
        Xg_test, Xu_test, y_test, None, None, None, mcfg.batch_size, device,
    )

    # Model
    model = TwoTowerMLP(Xg.shape[1], Xu.shape[1], mcfg.hidden_dims, mcfg.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=mcfg.learning_rate,
                                  weight_decay=mcfg.weight_decay)

    best_score = float("-inf")
    best_state = None
    epochs_no_improve = 0
    monitor = mcfg.es_metric

    epoch_iter = tqdm(range(mcfg.epochs), desc="epochs", leave=False)
    for _ in epoch_iter:
        loss, train_acc = train_epoch_two_tower(model, train_loader, optimizer, criterion, device)
        postfix = {"loss": f"{loss:.4f}", "train_acc": f"{train_acc:.3f}"}
        if val_loader is not None:
            val_m = evaluate_two_tower(model, val_loader, device)
            cur = val_m.get(monitor, float("nan"))
            postfix["val_" + monitor] = f"{cur:.3f}" if not math.isnan(cur) else "nan"
            improved = (not math.isnan(cur)) and (cur > best_score + mcfg.es_min_delta)
            if improved:
                best_score = cur
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if mcfg.use_early_stopping and epochs_no_improve >= mcfg.es_patience:
                epoch_iter.set_postfix(postfix)
                break
        epoch_iter.set_postfix(postfix)

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate_two_tower(model, test_loader, device)
    state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return metrics, state
