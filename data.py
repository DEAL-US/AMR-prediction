"""
Dataset loading, preparation, and feature/label extraction utilities.

Handles three dataset formats:
  1. NDARO CSV  – standard CSV with ``g_*`` features and ``a_*`` antibiotic labels
  2. Sparse BAKTA – triplet of ``.npz`` (scipy sparse matrix) + ``_assemblies.pkl``
     + ``_columns.pkl``
  3. Combined  – inner-join of NDARO + a sparse BAKTA dataset, aligned by assembly id
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse

from config import DatasetConfig

logger = logging.getLogger("baseline")


# ---------------------------------------------------------------------------
# Dataset config generation
# ---------------------------------------------------------------------------
def generate_dataset_configs(
    ndaro_path: Path,
    sparse_datasets: Dict[str, Path],
) -> List[DatasetConfig]:
    """
    Build the list of :class:`DatasetConfig` objects to evaluate.

    Returns (in order):
      - NDARO baseline
      - For each sparse dataset: the sparse dataset alone, then combined with NDARO
    """
    configs: List[DatasetConfig] = []

    # 1. NDARO baseline
    configs.append(DatasetConfig(
        name="NDARO",
        path=ndaro_path,
        feature_prefix="g_",
        antibiotic_prefix="a_",
    ))

    # 2. Each sparse dataset alone + combined with NDARO
    for sparse_name, sparse_path in sparse_datasets.items():
        configs.append(DatasetConfig(
            name=sparse_name,
            path=sparse_path,
            feature_prefix="U_",
            antibiotic_prefix="a_",
        ))
        configs.append(DatasetConfig(
            name=f"{sparse_name} + NDARO",
            path=Path("."),               
            feature_prefix="g_|U_",       
            antibiotic_prefix="a_",
            is_combined=True,
            sparse_component_name=sparse_name,
        ))

    return configs


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------
def read_dataset(
    cfg: DatasetConfig,
    ndaro_path: Path,
    sparse_datasets: Dict[str, Path],
    allowed_assemblies: Optional[Set[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Load a dataset described by *cfg*.

    Parameters
    ----------
    cfg : DatasetConfig
        Which dataset to load.
    ndaro_path : Path
        Path to the NDARO CSV (needed when building combined datasets).
    sparse_datasets : dict
        Mapping of sparse dataset names to their ``.npz`` base paths.
    allowed_assemblies : set, optional
        If given, only keep rows whose ``assembly`` column is in this set.
    """

    # --- Combined datasets (e.g. "BAKTA50 + NDARO") -------------------------
    if cfg.is_combined and cfg.sparse_component_name:
        return _load_combined(cfg, ndaro_path, sparse_datasets, allowed_assemblies)

    # --- Sparse datasets (BAKTA variants) ------------------------------------
    if _is_sparse_dataset(cfg.name, sparse_datasets):
        return _load_sparse(cfg, allowed_assemblies)

    # --- CSV datasets (NDARO) ------------------------------------------------
    return _load_csv(cfg, allowed_assemblies)


# ---------------------------------------------------------------------------
# Private loaders
# ---------------------------------------------------------------------------
def _is_sparse_dataset(name: str, sparse_datasets: Dict[str, Path]) -> bool:
    return name in sparse_datasets or name.startswith("BAKTA")


def _load_combined(
    cfg: DatasetConfig,
    ndaro_path: Path,
    sparse_datasets: Dict[str, Path],
    allowed_assemblies: Optional[Set[str]],
) -> Optional[pd.DataFrame]:
    sparse_name = cfg.sparse_component_name
    if sparse_name not in sparse_datasets:
        logger.warning(
            "Sparse component '%s' not found in sparse_datasets; skipping %s",
            sparse_name, cfg.name,
        )
        return None

    logger.info("Building combined dataset '%s' from NDARO + %s", cfg.name, sparse_name)

    ndaro_cfg = DatasetConfig(name="NDARO", path=ndaro_path, feature_prefix="g_")
    sparse_cfg = DatasetConfig(
        name=sparse_name, path=sparse_datasets[sparse_name], feature_prefix="U_",
    )

    ndaro_df = read_dataset(ndaro_cfg, ndaro_path, sparse_datasets, allowed_assemblies)
    sparse_df = read_dataset(sparse_cfg, ndaro_path, sparse_datasets, allowed_assemblies)

    if ndaro_df is None or sparse_df is None:
        logger.warning("A component dataset could not be loaded; skipping combined dataset")
        return None
    if "assembly" not in ndaro_df.columns or "assembly" not in sparse_df.columns:
        logger.warning("'assembly' column missing in a component; skipping combined dataset")
        return None

    # Align on the intersection of assemblies
    ndaro_df = ndaro_df.drop_duplicates(subset=["assembly"]).set_index("assembly")
    sparse_df = sparse_df.drop_duplicates(subset=["assembly"]).set_index("assembly")
    common = ndaro_df.index.intersection(sparse_df.index)
    if len(common) == 0:
        logger.warning("No common assemblies between NDARO and %s", sparse_name)
        return None
    ndaro_df = ndaro_df.loc[common]
    sparse_df = sparse_df.loc[common]

    # Antibiotic columns: coalesce preferring sparse labels
    ab_ndaro = [c for c in ndaro_df.columns if isinstance(c, str) and c.startswith("a_")]
    ab_sparse = [c for c in sparse_df.columns if isinstance(c, str) and c.startswith("a_")]
    ab_all = sorted(set(ab_ndaro) | set(ab_sparse))
    ab_data = {}
    for c in ab_all:
        s_sp = sparse_df[c] if c in sparse_df.columns else pd.Series(index=common, dtype=object)
        s_nd = ndaro_df[c] if c in ndaro_df.columns else pd.Series(index=common, dtype=object)
        ab_data[c] = s_sp.combine_first(s_nd)
    ab_df = pd.DataFrame(ab_data, index=common)

    # Feature blocks
    g_feats = [c for c in ndaro_df.columns if isinstance(c, str) and c.startswith("g_")]
    u_feats = [c for c in sparse_df.columns if isinstance(c, str) and c.startswith("U_")]
    g_block = ndaro_df[g_feats] if g_feats else pd.DataFrame(index=common)
    u_block = sparse_df[u_feats] if u_feats else pd.DataFrame(index=common)

    combined_df = pd.DataFrame({"assembly": common}, index=common)
    combined_df = pd.concat([combined_df, ab_df, g_block, u_block], axis=1)
    combined_df = combined_df.reset_index(drop=True)
    logger.info(
        "Combined dataset built: assemblies=%d, antibiotics=%d, "
        "g_feats=%d, U_feats=%d, total_cols=%d",
        len(common), len(ab_all), len(g_feats), len(u_feats), combined_df.shape[1],
    )
    return combined_df


def _load_sparse(
    cfg: DatasetConfig,
    allowed_assemblies: Optional[Set[str]],
) -> Optional[pd.DataFrame]:
    base = cfg.path
    base_root = base.with_suffix("") if base.suffix == ".npz" else base
    npz_path = base if base.suffix == ".npz" else Path(str(base) + ".npz")
    assemblies_path = Path(str(base_root) + "_assemblies.pkl")
    columns_path = Path(str(base_root) + "_columns.pkl")

    if not (npz_path.exists() and assemblies_path.exists() and columns_path.exists()):
        logger.warning(
            "Sparse dataset files missing for %s: %s, %s, %s",
            cfg.name, npz_path, assemblies_path, columns_path,
        )
        return None

    logger.info("Loading sparse dataset '%s' from base: %s", cfg.name, base_root)
    mat = scipy.sparse.load_npz(str(npz_path))
    with open(columns_path, "rb") as f:
        columns = pickle.load(f)
    with open(assemblies_path, "rb") as f:
        assemblies = pickle.load(f)

    # Separate antibiotic vs. feature columns
    ab_idx = [i for i, c in enumerate(columns)
              if isinstance(c, str) and c.startswith(cfg.antibiotic_prefix)]
    feat_idx = [i for i, c in enumerate(columns) if i not in set(ab_idx)]
    ab_cols = [columns[i] for i in ab_idx]
    feat_cols_raw = [columns[i] for i in feat_idx]
    feat_cols = [
        c if isinstance(c, str) and c.startswith(cfg.feature_prefix)
        else f"{cfg.feature_prefix}{c}"
        for c in feat_cols_raw
    ]

    df = pd.DataFrame({"assembly": assemblies})

    # Antibiotic block (small, dense) – remap numeric 0/1 to "S"/"R"
    Y = mat[:, ab_idx].toarray().astype("float32")
    ab_df = pd.DataFrame(Y, columns=ab_cols)
    ab_df = ab_df.apply(lambda s: s.map({0: "S", 1: "R", 0.0: "S", 1.0: "R"}))
    df = pd.concat([df, ab_df], axis=1)

    # Feature block – keep sparse for memory efficiency
    X_sparse = mat[:, feat_idx]
    feat_df = pd.DataFrame.sparse.from_spmatrix(X_sparse, columns=feat_cols)
    df = pd.concat([df, feat_df], axis=1)

    if allowed_assemblies is not None and not df.empty:
        df = df[df["assembly"].isin(allowed_assemblies)]

    logger.info(
        "Loaded sparse dataset shape: %s (features: %d, antibiotics: %d)",
        df.shape, len(feat_cols), len(ab_cols),
    )
    return df


def _load_csv(
    cfg: DatasetConfig,
    allowed_assemblies: Optional[Set[str]],
) -> Optional[pd.DataFrame]:
    if not cfg.path.exists():
        logger.warning("Dataset not found, skipping: %s -> %s", cfg.name, cfg.path)
        return None

    logger.info("Loading dataset '%s' from %s", cfg.name, cfg.path)
    df = pd.read_csv(cfg.path)

    if allowed_assemblies is not None and "assembly" in df.columns:
        df = df[df["assembly"].isin(allowed_assemblies)]
    logger.info("Loaded shape: %s", df.shape)

    # NDARO deduplication
    if cfg.name == "NDARO":
        before = df.shape[0]
        if "assembly" in df.columns:
            df = df.drop_duplicates(subset=["assembly"]).reset_index(drop=True)
        else:
            df = df.drop_duplicates().reset_index(drop=True)
        logger.info("Deduplicated '%s' rows: %d -> %d", cfg.name, before, df.shape[0])

    return df


# ---------------------------------------------------------------------------
# Feature / label helpers
# ---------------------------------------------------------------------------
def get_antibiotic_columns(df: pd.DataFrame, prefix: str = "a_") -> List[str]:
    """Return column names that start with the antibiotic prefix."""
    return [c for c in df.columns if c.startswith(prefix)]


def get_feature_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    """Return column names that start with the given feature prefix."""
    return [c for c in df.columns if c.startswith(prefix)]


def prepare_xy(
    df: pd.DataFrame,
    antibiotic_col: str,
    feature_cols: List[str],
    label_map: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Filter rows to those with labels in *label_map* and return
    ``(X, y, counts_dict)``.
    """
    filtered = df[df[antibiotic_col].isin(label_map.keys())].copy()
    if filtered.empty:
        return (
            np.empty((0, len(feature_cols))),
            np.empty((0,), dtype=np.int64),
            {"n_total": 0, "n_pos": 0, "n_neg": 0},
        )

    X = (filtered[feature_cols]
         .apply(pd.to_numeric, errors="coerce")
         .fillna(0.0)
         .to_numpy(dtype=np.float32))
    y = filtered[antibiotic_col].map(label_map).to_numpy(dtype=np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return X, y, {"n_total": len(y), "n_pos": n_pos, "n_neg": n_neg}
