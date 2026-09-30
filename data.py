"""
Dataset loading, alignment and antibiotic selection.

Inputs (BioStudies S-BSST2698):
  - ``ndaro_baseline.csv``: NDARO isolates. Columns ``assembly``, ``organism``, 935 ``g_*``
    gene presence/absence features and 112 ``a_*`` phenotypes (``S``, ``R`` or empty).
    An assembly can appear in several rows (one per NCBI release); the first row is kept.
  - ``<name>.npz`` + ``<name>_columns.pkl`` + ``<name>_assemblies.pkl`` for
    ``bakta50``, ``bakta50_amr``, ``bakta90`` and ``bakta90_amr``: sparse matrix whose
    columns are UniRef cluster presence features plus ``a_*`` phenotype columns
    (1 = R, 0 = S, NaN = not tested).

Every source is converted to a :class:`Source` (sparse features + labels coded as
-1 = missing, 0 = S, 1 = R). Combined representations are built by aligning sources on
their shared assemblies; phenotype labels are taken from the first source (NDARO for
the combined representations).
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger("amr")

# BioStudies file stem of each BAKTA representation.
BAKTA_FILES: Dict[str, str] = {
    "BAKTA50": "bakta50",
    "BAKTA50 AMR": "bakta50_amr",
    "BAKTA90": "bakta90",
    "BAKTA90 AMR": "bakta90_amr",
}

# The nine representations of Table 2 and the sources they are built from.
# The first source provides the phenotype labels (and the organism, if present).
DATASETS: Dict[str, List[str]] = {
    "NDARO": ["NDARO"],
    "NDARO + BAKTA50": ["NDARO", "BAKTA50"],
    "NDARO + BAKTA50 AMR": ["NDARO", "BAKTA50 AMR"],
    "NDARO + BAKTA90": ["NDARO", "BAKTA90"],
    "NDARO + BAKTA90 AMR": ["NDARO", "BAKTA90 AMR"],
    "BAKTA50 AMR": ["BAKTA50 AMR"],
    "BAKTA90 AMR": ["BAKTA90 AMR"],
    "BAKTA50": ["BAKTA50"],
    "BAKTA90": ["BAKTA90"],
}


@dataclass
class Source:
    """One feature source (NDARO or a BAKTA representation)."""
    features: sp.csr_matrix          # (n_assemblies, n_features), float32
    feature_names: List[str]
    assemblies: List[str]
    labels: np.ndarray               # (n_assemblies, n_antibiotics), int8: -1 missing, 0 S, 1 R
    label_names: List[str]           # "a_<antibiotic>"
    organism: Optional[List[str]] = None


@dataclass
class AlignedData:
    """One or two feature blocks restricted to the assemblies shared by all sources."""
    blocks: List[sp.csr_matrix]
    assemblies: List[str]
    labels: np.ndarray
    label_names: List[str]
    organism: Optional[np.ndarray] = None

    def subset(self, rows: np.ndarray) -> "AlignedData":
        """Keep only the given rows (boolean mask or integer indices)."""
        rows = np.flatnonzero(rows) if rows.dtype == bool else rows
        return AlignedData(
            blocks=[b[rows] for b in self.blocks],
            assemblies=[self.assemblies[i] for i in rows],
            labels=self.labels[rows],
            label_names=self.label_names,
            organism=None if self.organism is None else self.organism[rows],
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_ndaro_csv(csv_path: Path, chunksize: int = 100_000) -> Source:
    """Read the NDARO CSV in chunks, keeping the first row of every assembly."""
    header = pd.read_csv(csv_path, nrows=0).columns
    g_cols = [c for c in header if c.startswith("g_")]
    a_cols = [c for c in header if c.startswith("a_")]
    keep = ["assembly", "organism"] + g_cols + a_cols
    logger.info("Reading %s (%d genes, %d antibiotics)", csv_path, len(g_cols), len(a_cols))

    seen: set = set()
    parts = []
    dtypes = {**{c: "Int8" for c in g_cols}, **{c: str for c in a_cols}}
    for chunk in pd.read_csv(csv_path, usecols=keep, chunksize=chunksize, dtype=dtypes):
        chunk = chunk[~chunk["assembly"].isin(seen)].drop_duplicates(subset=["assembly"])
        seen.update(chunk["assembly"].tolist())
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)

    features = sp.csr_matrix(df[g_cols].fillna(0).to_numpy(dtype=np.int8).astype(np.float32))
    labels = np.full((len(df), len(a_cols)), -1, dtype=np.int8)
    for j, c in enumerate(a_cols):
        col = df[c].astype("string")
        labels[(col == "S").fillna(False).to_numpy(dtype=bool), j] = 0
        labels[(col == "R").fillna(False).to_numpy(dtype=bool), j] = 1
    logger.info("NDARO: %d unique assemblies", len(df))
    return Source(features=features, feature_names=g_cols, assemblies=df["assembly"].tolist(),
                  labels=labels, label_names=a_cols, organism=df["organism"].tolist())


def load_bakta(prefix: Path) -> Source:
    """Read a BAKTA triplet ``<prefix>.npz``, ``<prefix>_columns.pkl``, ``<prefix>_assemblies.pkl``."""
    mat = sp.load_npz(str(prefix) + ".npz")
    with open(str(prefix) + "_columns.pkl", "rb") as f:
        columns = pickle.load(f)
    with open(str(prefix) + "_assemblies.pkl", "rb") as f:
        assemblies = pickle.load(f)
    ab_idx = [i for i, c in enumerate(columns) if str(c).startswith("a_")]
    ab_set = set(ab_idx)
    feat_idx = [i for i in range(len(columns)) if i not in ab_set]

    raw = mat[:, ab_idx].toarray()
    labels = np.full(raw.shape, -1, dtype=np.int8)
    labels[raw == 0] = 0
    labels[raw == 1] = 1
    logger.info("%s: %d assemblies, %d features", Path(prefix).name, len(assemblies), len(feat_idx))
    return Source(features=mat[:, feat_idx].tocsr().astype(np.float32),
                  feature_names=[str(columns[i]) for i in feat_idx],
                  assemblies=list(assemblies),
                  labels=labels, label_names=[str(columns[i]) for i in ab_idx])


class SourceCache:
    """Loads every source at most once."""

    def __init__(self, ndaro_csv: Path, bakta_dir: Path):
        self.ndaro_csv = Path(ndaro_csv)
        self.bakta_dir = Path(bakta_dir)
        self._cache: Dict[str, Source] = {}

    def get(self, name: str) -> Source:
        if name not in self._cache:
            if name == "NDARO":
                self._cache[name] = load_ndaro_csv(self.ndaro_csv)
            elif name in BAKTA_FILES:
                self._cache[name] = load_bakta(self.bakta_dir / BAKTA_FILES[name])
            else:
                raise ValueError(f"Unknown source {name!r}")
        return self._cache[name]

    def organism_of(self) -> Dict[str, str]:
        """assembly -> organism, from NDARO (BAKTA files carry no organism)."""
        nd = self.get("NDARO")
        return dict(zip(nd.assemblies, nd.organism))


# ---------------------------------------------------------------------------
# Alignment and selection
# ---------------------------------------------------------------------------
def align(sources: Sequence[Source]) -> AlignedData:
    """Restrict every source to the (sorted) assemblies they share."""
    common = sorted(set.intersection(*[{a for a in s.assemblies if isinstance(a, str) and a}
                                       for s in sources]))
    position = {a: i for i, a in enumerate(common)}
    source_rows = []
    for s in sources:
        rows = np.zeros(len(common), dtype=np.int64)
        for i, a in enumerate(s.assemblies):
            if a in position:
                rows[position[a]] = i
        source_rows.append(rows)

    organism = next((np.asarray([s.organism[i] for i in rows], dtype=object)
                     for s, rows in zip(sources, source_rows) if s.organism is not None), None)
    return AlignedData(blocks=[s.features[rows] for s, rows in zip(sources, source_rows)],
                       assemblies=common,
                       labels=sources[0].labels[source_rows[0]],
                       label_names=sources[0].label_names,
                       organism=organism)


def load_dataset(name: str, cache: SourceCache, organism: Optional[str] = None) -> AlignedData:
    """Build one of the representations in :data:`DATASETS`, optionally for a single organism."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(DATASETS)}")
    data = align([cache.get(s) for s in DATASETS[name]])
    if organism is not None:
        if data.organism is None:
            lookup = cache.organism_of()
            data.organism = np.asarray([lookup.get(a) for a in data.assemblies], dtype=object)
        data = data.subset(data.organism == organism)
        if not data.assemblies:
            raise ValueError(f"No assemblies for organism {organism!r}")
    return data


def eligible_antibiotics(data: AlignedData, min_samples: int, min_minority: int,
                         only: Optional[Sequence[str]] = None) -> List[str]:
    """Antibiotics (``a_*`` names) with >= ``min_samples`` labelled isolates and
    >= ``min_minority`` isolates in the minority class."""
    names = data.label_names if only is None else [f"a_{a}" for a in only]
    out = []
    for name in names:
        if name not in data.label_names:
            logger.warning("Antibiotic %s not found; skipped", name)
            continue
        y = data.labels[:, data.label_names.index(name)]
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        if n_pos + n_neg >= min_samples and min(n_pos, n_neg) >= min_minority:
            out.append(name)
    return out


def build_xy(data: AlignedData, antibiotic: str):
    """Dense feature blocks, labels and organism of the isolates labelled for one antibiotic."""
    y_raw = data.labels[:, data.label_names.index(antibiotic)]
    mask = y_raw >= 0
    blocks = [b[mask].toarray().astype(np.float32) for b in data.blocks]
    organism = None if data.organism is None else data.organism[mask]
    return blocks, y_raw[mask].astype(np.int64), organism
