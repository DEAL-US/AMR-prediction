"""
Configuration dataclasses and default constants for the baseline experiments.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Hyperparameters for the MLP models."""
    hidden_dims: List[int] = None       # e.g., [512, 256]
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 512
    epochs: int = 30
    # Validation / Early stopping
    val_fraction: float = 0.1
    use_early_stopping: bool = True
    es_metric: str = "f1"               # one of: f1, accuracy, precision, recall
    es_patience: int = 10
    es_min_delta: float = 0.0

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256]


@dataclass
class RunConfig:
    """Controls the experimental protocol (seeds, splits, label mapping)."""
    test_size: float = 0.2
    runs_per_antibiotic: int = 30
    base_seed: int = 1337
    label_map: Dict[str, int] = None    # mapping of antibiotic phenotype labels to 0/1

    def __post_init__(self):
        if self.label_map is None:
            self.label_map = {"S": 0, "R": 1}


@dataclass
class DatasetConfig:
    """Describes a single dataset (or a combined dataset) to evaluate."""
    name: str
    path: Path
    feature_prefix: str
    antibiotic_prefix: str = "a_"
    is_combined: bool = False                       # True when this is a combined dataset (NDARO + sparse)
    sparse_component_name: Optional[str] = None     # Key into sparse_datasets dict for combined datasets
