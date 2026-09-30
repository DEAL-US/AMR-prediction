"""
Configuration dataclasses shared by the training entry points
(``main.py`` and ``scripts/loso_evaluation.py``).
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """Network architecture and optimisation hyperparameters."""
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 512
    max_epochs: int = 200
    patience: int = 10              # early stopping on validation F1
    min_delta: float = 1e-6         # minimum validation-F1 improvement that resets patience


@dataclass
class RunConfig:
    """Experimental protocol: splits, seeds, antibiotic eligibility and decision threshold."""
    n_seeds: int = 30               # independent seeds per (dataset, antibiotic)
    seed_offset: int = 0            # seeds used: seed_offset, ..., seed_offset + n_seeds - 1
    test_size: float = 0.2          # 20% test
    val_fraction: float = 0.1       # 10% of the remaining 80% -> 72 / 8 / 20 split
    min_samples: int = 50           # antibiotics need >= 50 labelled isolates ...
    min_minority: int = 5           # ... and >= 5 isolates in the minority class
    threshold: float = 0.5          # decision threshold on the predicted probability
