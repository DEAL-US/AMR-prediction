# Antibiotic Resistance Prediction – Combining curated AMR genes and genome-wide annotations

Binary classification of antibiotic **susceptibility (S) vs. resistance (R)** using neural networks trained on genomic feature matrices.

## Overview

This repository contains the code for reproducing experiments for AMR prediction. Given a set of bacterial genome assemblies from NCBI NDARO database annotated with antibiotic susceptibility phenotypes and their annotations by BAKTA, the pipeline:

1. Loads one or more genomic feature datasets (gene presence/absence matrices).
2. For each antibiotic, trains a binary classifier over multiple random seeds.
3. Reports aggregated metrics (F1, accuracy, precision, recall, confusion matrix counts) in a single results CSV.

Two model architectures are provided:

| Architecture | Use case | Description |
|---|---|---|
| **BaselineMLP** | Single feature source | Standard feed-forward network: `[Linear → ReLU → Dropout] × N → Linear(1)` |
| **TwoTowerMLP** | Combined feature sources | Two parallel encoder branches (one per feature source) whose embeddings are concatenated and passed through a linear fusion head |

## Datasets

The experiments use three types of datasets. All are available on BioStudies:

> **[Dataset in BioStudies](https://www.ebi.ac.uk/biostudies/studies/S-BSST2698)**

### 1. NDARO (CSV)

Isolate-level information from NDARO Isolates Browser, comprising approximately 20,000 isolates from diverse bacterial species, with resistance annotations for 935 AMR-associated genes and phenotypic susceptibility information spanning 112 antibiotics. A single CSV file (`processed.csv`) with columns:

| Column pattern | Description |
|---|---|
| `assembly` | Unique genome assembly identifier |
| `organism` | Organism name |
| `g_*` | Gene presence/absence features (binary 0/1) |
| `a_*` | Antibiotic phenotype labels (`S` = susceptible, `R` = resistant, or missing) |

### 2. BAKTA Datasets (NPZ + PKL)

In addition to the isolate-level metadata, we processed the genome assembly of each sample thorugh BAKTA, an automated bacterial genome annotation tool that produces standardized functional descriptions of genomic elements. In order to save space with a more compressed format, each BAKTA dataset consists of a **triplet** of files sharing the same base name:

| File | Format | Content |
|---|---|---|
| `<base>.npz` | SciPy sparse matrix | Rows = assemblies, columns = features + antibiotic labels |
| `<base>_assemblies.pkl` | Python pickle (list) | Assembly identifiers (one per row) |
| `<base>_columns.pkl` | Python pickle (list) | Column names; those starting with `a_` are antibiotic labels, the rest are features |

Four variants are provided:

| Name | Description |
|---|---|
| **BAKTA50** | UniRef50 cluster presence, filtered to clusters present in ≥500 assemblies |
| **BAKTA50 AMR** | UniRef50 clusters restricted to AMR-associated genes (via AMRFinder) |
| **BAKTA90** | UniRef90 cluster presence, filtered ≥500 |
| **BAKTA90 AMR** | UniRef90 clusters restricted to AMR-associated genes |

Feature columns in the sparse datasets are prefixed with `U_` (UniRef).

### 3. Combined Datasets (built on-the-fly)

For each BAKTA variant, the pipeline also creates a **combined** dataset by inner-joining NDARO and the sparse dataset on the `assembly` column. These combined datasets have both `g_*` (gene) and `U_*` (UniRef) feature blocks and are evaluated using the `TwoTowerMLP` architecture.

### Column naming conventions

| Prefix | Meaning |
|---|---|
| `g_` | Gene-level features (from NDARO) |
| `U_` | UniRef cluster features (from BAKTA) |
| `a_` | Antibiotic phenotype labels |

## Project Structure

```
├── main.py           # Entry point: configuration and experiment orchestration
├── config.py         # Dataclass definitions (ModelConfig, RunConfig, DatasetConfig)
├── models.py         # Neural network architectures (BaselineMLP, TwoTowerMLP)
├── data.py           # Dataset loading (CSV, sparse NPZ/PKL, combined)
├── training.py       # Training loops, evaluation, data-loader construction
├── utils.py          # Metric aggregation, results I/O, logging, device selection
├── scripts/          # Standalone analyses (AMRFinderPlus benchmark, LOSO)
├── requirements.txt  # Python dependencies
└── README.md
```

## Additional analyses

The `scripts/` folder contains two standalone analyses, with their own
[README](scripts/README.md):

- **`amrfinderplus_baseline.py`** — rule-based AMRFinderPlus benchmark.
- **`loso_evaluation.py`** — leave-one-species-out cross-species generalisation.

Both are configured via a `CONFIG` block at the top of each file (as in
`main.py`), are independent of the main pipeline, and reuse the same datasets.

## Prerequisites

- **Python** ≥ 3.8
- **PyTorch** (CPU or CUDA)
- The remaining dependencies are listed in `requirements.txt`

> **Note:** CUDA is strongly recommended. Training on CPU is functional but significantly slower.

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd <repo-name>

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3.1 Install PyTorch for your system (example for CUDA 12.6)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# 3.2 Or install PyTorch for CPU usage
pip3 install torch torchvision

# All PyTorch installation configurations can be found at https://pytorch.org/get-started/locally/

# 4. Install remaining dependencies
pip install -r requirements.txt
```

## Configuration

All settings are defined in the `config` dictionary inside `main.py`. Open the file and edit the values to match your setup:

### Dataset paths

```python
"ndaro_path": "/path/to/data/ndaro/processed.csv",
"sparse_datasets": {
    "BAKTA50":     "/path/to/data/bakta/bakta50.npz",
    "BAKTA50 AMR": "/path/to/data/bakta/bakta50_amr.npz",
    "BAKTA90":     "/path/to/data/bakta/bakta90.npz",
    "BAKTA90 AMR": "/path/to/data/bakta/bakta90_amr.npz",
},
```

For each sparse dataset path (e.g. `.../bakta50.npz`), the loader expects `_assemblies.pkl` and `_columns.pkl` companion files with the same base name in the same directory.

Datasets link:
> **[Dataset in BioStudies](https://www.ebi.ac.uk/biostudies/studies/S-BSST2698)**

### Model hyperparameters

| Key | Default | Description |
|---|---|---|
| `hidden_dims` | `[512, 256]` | Hidden layer sizes for each MLP |
| `dropout` | `0.2` | Dropout rate between hidden layers |
| `lr` | `1e-3` | Learning rate (AdamW) |
| `weight_decay` | `0.0` | L2 regularization |
| `batch_size` | `512` | Mini-batch size |
| `epochs` | `200` | Maximum training epochs |

### Early stopping

| Key | Default | Description |
|---|---|---|
| `val_fraction` | `0.1` | Fraction of training data held out for validation |
| `use_early_stopping` | `True` | Enable early stopping based on validation metric |
| `es_metric` | `"f1"` | Metric to monitor (`f1`, `accuracy`, `precision`, `recall`) |
| `es_patience` | `10` | Epochs without improvement before stopping |
| `es_min_delta` | `0.0` | Minimum improvement to reset patience counter |

### Experiment protocol

| Key | Default | Description |
|---|---|---|
| `runs` | `30` | Number of independent random seeds per antibiotic |
| `output_dir` | `"results"` | Directory where `results.csv` will be written |
| `limit_antibiotics` | `None` | List of antibiotic names to evaluate, or `None` for all |
| `save_models` | `False` | Save the best model's `state_dict` (`.pt`) per (dataset, antibiotic) |
| `save_detailed_results` | `False` | Save per-run metrics to `results_detailed.csv` |

## Running

```bash
python main.py
```

The script will iterate over all configured datasets (NDARO alone, each BAKTA variant alone, and each BAKTA + NDARO combined) and, for each antibiotic column, run the specified number of training seeds.

Training progress is displayed via `tqdm` progress bars at the dataset, antibiotic, run, and epoch levels.

## Output

Results are written to `<output_dir>/results.csv` (default: `results/results.csv`). Each row corresponds to one (dataset, antibiotic) pair with metrics averaged over all runs:

| Column | Description |
|---|---|
| `dataset` | Dataset name (e.g. `NDARO`, `BAKTA50`, `BAKTA50 + NDARO`) |
| `antibiotic` | Antibiotic name (without the `a_` prefix) |
| `n_samples` | Total samples with S or R label for this antibiotic |
| `n_pos` | Number of resistant (R) samples |
| `n_neg` | Number of susceptible (S) samples |
| `runs` | Number of random seeds used |
| `f1` | Mean F1 score across runs |
| `accuracy` | Mean accuracy across runs |
| `precision` | Mean precision across runs |
| `recall` | Mean recall across runs |
| `tp`, `tn`, `fp`, `fn` | Mean confusion matrix counts (rounded) |
| `hidden_dims` | Model hidden layer sizes (JSON) |
| `dropout` | Dropout rate used |
| `learning_rate` | Learning rate used |
| `batch_size` | Batch size used |
| `epochs` | Maximum epochs configured |

### Resume support

The script checks the existing `results.csv` before starting each antibiotic. If a (dataset, antibiotic) pair is already present, it is skipped. This means user can safely **interrupt and restart** the script.

### Detailed per-run results (`save_detailed_results`)

When enabled, the script writes `<output_dir>/results_detailed.csv` alongside the aggregated `results.csv`. Each row records a **single run** instead of the averaged metrics:

| Column | Description |
|---|---|
| `dataset` | Dataset name |
| `antibiotic` | Antibiotic name |
| `run` | Run index (0-based) |
| `seed` | Random seed used for this run |
| `n_samples`, `n_pos`, `n_neg` | Sample counts |
| `f1`, `accuracy`, `precision`, `recall` | Metrics for this individual run (4 decimal places) |
| `tp`, `tn`, `fp`, `fn` | Confusion matrix counts for this run |

### Saved models (`save_models`)

When enabled, the script saves the PyTorch `state_dict` of the **best model** (highest test F1 across all runs) for each (dataset, antibiotic) pair:

```
<output_dir>/models/<dataset_name>/<antibiotic_name>.pt
```

To reload a saved model:

```python
import torch
from models import BaselineMLP  # or TwoTowerMLP for combined datasets

model = BaselineMLP(input_dim=..., hidden_dims=[512, 256], dropout=0.2)
model.load_state_dict(torch.load("results/models/NDARO/ciprofloxacin.pt"))
model.eval()
```

## Training details

- **Class balancing**: per-sample weights inversely proportional to class frequency are applied during training to handle imbalanced S/R distributions.
- **Loss function**: `BCEWithLogitsLoss` with per-sample weighting.
- **Optimizer**: AdamW.

## Hardware

Experiments were run on the following machine:

| Component | Specification |
|---|---|
| **CPU** | AMD Ryzen Threadripper PRO 3955WX (16 cores / 32 threads) |
| **RAM** | 64 GB DDR4 |
| **GPU** | NVIDIA RTX A5000 (24 GB VRAM) |
| **OS** | Ubuntu Linux |


