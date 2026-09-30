# Neural networks combining curated resistance genes and genome-wide annotations for antimicrobial resistance prediction

For every antibiotic, a binary classifier predicts **resistance (R) vs susceptibility (S)** of bacterial isolates from genomic features. Nine feature representations are compared with the same learning algorithm, splits and seeds:

| Representation | Features |
|---|---|
| **NDARO** | presence/absence of the 935 curated AMR genes detected by AMRFinderPlus (NCBI NDARO) |
| **BAKTA50 AMR** / **BAKTA90 AMR** | AMR-related genes annotated by BAKTA, clustered at UniRef50 (1,047 clusters) / UniRef90 (1,901) |
| **BAKTA50** / **BAKTA90** | all BAKTA-annotated genes, clustered at UniRef50 (20,407 clusters) / UniRef90 (25,647), keeping clusters present in at least 500 isolates |
| **NDARO + BAKTA…** | each of the four BAKTA representations combined with NDARO (two-tower model) |

All models are compared with a **rule-based AMRFinderPlus baseline**, which needs no training: an isolate is predicted resistant when it carries at least one AMRFinderPlus-detected gene (the same 935 NDARO genes) that targets the antibiotic's drug class according to the CARD ARO index (`scripts/amrfinderplus_baseline.py`).

Two architectures are used:

| Architecture | Used for | Description |
|---|---|---|
| **BaselineMLP** | one feature source | `[Linear → ReLU → Dropout] × 2 → Linear(1)`, hidden sizes 512 and 256 |
| **TwoTowerMLP** | two feature sources | one 512 → 256 tower per source; the two embeddings are concatenated and passed to a linear output layer |

## Data

All inputs are available on BioStudies: **[S-BSST2698](https://www.ebi.ac.uk/biostudies/studies/S-BSST2698)**. Download them into `data/`:

| File | Content |
|---|---|
| `ndaro_baseline.csv` | NDARO isolates: `assembly`, `organism`, 935 `g_*` gene features (0/1) and 112 `a_*` phenotypes (`S`, `R` or empty). An assembly can appear in several rows (one per NCBI release); the first row is used. |
| `bakta50.npz`, `bakta50_amr.npz`, `bakta90.npz`, `bakta90_amr.npz` | sparse matrices (SciPy) with one row per assembly and one column per UniRef cluster, plus the `a_*` phenotype columns (1 = R, 0 = S, NaN = not tested) |
| `<name>_columns.pkl`, `<name>_assemblies.pkl` | column names and assembly identifiers of each matrix |
| `S1_antibiotic_summary.csv`, `S2_antibiotic_species_breakdown.csv` | number of labelled, resistant and susceptible isolates per antibiotic, and per species |

The AMRFinderPlus benchmark also needs the CARD ARO index (`aro_index.tsv`, from [card.mcmaster.ca/download](https://card.mcmaster.ca/download)) in `data/`.

**Antibiotics.** An antibiotic is analysed when it has at least 50 labelled isolates and at least 5 isolates in the minority class. This gives the 61 antibiotics of the paper (amoxicillin, florfenicol, temocillin and ticarcillin-clavulanic acid have 50 or more isolates but fewer than 5 in the minority class).

## Installation

```bash
git clone https://github.com/DEAL-US/AMR-prediction.git
cd AMR-prediction

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# PyTorch for your system (see https://pytorch.org/get-started/locally/), e.g. CUDA 12.6:
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

A GPU is strongly recommended; training on CPU works but is much slower.

## Usage

All scripts are run from the repository root. Settings live in a `CONFIG` block at the top of each script; the main ones can also be overridden from the command line (`--help`).

| Script | Results for | Output (default) |
|---|---|---|
| `main.py` | Table 2, Figure 4, Table 3, Supplementary Table S2 | `results/random_split/` |
| `main.py --organism "<species>"` | Table 5 (models trained within one species) | set `--output-dir` |
| `scripts/amrfinderplus_baseline.py` | AMRFinderPlus baseline (Tables 2, 5 and 6) | `results/amrfinderplus/` |
| `scripts/loso_evaluation.py` | Table 6, Figure 5 (leave-one-species-out) | `results/loso/` |
| `scripts/integrated_gradients.py` | Table 4 (non-AMR clusters, Integrated Gradients) | `results/integrated_gradients/` |

```bash
# All representations, 30 seeds, every eligible antibiotic
python main.py

# Within-species training, e.g. Campylobacter jejuni
python main.py --organism "Campylobacter jejuni" --output-dir results/campylobacter_jejuni

# Rule-based AMRFinderPlus baseline (all isolates and each species)
python scripts/amrfinderplus_baseline.py

# Leave-one-species-out (uses the baseline output for the AMRFinderPlus column)
python scripts/loso_evaluation.py

# Integrated Gradients (uses main.py's results.csv to select the antibiotics)
python scripts/integrated_gradients.py
```

Progress is shown with `tqdm` bars at the dataset, antibiotic, seed and epoch levels (the epoch bar shows the training loss and validation F1; the seed bar, the last test F1 and geometric-mean accuracy).

Organism names are those of the `organism` column: `Salmonella enterica`, `E.coli and Shigella`, `Campylobacter jejuni`, `Acinetobacter baumannii`, …

## Protocol

- **Splits**: stratified 72% training / 8% validation / 20% test (`StratifiedShuffleSplit`), 30 seeds (0–29) per antibiotic and representation. The seed fixes the split, the initialisation, the batch order and dropout, so results are reproducible up to GPU floating-point non-determinism.
- **Within-species training** (`--organism`) applies the same protocol and antibiotic eligibility to the isolates of one species.
- **Training**: AdamW (learning rate 1e-3, no weight decay), batch size 512, binary cross-entropy with per-sample weights inversely proportional to class frequency in the training part, dropout 0.2. At most 200 epochs, with early stopping on validation F1 (patience 10); the best validation checkpoint is evaluated on the test set.
- **Metrics**: F1, precision, recall and accuracy at a 0.5 threshold; specificity; geometric-mean accuracy $\sqrt{\text{sensitivity} \times \text{specificity}}$; PR-AUC and ROC-AUC. Per-antibiotic results are averaged over seeds, then across antibiotics (each antibiotic weighs the same).
- **Combined representations** use the phenotypes of the NDARO file; NDARO alone uses all NDARO assemblies, and combined representations use the assemblies shared by both sources.
- **AMRFinderPlus baseline**: an isolate is called resistant when it carries at least one detected gene targeting the antibiotic's drug class (CARD ARO). It is evaluated on all isolates (Table 2) and on the isolates of each species (per-species and leave-one-species-out comparisons). Its summary averages the antibiotics that the rule can address (at least one gene maps to the drug class).
- **Leave-one-species-out**: each of the four most represented species is held out in turn; the models are trained on the other species (90% training / 10% validation) and tested on all isolates of the held-out species, 5 seeds. A held-out antibiotic is kept when its test set has at least 5 isolates of the minority class.
- **Integrated Gradients**: for the antibiotics where NDARO + BAKTA50 outperforms NDARO + BAKTA50 AMR, IG attributions (zero baseline, 50 steps) are computed for the UniRef50 clusters while each isolate's NDARO genes are held at their observed values. The attribution model follows the settings of the original analysis, documented at the top of the script.

## Outputs

`main.py` writes to `output_dir`:

| File | Content |
|---|---|
| `results_detailed.csv` | one row per (dataset, organism, antibiotic, seed): sample sizes, `f1`, `pr_auc`, `gmean`, `accuracy`, `precision`, `recall`, `specificity`, `roc_auc`, confusion-matrix counts, run time |
| `results.csv` | per (dataset, organism, antibiotic): number of seeds, sample sizes, and `<metric>_mean` / `<metric>_std` across seeds |

Runs can be interrupted and restarted: completed seeds are read from `results_detailed.csv` and skipped. With `save_models = True` the model of the first seed of every (dataset, antibiotic) is saved to `models/<dataset>/<antibiotic>_seed<seed>.pt`:

```python
import torch
from models import BaselineMLP, TwoTowerMLP

model = BaselineMLP(input_dim=935, hidden_dims=[512, 256], dropout=0.2)   # NDARO
model.load_state_dict(torch.load("results/random_split/models/NDARO/ciprofloxacin_seed0.pt"))
model.eval()
```

The outputs of the scripts in `scripts/` are described in [scripts/README.md](scripts/README.md).

## Project structure

```
├── main.py          # entry point: per-antibiotic training and evaluation (random or within-species)
├── config.py        # ModelConfig and RunConfig dataclasses
├── data.py          # loading NDARO and BAKTA files, alignment, antibiotic eligibility
├── models.py        # BaselineMLP, TwoTowerMLP
├── training.py      # splits, training loop with early stopping, metrics
├── utils.py         # logging, device, results I/O and aggregation
├── scripts/         # AMRFinderPlus baseline, leave-one-species-out, Integrated Gradients
└── requirements.txt
```

## Hardware

Experiments were run on the following machine:

| Component | Specification |
|---|---|
| **CPU** | AMD Ryzen Threadripper PRO 3955WX (16 cores / 32 threads) |
| **RAM** | 64 GB DDR4 |
| **GPU** | NVIDIA RTX A5000 (24 GB VRAM) |
| **OS** | Ubuntu Linux |


