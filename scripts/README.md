# Additional analyses

Standalone scripts that complement the main pipeline (`../main.py`). They are
not required to reproduce the primary results.

| Script | Purpose |
|---|---|
| `amrfinderplus_baseline.py` | Rule-based AMRFinderPlus benchmark |
| `loso_evaluation.py` | Leave-one-species-out cross-species generalisation |

As in `../main.py`, parameters are set directly in the script: edit the
`CONFIG` block at the top of each file, then run it with no arguments. Both
scripts use only the dependencies already listed in `../requirements.txt`.

## Data

The scripts read the same inputs described in the main README:

- **`processed.csv`** – the NDARO isolate table (`assembly`, `organism`, `g_*`,
  `a_*` columns), available on [BioStudies (S-BSST2698)](https://www.ebi.ac.uk/biostudies/studies/S-BSST2698).
- **`aro_index.tsv`** – the CARD ARO index used to map genes to drug classes,
  downloadable from the [CARD database](https://card.mcmaster.ca/download).
- **BAKTA `.npz` matrices** (`bakta50.npz`, `bakta50_amr.npz`, `bakta90.npz`,
  `bakta90_amr.npz`) and their `_columns.pkl` / `_assemblies.pkl` companions,
  also on BioStudies.

The default `CONFIG` paths assume these files live under `data/` and write
outputs to `results/`.

## 1. AMRFinderPlus benchmark

Predicts resistance whenever the isolate carries at least one detected AMR gene
targeting the antibiotic's drug class, then scores that rule against the
phenotypic labels per antibiotic.

```bash
# edit CONFIG (ndaro_path, aro_path, output_dir) at the top of the script, then:
python scripts/amrfinderplus_baseline.py
```

Outputs `amrfinder_baseline_per_antibiotic.csv` (per-antibiotic
F1/precision/recall/PR-AUC/ROC-AUC) and `amrfinder_baseline_summary.json`.

## 2. Leave-one-species-out (LOSO) evaluation

For each held-out species the model is trained on all remaining species and
evaluated on the excluded one, with ROC-AUC as the primary metric. Training
uses the same protocol as the main pipeline (early stopping, patience 10, max
200 epochs). The NDARO CSV is converted to a sparse matrix in memory, so there
is no separate preparation step.

```bash
# edit CONFIG at the top of the script, then:
python scripts/loso_evaluation.py
```

Key `CONFIG` fields:

| Field | Default | Meaning |
|---|---|---|
| `ndaro_csv` | `data/processed.csv` | NDARO CSV (loaded & sparsified in memory) |
| `bakta_dir` | `data` | directory holding the BAKTA `.npz` matrices |
| `datasets` | NDARO + the four NDARO+BAKTA combinations | representations to evaluate |
| `holdout_species` | the four most-represented species | held out one at a time (must match the `organism` strings) |
| `seeds` | `5` | independent seeds per (dataset, antibiotic, held-out species) |
| `amrfinder_csv` | `None` | optional path to `amrfinder_baseline_per_antibiotic.csv`; when set, adds a rule-based AMRFinderPlus reference column to the summary |

Results are written to four CSVs under `output_dir`:

| File | Content |
|---|---|
| `results_detailed.csv` | one row per (dataset, antibiotic, held-out species, seed) |
| `results_aggregated.csv` | mean ± std across seeds |
| `loso_per_species.csv` | mean over reliable antibiotics, per held-out species × dataset |
| `loso_summary.csv` | compact view: NDARO vs the best combined representation (highest ROC-AUC) per held-out species |

For the last two, a (held-out species, antibiotic) fold is kept only when its
test set has at least `min_minority_test` (default 5) minority-class isolates,
and metrics are then averaged over those reliable antibiotics. If `amrfinder_csv`
points to the output of `amrfinderplus_baseline.py`, its F1/ROC-AUC are averaged
over each species' reliable antibiotics and added to `loso_summary.csv` as a
rule-based reference (the rule is deterministic, so no per-species training).
