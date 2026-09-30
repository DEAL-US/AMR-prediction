# Additional analyses

Standalone scripts that complement `../main.py`. They share its data loading, model and
training code, and are run from the repository root. As in `main.py`, parameters are set in
the `CONFIG` block at the top of each file, and the main ones can be overridden from the
command line (`--help`). Default paths expect the BioStudies files
([S-BSST2698](https://www.ebi.ac.uk/biostudies/studies/S-BSST2698)) in `data/` and write to `results/`.

| Script | Purpose |
|---|---|
| `amrfinderplus_baseline.py` | Rule-based AMRFinderPlus benchmark, on all isolates and per species |
| `loso_evaluation.py` | Leave-one-species-out cross-species generalisation |
| `integrated_gradients.py` | Non-AMR UniRef50 clusters behind the NDARO + BAKTA50 gain (Integrated Gradients) |

Suggested order: `main.py` → `amrfinderplus_baseline.py` → `loso_evaluation.py` → `integrated_gradients.py`.

## 1. AMRFinderPlus benchmark

An isolate is predicted resistant when it carries at least one AMRFinderPlus-detected gene
(NDARO `g_*` features) that targets the antibiotic's drug class according to the CARD ARO
index (`data/aro_index.tsv`, from [card.mcmaster.ca/download](https://card.mcmaster.ca/download)).
The number of matching genes is the score for PR-AUC and ROC-AUC. The rule is evaluated on all
isolates and, separately, on the isolates of each species listed in `CONFIG["species"]`.

```bash
python scripts/amrfinderplus_baseline.py
```

| File | Content |
|---|---|
| `amrfinder_baseline_per_antibiotic.csv` | one row per (organism, antibiotic): class counts, number of rule genes, F1, PR-AUC, geometric-mean accuracy, accuracy, precision, recall, specificity, ROC-AUC, confusion matrix, and the bootstrap std of each metric (1,000 resamples of the isolates; antibiotics with >= 50 labelled isolates) |
| `amrfinder_baseline_summary.csv` | per organism (`all` and each species): mean and std across the evaluable antibiotics (>= 50 labelled isolates, >= 5 in the minority class, at least one gene mapped to the drug class) |

## 2. Leave-one-species-out (LOSO)

For each held-out species, the models are trained on the isolates of the other species
(90% training / 10% validation, early stopping on validation F1) and tested on every isolate of
the held-out species. Training is identical to `main.py`; only the split changes. ROC-AUC is the
primary metric because the 0.5 threshold learned on other species need not transfer.

```bash
python scripts/loso_evaluation.py
```

| `CONFIG` field | Default | Meaning |
|---|---|---|
| `datasets` | NDARO and the four NDARO + BAKTA combinations | representations to evaluate (NDARO provides the organism) |
| `holdout_species` | the four most represented species | held out one at a time |
| `n_seeds` | `5` | seeds per (dataset, held-out species, antibiotic) |
| `min_minority_test` | `5` | a held-out antibiotic is reliable when its test set has >= 5 minority-class isolates |
| `amrfinder_csv` | output of `amrfinderplus_baseline.py` | adds the rule-based reference, evaluated on the same held-out isolates |

| File | Content |
|---|---|
| `results_detailed.csv` | one row per (dataset, held-out species, antibiotic, seed) |
| `loso_per_antibiotic.csv` | mean over seeds, test-set class counts and reliability flag |
| `loso_per_species.csv` | mean F1 / ROC-AUC / geometric-mean accuracy over reliable antibiotics, per held-out species and dataset |
| `loso_summary.csv` | per held-out species: AMRFinderPlus, NDARO and the best combined representation (highest geometric-mean accuracy) |

## 3. Integrated Gradients

For the antibiotics where NDARO + BAKTA50 outperforms NDARO + BAKTA50 AMR in `main.py`'s
`results.csv`, one NDARO + BAKTA50 model is trained per antibiotic and Integrated Gradients
(zero baseline, 50 steps) are computed for the UniRef50 clusters while each isolate's NDARO
genes are kept at their observed values. Clusters are ranked by mean |IG| over the labelled
isolates, and the non-AMR clusters (absent from BAKTA50 AMR) are aggregated across antibiotics
(mean |IG| over the antibiotics in whose top 100 they appear).

```bash
python scripts/integrated_gradients.py
```

| File | Content |
|---|---|
| `ig/ig_summary_<antibiotic>.csv` | mean \|IG\| and mean signed IG of every UniRef50 cluster (existing files are reused, so runs resume) |
| `ig_selected_antibiotics.csv` | antibiotics analysed and their F1 gain |
| `ig_amr_vs_non_amr.csv` | per antibiotic: top cluster, number of non-AMR clusters in the top 20, largest AMR and non-AMR \|IG\| |
| `non_amr_ig_per_antibiotic.csv` | top-100 non-AMR clusters of every antibiotic |
| `non_amr_ig_combined_summary.csv` | non-AMR clusters ranked across antibiotics |
