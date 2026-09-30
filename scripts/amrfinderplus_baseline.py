"""
Rule-based AMRFinderPlus benchmark
==================================
A simple, rule-based AMRFinderPlus-style predictor that mirrors how the tool is used
in surveillance pipelines:

    predict R  <=>  at least one resistance gene targeting the drug class of
                    the antibiotic is detected in the isolate.

The NDARO ``g_*`` features are AMRFinderPlus detections (NCBI runs AMRFinderPlus on every
assembly), so the rule reduces to OR-ing the ``g_*`` columns of the genes that target the
antibiotic's drug class. Gene-to-drug-class mappings come from the CARD ARO index
(``aro_index.tsv``, https://card.mcmaster.ca/download). The number of matching genes is
used as the score for PR-AUC and ROC-AUC.

The rule is evaluated on every labelled isolate ("all", Table 2) and, separately, on the
isolates of each species in ``CONFIG["species"]`` (per-species and leave-one-species-out
comparisons). Run from the repository root::

    python scripts/amrfinderplus_baseline.py

Outputs (in ``output_dir``):
    amrfinder_baseline_per_antibiotic.csv  one row per (organism, antibiotic): metrics and
                                           bootstrap std (1,000 resamples of the isolates)
    amrfinder_baseline_summary.csv         per organism: mean and std across the evaluable
                                           antibiotics (>= min_samples labelled isolates,
                                           >= min_minority in the minority class, >= 1 rule gene)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import load_ndaro_csv  # noqa: E402

# ===========================================================================
#  CONFIGURATION - edit the values below to match your setup
# ===========================================================================
CONFIG = {
    "ndaro_csv": "data/ndaro_baseline.csv",
    "aro_path": "data/aro_index.tsv",          # CARD ARO index
    "output_dir": "results/amrfinderplus",
    # Species evaluated separately (in addition to all isolates); names as in `organism`.
    "species": ["Salmonella enterica", "E.coli and Shigella",
                "Campylobacter jejuni", "Acinetobacter baumannii"],
    "min_samples": 50,
    "min_minority": 5,
    "bootstrap": 1000,
}
# ===========================================================================

METRICS = ["f1", "pr_auc", "gmean", "accuracy", "precision", "recall", "specificity", "roc_auc"]

# Manual antibiotic -> CARD/ARO drug-class mapping.
# Drug-class strings match those used in `aro_index.tsv` (Drug Class column).
ANTIBIOTIC_CLASS_MAP: dict[str, list[str]] = {
    # Penicillins
    "ampicillin": ["penam"],
    "ampicillin-sulbactam": ["penam"],
    "amoxicillin": ["penam"],
    "amoxicillin-clavulanic acid": ["penam"],
    "penicillin": ["penam"],
    "oxacillin": ["penam"],
    "piperacillin": ["penam"],
    "piperacillin-tazobactam": ["penam"],
    "ticarcillin": ["penam"],
    "ticarcillin-clavulanic acid": ["penam"],
    # Cephalosporins
    "cefazolin": ["cephalosporin"],
    "cefuroxime": ["cephalosporin"],
    "cefoxitin": ["cephalosporin", "cephamycin"],
    "cefotaxime": ["cephalosporin"],
    "ceftriaxone": ["cephalosporin"],
    "ceftazidime": ["cephalosporin"],
    "cefepime": ["cephalosporin"],
    "ceftaroline": ["cephalosporin"],
    "cefiderocol": ["cephalosporin"],
    "cefpodoxime": ["cephalosporin"],
    "cefotaxime-clavulanic acid": ["cephalosporin"],
    "ceftazidime-clavulanic acid": ["cephalosporin"],
    "ceftiofur": ["cephalosporin"],
    "cefquinome": ["cephalosporin"],
    # Carbapenems
    "imipenem": ["carbapenem"],
    "meropenem": ["carbapenem"],
    "ertapenem": ["carbapenem"],
    "doripenem": ["carbapenem"],
    # Monobactams
    "aztreonam": ["monobactam"],
    "aztreonam-avibactam": ["monobactam"],
    # Aminoglycosides
    "gentamicin": ["aminoglycoside antibiotic"],
    "tobramycin": ["aminoglycoside antibiotic"],
    "amikacin": ["aminoglycoside antibiotic"],
    "streptomycin": ["aminoglycoside antibiotic"],
    "kanamycin": ["aminoglycoside antibiotic"],
    "neomycin": ["aminoglycoside antibiotic"],
    "netilmicin": ["aminoglycoside antibiotic"],
    "spectinomycin": ["aminoglycoside antibiotic"],
    # Fluoroquinolones / quinolones
    "ciprofloxacin": ["fluoroquinolone antibiotic"],
    "levofloxacin": ["fluoroquinolone antibiotic"],
    "moxifloxacin": ["fluoroquinolone antibiotic"],
    "ofloxacin": ["fluoroquinolone antibiotic"],
    "norfloxacin": ["fluoroquinolone antibiotic"],
    "enrofloxacin": ["fluoroquinolone antibiotic"],
    "danofloxacin": ["fluoroquinolone antibiotic"],
    "nalidixic acid": ["fluoroquinolone antibiotic"],
    "marbofloxacin": ["fluoroquinolone antibiotic"],
    "pefloxacin": ["fluoroquinolone antibiotic"],
    "delafloxacin": ["fluoroquinolone antibiotic"],
    # Macrolides / lincosamides / streptogramins
    "erythromycin": ["macrolide antibiotic"],
    "azithromycin": ["macrolide antibiotic"],
    "clarithromycin": ["macrolide antibiotic"],
    "telithromycin": ["macrolide antibiotic"],
    "tylosin": ["macrolide antibiotic"],
    "tilmicosin": ["macrolide antibiotic"],
    "clindamycin": ["lincosamide antibiotic"],
    "lincomycin": ["lincosamide antibiotic"],
    "pristinamycin": ["streptogramin antibiotic"],
    "quinupristin-dalfopristin": ["streptogramin antibiotic"],
    # Tetracyclines
    "tetracycline": ["tetracycline antibiotic"],
    "doxycycline": ["tetracycline antibiotic"],
    "minocycline": ["tetracycline antibiotic"],
    "oxytetracycline": ["tetracycline antibiotic"],
    "chlortetracycline": ["tetracycline antibiotic"],
    "tigecycline": ["tetracycline antibiotic", "glycylcycline antibiotic"],
    "eravacycline": ["tetracycline antibiotic"],
    "omadacycline": ["tetracycline antibiotic"],
    # Sulphonamides, trimethoprim and folate antagonists
    "trimethoprim": ["diaminopyrimidine antibiotic"],
    "sulfamethoxazole": ["sulfonamide antibiotic"],
    "sulfisoxazole": ["sulfonamide antibiotic"],
    "trimethoprim-sulfamethoxazole": [
        "diaminopyrimidine antibiotic", "sulfonamide antibiotic"
    ],
    # Phenicols
    "chloramphenicol": ["phenicol antibiotic"],
    "florfenicol": ["phenicol antibiotic"],
    # Glycopeptides
    "vancomycin": ["glycopeptide antibiotic"],
    "teicoplanin": ["glycopeptide antibiotic"],
    # Lipopeptides / polypeptides
    "daptomycin": ["lipopeptide antibiotic"],
    "colistin": ["peptide antibiotic"],
    "polymyxin": ["peptide antibiotic"],
    "polymyxin b": ["peptide antibiotic"],
    # Oxazolidinones
    "linezolid": ["oxazolidinone antibiotic"],
    "tedizolid": ["oxazolidinone antibiotic"],
    # Fosfomycin / nitrofurans / nitroimidazoles / rifamycins
    "fosfomycin": ["fosfomycin"],
    "nitrofurantoin": ["nitrofuran antibiotic"],
    "metronidazole": ["nitroimidazole antibiotic"],
    "rifampin": ["rifamycin antibiotic"],
    "rifampicin": ["rifamycin antibiotic"],
    # Others
    "fusidic acid": ["fusidane antibiotic"],
    "mupirocin": ["mupirocin"],
    "bacitracin": ["peptide antibiotic"],
    "novobiocin": ["aminocoumarin antibiotic"],
    # Combination drugs
    "ceftolozane-tazobactam": ["cephalosporin"],
    "ceftazidime-avibactam": ["cephalosporin"],
    "meropenem-vaborbactam": ["carbapenem"],
    "imipenem-relebactam": ["carbapenem"],
}


def normalise_token(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\s_]+", "", s)
    return s


def gene_to_classes(aro_path: str) -> dict[str, set[str]]:
    """
    Map gene 'token' (lower-case, no parens or hyphens) to the set of drug
    classes it confers resistance to, per CARD ARO.
    """
    aro = pd.read_csv(aro_path, sep="\t")
    out: dict[str, set[str]] = {}
    for _, row in aro.iterrows():
        names = []
        for col in ["Model Name", "ARO Name", "CARD Short Name"]:
            val = row.get(col)
            if isinstance(val, str) and val:
                names.append(val)
        drugs = row.get("Drug Class", "")
        if not isinstance(drugs, str):
            continue
        drug_set = {d.strip() for d in drugs.split(";") if d.strip()}
        for n in names:
            tok = normalise_token(n)
            out.setdefault(tok, set()).update(drug_set)
            # also strip parens etc.
            stripped = re.sub(r"[^a-z0-9]", "", tok)
            out.setdefault(stripped, set()).update(drug_set)
    return out


def match_gene(gene_col: str, gene2class: dict[str, set[str]]) -> set[str]:
    """gene_col looks like 'g_blaKPC-11'; return its drug-class set or empty."""
    gene = gene_col[2:] if gene_col.startswith("g_") else gene_col
    tok = normalise_token(gene)
    if tok in gene2class:
        return gene2class[tok]
    # try without parenthesised qualifiers and without trailing -allele numbers
    no_alleles = re.sub(r"-?\d+$", "", tok)
    if no_alleles and no_alleles in gene2class:
        return gene2class[no_alleles]
    # Strip parens/punct
    plain = re.sub(r"[^a-z0-9]", "", tok)
    if plain in gene2class:
        return gene2class[plain]
    # Family-level partial match: try the longest prefix that matches a known key
    for cut in range(len(tok), 2, -1):
        if tok[:cut] in gene2class:
            return gene2class[tok[:cut]]
    return set()



def bootstrap_std(y_true, y_pred, y_score, B: int = 1000, seed: int = 0) -> dict:
    """Std of each metric under bootstrap resampling of the isolates.
    The rule predictions are fixed; this captures the sampling uncertainty of
    the metric estimate on a finite set of isolates."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.RandomState(seed)
    vals = {k: [] for k in METRICS}
    has_score = y_score is not None
    for _ in range(B):
        idx = rng.randint(0, n, n)
        yt = y_true[idx]; yp = y_pred[idx]
        tp = np.sum((yt == 1) & (yp == 1)); fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0)); tn = np.sum((yt == 0) & (yp == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        vals["f1"].append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
        vals["precision"].append(prec); vals["recall"].append(rec); vals["specificity"].append(spec)
        vals["gmean"].append(np.sqrt(rec * spec)); vals["accuracy"].append((tp + tn) / n)
        if has_score and 0 < yt.sum() < n:
            vals["pr_auc"].append(average_precision_score(yt, y_score[idx]))
            vals["roc_auc"].append(roc_auc_score(yt, y_score[idx]))
    return {f"{k}_std": float(np.std(v)) if v else float("nan") for k, v in vals.items()}


def evaluate_rule(nd, rows: np.ndarray, col_classes: list, organism: str, cfg: dict) -> list:
    """Per-antibiotic metrics of the rule on the isolates selected by ``rows``."""
    out = []
    for j, a_col in enumerate(tqdm(nd.label_names, desc=organism[:25], leave=False)):
        ab = a_col[2:]
        y_all = nd.labels[rows, j]
        labelled = y_all >= 0
        n_total = int(labelled.sum())
        if n_total < 20:
            continue
        y_true = y_all[labelled].astype(int)
        n_pos = int(y_true.sum())
        target_classes = set(ANTIBIOTIC_CLASS_MAP.get(ab.lower(), []))
        relevant = [i for i, s in enumerate(col_classes) if s & target_classes]
        if relevant:
            X = nd.features[np.flatnonzero(rows)[labelled]][:, relevant]
            y_score = np.asarray(X.sum(axis=1)).ravel().astype(float)
            y_pred = (y_score > 0).astype(int)
        else:                       # no gene targets the drug class: the rule predicts S
            y_score, y_pred = None, np.zeros_like(y_true)
        both = 0 < n_pos < n_total
        tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
        rec = recall_score(y_true, y_pred, zero_division=0)
        spec = tn / (tn + fp) if tn + fp > 0 else float("nan")
        row = {
            "organism": organism, "antibiotic": ab,
            "n_total": n_total, "n_pos": n_pos, "n_neg": n_total - n_pos,
            "minority": min(n_pos, n_total - n_pos),
            "n_rule_genes": len(relevant),
            "drug_classes": ";".join(sorted(target_classes)),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "pr_auc": average_precision_score(y_true, y_score) if relevant and both else float("nan"),
            "gmean": float(np.sqrt(rec * spec)) if not np.isnan(spec) else float("nan"),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": rec,
            "specificity": spec,
            "roc_auc": roc_auc_score(y_true, y_score) if relevant and both else float("nan"),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        }
        if n_total >= cfg["min_samples"]:
            row.update(bootstrap_std(y_true, y_pred, y_score, B=cfg["bootstrap"], seed=0))
        out.append(row)
    return out


def main(cfg: dict) -> None:
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    g2c = gene_to_classes(cfg["aro_path"])
    print(f"{len(g2c)} normalised gene tokens with at least one drug class (CARD ARO)")

    nd = load_ndaro_csv(Path(cfg["ndaro_csv"]))
    col_classes = [match_gene(g, g2c) for g in nd.feature_names]
    unmatched = sum(1 for s in col_classes if not s)
    print(f"{len(nd.assemblies)} assemblies | {len(nd.feature_names)} genes "
          f"({unmatched} without a drug class) | {len(nd.label_names)} antibiotics")

    organism = np.asarray(nd.organism, dtype=object)
    rows = evaluate_rule(nd, np.ones(len(organism), dtype=bool), col_classes, "all", cfg)
    for species in cfg["species"]:
        mask = organism == species
        if not mask.any():
            print(f"No isolates for species {species!r}; skipped")
            continue
        rows += evaluate_rule(nd, mask, col_classes, species, cfg)

    per_ab = pd.DataFrame(rows).sort_values(["organism", "n_total"], ascending=[True, False])
    per_ab.to_csv(out_dir / "amrfinder_baseline_per_antibiotic.csv", index=False)

    evaluable = per_ab[(per_ab.n_total >= cfg["min_samples"]) & (per_ab.minority >= cfg["min_minority"])
                       & (per_ab.n_rule_genes > 0)]
    summary = evaluable.groupby("organism", sort=False).agg(
        n_antibiotics=("antibiotic", "nunique"),
        **{f"{m}_mean": (m, "mean") for m in METRICS},
        **{f"{m}_std": (m, "std") for m in METRICS},
    ).reset_index()
    summary.to_csv(out_dir / "amrfinder_baseline_summary.csv", index=False)
    print(summary[["organism", "n_antibiotics"] + [f"{m}_mean" for m in METRICS]].round(3).to_string(index=False))
    print(f"Saved {out_dir / 'amrfinder_baseline_per_antibiotic.csv'} and {out_dir / 'amrfinder_baseline_summary.csv'}")


if __name__ == "__main__":
    main(CONFIG)
