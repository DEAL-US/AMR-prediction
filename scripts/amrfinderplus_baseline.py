"""
Rule-based AMRFinderPlus benchmark
==================================
We build a simple, rule-based AMRFinderPlus-style predictor that mirrors how
the tool is actually used in surveillance pipelines:

    predict R  <=>  at least one resistance gene targeting the drug class of
                    the antibiotic is detected in the isolate.

The NDARO `g_*` features are precisely AMRFinder annotations (NCBI's pipeline
runs AMRFinder against every genome), so the rule reduces to ORing those
`g_*` columns for the genes that target the antibiotic's drug class.
Gene-to-drug-class mappings come from the CARD ARO index (`aro_index.tsv`),
which uses the same standardised ontology as AMRFinder. The ARO index can be
downloaded from the CARD database (https://card.mcmaster.ca/download).

Outputs (written under --out):
  - amrfinder_baseline_per_antibiotic.csv   (per-antibiotic F1/precision/recall/PR-AUC/ROC-AUC)
  - amrfinder_baseline_summary.json         (aggregate metrics over evaluable antibiotics)
"""
from __future__ import annotations
import json, re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    average_precision_score, roc_auc_score,
)

# ===========================================================================
#  CONFIGURATION – edit the values below to match your setup
# ===========================================================================
CONFIG = {
    # NDARO CSV (columns: assembly, organism, g_* features, a_* labels);
    # the `processed.csv` described in the repository README.
    "ndaro_path": "data/processed.csv",
    # CARD ARO index TSV (https://card.mcmaster.ca/download).
    "aro_path": "data/aro_index.tsv",
    # Output directory.
    "output_dir": "results/amrfinderplus",
}
# ===========================================================================

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
    """Std of each metric under bootstrap resampling of the test isolates.
    The rule predictions are fixed; this captures the sampling uncertainty of
    the metric estimate on a finite test set."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.RandomState(seed)
    f1s, accs, precs, recs, prs, rocs = [], [], [], [], [], []
    has_score = y_score is not None and not np.all(np.isnan(np.asarray(y_score, dtype=float)))
    ys_full = np.asarray(y_score, dtype=float) if has_score else None
    for _ in range(B):
        idx = rng.randint(0, n, n)
        yt = y_true[idx]; yp = y_pred[idx]
        tp = np.sum((yt == 1) & (yp == 1)); fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0)); tn = np.sum((yt == 0) & (yp == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1); precs.append(prec); recs.append(rec); accs.append((tp + tn) / n)
        if has_score and 0 < yt.sum() < n:
            ys = ys_full[idx]
            try: prs.append(average_precision_score(yt, ys)); rocs.append(roc_auc_score(yt, ys))
            except ValueError: pass
    sd = lambda a: float(np.std(a)) if len(a) else float("nan")
    return {"f1_std": sd(f1s), "accuracy_std": sd(accs), "precision_std": sd(precs),
            "recall_std": sd(recs), "pr_auc_std": sd(prs), "roc_auc_std": sd(rocs)}


def main(ndaro_csv: str, aro_path: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading gene -> drug-class table from CARD ARO index...")
    g2c = gene_to_classes(aro_path)
    print(f"  {len(g2c)} normalised gene tokens with at least one drug class")

    print("Loading NDARO (memory-efficient: only assembly/organism/g_*/a_* columns)...")
    # First pass: just get column names so we can filter
    header = pd.read_csv(ndaro_csv, nrows=0).columns
    keep = [c for c in header if c in ("assembly", "organism") or c.startswith("g_") or c.startswith("a_")]
    # Read in chunks of 100k rows to avoid OOM (file has ~660k rows but most are duplicates per assembly)
    chunks = []
    seen_assemblies: set[str] = set()
    g_cols_template = [c for c in keep if c.startswith("g_")]
    dtype_map = {c: "Int8" for c in g_cols_template}
    for ch in pd.read_csv(ndaro_csv, usecols=keep, chunksize=100_000, dtype=dtype_map):
        # Drop assemblies already seen in earlier chunks
        new_mask = ~ch["assembly"].isin(seen_assemblies)
        ch = ch[new_mask].drop_duplicates(subset=["assembly"])
        seen_assemblies.update(ch["assembly"].tolist())
        chunks.append(ch)
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    g_cols = [c for c in df.columns if c.startswith("g_")]
    a_cols = [c for c in df.columns if c.startswith("a_")]
    print(f"  {len(df)} unique assemblies | {len(g_cols)} g_* genes | {len(a_cols)} antibiotics")

    # Map every gene column to its drug-class set
    col_classes: dict[str, set[str]] = {c: match_gene(c, g2c) for c in g_cols}
    unmatched = [c for c, s in col_classes.items() if not s]
    print(f"  unmatched genes (no drug class in ARO): {len(unmatched)} / {len(g_cols)}")

    # For every antibiotic, build the rule:
    rows = []
    for c in a_cols:
        ab = c[2:]
        labelled = df[df[c].isin(["S", "R"])]
        n_total = len(labelled)
        if n_total < 20:
            continue
        y_true = (labelled[c] == "R").astype(int).to_numpy()
        # Determine which g_* columns target this antibiotic
        target_classes = set(ANTIBIOTIC_CLASS_MAP.get(ab.lower(), []))
        relevant_cols = [g for g in g_cols if col_classes[g] & target_classes]
        if not relevant_cols:
            # No mapping available - rule predicts S for all
            y_pred = np.zeros_like(y_true)
            n_rules = 0
        else:
            n_rules = len(relevant_cols)
            X = labelled[relevant_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
            y_pred = (X.sum(axis=1) > 0).astype(int)
        # Probability-like score: number of matching genes (used for PR/ROC AUC)
        if relevant_cols:
            X = labelled[relevant_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
            y_score = X.sum(axis=1).astype(float)
            try:
                pr_auc = average_precision_score(y_true, y_score) if y_true.sum() and y_true.sum() < len(y_true) else float("nan")
            except ValueError:
                pr_auc = float("nan")
            try:
                roc_auc = roc_auc_score(y_true, y_score) if y_true.sum() and y_true.sum() < len(y_true) else float("nan")
            except ValueError:
                roc_auc = float("nan")
        else:
            y_score = None
            pr_auc = float("nan")
            roc_auc = float("nan")
        # The rule is deterministic, so the STD of each metric is the sampling
        # uncertainty of the estimate on this finite test set: we obtain it by
        # bootstrapping the test isolates (1000 resamples).
        bstd = bootstrap_std(y_true, y_pred, y_score, B=1000, seed=0)
        rows.append({
            "antibiotic": ab,
            "n_total": n_total,
            "n_pos": int(y_true.sum()),
            "n_neg": int(n_total - y_true.sum()),
            "n_rule_genes": n_rules,
            "drug_classes": ";".join(sorted(target_classes)) if target_classes else "",
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "f1_std": bstd["f1_std"],
            "accuracy": accuracy_score(y_true, y_pred),
            "accuracy_std": bstd["accuracy_std"],
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "precision_std": bstd["precision_std"],
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "recall_std": bstd["recall_std"],
            "pr_auc": pr_auc,
            "pr_auc_std": bstd["pr_auc_std"],
            "roc_auc": roc_auc,
            "roc_auc_std": bstd["roc_auc_std"],
        })

    out = pd.DataFrame(rows).sort_values("n_total", ascending=False).reset_index(drop=True)
    out.to_csv(out_dir / "amrfinder_baseline_per_antibiotic.csv", index=False)

    # Restrict to "evaluable" antibiotics (>=50 samples, has rules)
    evaluable = out[(out["n_total"] >= 50) & (out["n_rule_genes"] > 0)]
    summary = {
        "n_antibiotics_with_rules_and_50plus_samples": len(evaluable),
        "mean_f1_amrfinder_rule": float(evaluable["f1"].mean()),
        "median_f1_amrfinder_rule": float(evaluable["f1"].median()),
        "mean_precision": float(evaluable["precision"].mean()),
        "mean_recall": float(evaluable["recall"].mean()),
        "mean_pr_auc": float(evaluable["pr_auc"].mean(skipna=True)),
        "mean_roc_auc": float(evaluable["roc_auc"].mean(skipna=True)),
        # STD of each metric ACROSS antibiotics (heterogeneity of the rule baseline)
        "std_f1_across_antibiotics": float(evaluable["f1"].std()),
        "std_accuracy_across_antibiotics": float(evaluable["accuracy"].std()),
        "std_precision_across_antibiotics": float(evaluable["precision"].std()),
        "std_recall_across_antibiotics": float(evaluable["recall"].std()),
        "std_pr_auc_across_antibiotics": float(evaluable["pr_auc"].std(skipna=True)),
        "std_roc_auc_across_antibiotics": float(evaluable["roc_auc"].std(skipna=True)),
        # mean WITHIN-antibiotic bootstrap STD (sampling uncertainty of the estimate)
        "mean_bootstrap_std_f1": float(evaluable["f1_std"].mean()),
        "mean_bootstrap_std_pr_auc": float(evaluable["pr_auc_std"].mean(skipna=True)),
        "mean_bootstrap_std_roc_auc": float(evaluable["roc_auc_std"].mean(skipna=True)),
        "n_antibiotics_no_known_genes": int((out["n_rule_genes"] == 0).sum()),
        "n_antibiotics_with_drug_class_mapped": int((out["drug_classes"] != "").sum()),
    }
    (out_dir / "amrfinder_baseline_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Saved {out_dir / 'amrfinder_baseline_per_antibiotic.csv'}")


if __name__ == "__main__":
    main(CONFIG["ndaro_path"], CONFIG["aro_path"], Path(CONFIG["output_dir"]))
