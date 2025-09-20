"""
AlphaMissense (AM) exon aggregation → patient × gene matrices
-------------------------------------------------------------
Parses VEP output with AlphaMissense annotations and aggregates per-variant AM
metrics into per-patient, per-gene matrices 

What it does
------------
1) Loads:
   - VEP result (tab-delimited) with the 'Extra' column and an expanded table that
     includes AM fields (`am_class`, `am_pathogenicity`) and a stable gene id.
   - Patient to variants dictionary from Step 2:
       {patient_id: {"11":[variant_ids], "10":[variant_ids]}}
   - Gene map CSV with columns: `gene_id`, `gene_name`.

2) Normalizes / prepares the VEP table:
   - Expands 'Extra' (semi-colon key=value) into columns.
   - Merges the expanded AM table to add `gene_name` from `gene_id`.
   - Keeps the AM fields and de-duplicates rows by (SNP, gene_name).
   - Maps AM class to an integer score (fixed):
       benign to 1, ambiguous to 2, pathogenic to 3

3) Aggregates by patient and zygosity (separately for "11" and "10"):
   - AM_class_INT: max, sum, mean
   - am_pathogenicity: max, sum
   - counts: number of variant records per gene

   Output matrices (genes × patients):
     AM_max11, AM_sum11, AM_mean11, AM_counts11,
     AM_scores_sum11, AM_scores_max11,
     AM_max10, AM_sum10, AM_mean10, AM_counts10,
     AM_scores_sum10, AM_scores_max10

Required inputs (CLI)
---------------------
--vep_table            Path to the VEP output TSV (with 'Extra' column).
--am_expanded_csv      Path to the AM-expanded CSV derived from the same VEP run
                       (must include columns: '#Uploaded_variation' (as SNP),
                       'Gene' (as gene_id), 'am_class', 'am_pathogenicity').
--gene_map_csv         CSV with columns: gene_id, gene_name.
--patient_dict_pkl     Pickle from Step 2 containing patient→variants dict.
--out_pickle           Output pickle path for the AM matrices (will not overwrite).


"""


import os
import argparse
import pickle
import numpy as np
import pandas as pd

AM_CLASS_MAP = {"benign": 1, "ambiguous": 2, "pathogenic": 3}

def normalize_scores_table(scores_csv: str, gene_map_csv: str | None) -> pd.DataFrame:
    """
    Load the expanded VEP/AlphaMissense CSV and normalize column names so that:
      - SNP column is named 'SNP'
      - gene id column is 'gene_id'
      - gene symbol column is 'gene_name' (via merge if needed)
      - 'am_class' and 'am_pathogenicity' are present
    """
    df = pd.read_csv(scores_csv)
    
    # Accept alternative headers from prior runs
    if "#Uploaded_variation" in df.columns and "SNP" not in df.columns:
        df = df.rename(columns={"#Uploaded_variation": "SNP"})
    if "Gene" in df.columns and "gene_id" not in df.columns:
        df = df.rename(columns={"Gene": "gene_id"})

    # If gene_name missing and a map provided, merge it in
    if "gene_name" not in df.columns:
        if gene_map_csv is None:
            raise ValueError("gene_name not in scores and no --gene-map provided.")
        gene_map = pd.read_csv(gene_map_csv, usecols=["gene_name", "gene_id"])
        df = df.merge(gene_map, on="gene_id", how="inner")

    # Keep only the needed columns
    cols_needed = ["SNP", "gene_id", "Consequence", "am_class", "am_pathogenicity", "gene_name"]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in scores CSV: {missing}")
    df = df[cols_needed].dropna()

    # Map am_class to integer
    df["AM_class_INT"] = df["am_class"].replace(AM_CLASS_MAP)

    return df


def format_variant_ids(variant_ids: list[str], mode: str) -> list[str]:
    """
    Different ID handling from GTEx/CureGN previous steps: #TODO: unify
      - mode='gtex': join parts 0,1,3 of underscore-split ID
      - mode='curegn': use IDs as-is
    """
    if mode == "gtex":
        print("Warning: Reformatting variant IDs from GTEx style to match SNP column.")
        out = []
        for item in variant_ids:
            parts = item.split("_")
            # join indices 0,1,3 when present (silently skip if short)
            keep = []
            for i in (0, 1, 3):
                if i < len(parts):
                    keep.append(parts[i])
            out.append("_".join(keep))
        return out
    elif mode == "curegn":
        return variant_ids
    else:
        raise ValueError("Unknown --id-format (use 'gtex' or 'curegn').")


def build_matrices(scores_am: pd.DataFrame, patients_exon_dict: dict, id_format: str) -> dict:
    """
    Reproduce aggregations for 11/10:
      - AM_class_INT: max, sum, mean
      - am_pathogenicity: sum, max
      - counts: gene_name value_counts
      after drop_duplicates(['SNP','gene_name'])
    """
    pat_dict_max11, pat_dict_sum11, pat_dict_mean11, pat_dict_count11 = {}, {}, {}, {}
    pat_dict_scores_sum11, pat_dict_scores_max11 = {}, {}
    pat_dict_max10, pat_dict_sum10, pat_dict_mean10, pat_dict_count10 = {}, {}, {}, {}
    pat_dict_scores_sum10, pat_dict_scores_max10 = {}, {}

    pats = list(patients_exon_dict.keys())
    set_snp = set(scores_am["SNP"])

    for i, p in enumerate(pats):
        if i % 200 == 0:
            print(i)

        # init patient buckets
        pat_dict_max11[p] = {}
        pat_dict_sum11[p] = {}
        pat_dict_mean11[p] = {}
        pat_dict_count11[p] = {}
        pat_dict_scores_sum11[p] = {}
        pat_dict_scores_max11[p] = {}
        pat_dict_max10[p] = {}
        pat_dict_sum10[p] = {}
        pat_dict_mean10[p] = {}
        pat_dict_count10[p] = {}
        pat_dict_scores_sum10[p] = {}
        pat_dict_scores_max10[p] = {}

        if ("11" not in patients_exon_dict.get(p, {})) and ("10" not in patients_exon_dict.get(p, {})):
            continue

        # 11
        if "11" in patients_exon_dict.get(p, {}):
            vlist11_raw = patients_exon_dict[p]["11"]
            vlist11 = format_variant_ids(vlist11_raw, id_format)
            vlist11 = [v for v in vlist11 if v in set_snp]
            if vlist11:
                snp_pat11 = scores_am[scores_am["SNP"].isin(vlist11)].drop_duplicates(subset=["SNP", "gene_name"])
                grp11 = snp_pat11.groupby("gene_name")
                pat_dict_max11[p] = grp11["AM_class_INT"].max().to_dict()
                pat_dict_sum11[p] = grp11["AM_class_INT"].sum().to_dict()
                pat_dict_mean11[p] = grp11["AM_class_INT"].mean().to_dict()
                pat_dict_scores_sum11[p] = grp11["am_pathogenicity"].sum().to_dict()
                pat_dict_scores_max11[p] = grp11["am_pathogenicity"].max().to_dict()
                pat_dict_count11[p] = snp_pat11["gene_name"].value_counts().to_dict()

        # 10
        if "10" in patients_exon_dict.get(p, {}):
            vlist10_raw = patients_exon_dict[p]["10"]
            vlist10 = format_variant_ids(vlist10_raw, id_format)
            vlist10 = [v for v in vlist10 if v in set_snp]
            if vlist10:
                snp_pat10 = scores_am[scores_am["SNP"].isin(vlist10)].drop_duplicates(subset=["SNP", "gene_name"])
                grp10 = snp_pat10.groupby("gene_name")
                pat_dict_max10[p] = grp10["AM_class_INT"].max().to_dict()
                pat_dict_sum10[p] = grp10["AM_class_INT"].sum().to_dict()
                pat_dict_mean10[p] = grp10["AM_class_INT"].mean().to_dict()
                pat_dict_scores_sum10[p] = grp10["am_pathogenicity"].sum().to_dict()
                pat_dict_scores_max10[p] = grp10["am_pathogenicity"].max().to_dict()
                pat_dict_count10[p] = snp_pat10["gene_name"].value_counts().to_dict()

    # Pack DataFrames
    AM_max11 = pd.DataFrame(pat_dict_max11).fillna(0)
    AM_sum11 = pd.DataFrame(pat_dict_sum11).fillna(0)
    AM_mean11 = pd.DataFrame(pat_dict_mean11).fillna(0)
    AM_counts11 = pd.DataFrame(pat_dict_count11).fillna(0)
    AM_scores_sum11 = pd.DataFrame(pat_dict_scores_sum11).fillna(0)
    AM_scores_max11 = pd.DataFrame(pat_dict_scores_max11).fillna(0)

    AM_max10 = pd.DataFrame(pat_dict_max10).fillna(0)
    AM_sum10 = pd.DataFrame(pat_dict_sum10).fillna(0)
    AM_mean10 = pd.DataFrame(pat_dict_mean10).fillna(0)
    AM_counts10 = pd.DataFrame(pat_dict_count10).fillna(0)
    AM_scores_sum10 = pd.DataFrame(pat_dict_scores_sum10).fillna(0)
    AM_scores_max10 = pd.DataFrame(pat_dict_scores_max10).fillna(0)

    matrices = {
        "AM_max11": AM_max11,
        "AM_sum11": AM_sum11,
        "AM_mean11": AM_mean11,
        "AM_counts11": AM_counts11,
        "AM_scores_sum11": AM_scores_sum11,
        "AM_scores_max11": AM_scores_max11,
        "AM_max10": AM_max10,
        "AM_sum10": AM_sum10,
        "AM_mean10": AM_mean10,
        "AM_counts10": AM_counts10,
        "AM_scores_sum10": AM_scores_sum10,
        "AM_scores_max10": AM_scores_max10,
    }
    return matrices


def main():
    ap = argparse.ArgumentParser(
        description="Build AlphaMissense gene-level matrices (11/10) from VEP-expanded AM scores and patient→variants dict."
    )
    ap.add_argument("--scores-csv", required=True, help="Expanded VEP/AlphaMissense CSV.")
    ap.add_argument("--patient-dict", required=True, help="Pickle of patient→variants (keys '11','10').")
    ap.add_argument("--out-pkl", required=True, help="Output pickle path for AM matrices (no overwrite).")
    ap.add_argument(
        "--id-format",
        required=True,
        choices=["gtex", "curegn"],
        help="Match variant ID formatting used in the patient dict to the scores CSV ('gtex' reformats IDs; 'curegn' uses as-is).",
    )
    ap.add_argument(
        "--gene-map",
        default=None,
        help="Optional CSV with columns ['gene_id','gene_name'] if scores lack gene_name.",
    )
    args = ap.parse_args()

    # Inputs
    scores_am = normalize_scores_table(args.scores_csv, args.gene_map)

    with open(args.patient_dict, "rb") as f:
        patients_exon_dict = pickle.load(f)

    # Build
    matrices = build_matrices(scores_am, patients_exon_dict, args.id_format)

    if not os.path.exists(args.out_pkl):
        with open(args.out_pkl, "wb") as fh:
            pickle.dump(matrices, fh)
        print(f"Saved: {args.out_pkl}")
    else:
        print(f"[skip] Exists, not overwriting: {args.out_pkl}")


if __name__ == "__main__":
    import argparse
    main()
