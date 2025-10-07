"""
Collect ExpectoSC predictions into patient matrices & embeddings
-------
Aggregates ExpectoSC non-coding effect scores into per-patient, per-gene matrices
and derives fixed “embedding” matrices used by Volaria. Supports both GTEx and
CureGN by passing the corresponding input paths via CLI flags.

Inputs (CLI)
------------
--scores_file         CSV with variant-level ExpectoSC scores (index=variant_id).
                     Must include a 'gene_name' column and cell-type columns
                     (e.g., podocyte, cd4tcell, ...)
--pat_dict_file       Pickle of patient→variants dict created in Step 2:
                     {patient_id: {"11":[variant_ids], "10":[variant_ids]}}.
--save_file_full      Output pickle path for the raw patient matrices (no overwrite).
--save_file_embedding Output pickle path for the cell-type embeddings (no overwrite).

"""

import os
import time
import argparse
import pickle
import numpy as np
import pandas as pd
import re

# def _clean_cols(cols):
#     clean = []
#     for c in cols:
#         s = str(c).strip().lower()
#         # drop symbols/spaces 
#         s = re.sub(r"[^0-9a-z]+", "", s)
#         clean.append(s)
#     return clean

# def normalize_key4(s: str) -> str | None:
#     """
#     Accepts variants like '1_54490_G_A_b37', 'chr1_54490_G/A', '1_54490_G_A'
#     Returns '1_54490_G_A' or None if not parseable.
#     """
#     if s is None:
#         return None
#     s = str(s).strip()
#     # unify separators to underscores
#     s = s.replace("/", "_")
#     if s.lower().startswith("chr"):
#         s = s[3:]
#     parts = s.split("_")
#     if len(parts) < 4:
#         return None
#     chrom, pos, ref, alt = parts[0], parts[1], parts[2], parts[3]
#     if not pos.isdigit():
#         return None
#     # standardize bases
#     ref = ref.upper()
#     alt = alt.upper()
#     return f"{chrom}_{pos}_{ref}_{alt}"

def load_inputs(scores_file: str, pat_dict_file: str):
    t0 = time.time()
    with open(pat_dict_file, "rb") as f:
        patients_dict = pickle.load(f)
    scores = pd.read_csv(scores_file, index_col=0)
    print(scores.head())
    set_scores = set(scores.index)
    cell_types = scores.columns[:-1] 
    t1 = time.time()
    print(f"Time taken to load data: {t1 - t0:.2f} seconds")
    print("Number of people:", len(patients_dict))
    print("Size of scores:", scores.shape)
    return patients_dict, scores, set_scores, cell_types

def build_patient_matrices(patients_dict, scores, set_scores, cell_types):
    pat_dict_mean11, pat_dict_count11 = {}, {}
    pat_dict_mean10, pat_dict_count10 = {}, {}
    
    n_pats = len(patients_dict)
    for i, p in enumerate(list(patients_dict.keys())):
        if i % 10 == 0:
            print(f"Processing patient {i}/{n_pats}")

        if "11" in patients_dict[p]:
            variant_list11 = list(set(patients_dict[p]["11"]).intersection(set_scores))
            if variant_list11:
                snp_pat11 = scores.loc[variant_list11]
                pat_dict_mean11[p] = snp_pat11.groupby("gene_name")[cell_types].mean().to_dict()
                pat_dict_count11[p] = snp_pat11["gene_name"].value_counts().to_dict()

        if "10" in patients_dict[p]:
            variant_list10 = list(set(patients_dict[p]["10"]).intersection(set_scores))
            if variant_list10:
                snp_pat10 = scores.loc[variant_list10]
                pat_dict_mean10[p] = snp_pat10.groupby("gene_name")[cell_types].mean().to_dict()
                pat_dict_count10[p] = snp_pat10["gene_name"].value_counts().to_dict()
    mean11  = pd.DataFrame(pat_dict_mean11).fillna(0)
    counts11 = pd.DataFrame(pat_dict_count11).fillna(0)
    mean10  = pd.DataFrame(pat_dict_mean10).fillna(0)
    counts10 = pd.DataFrame(pat_dict_count10).fillna(0)

    matrices = {
        "mean11":  mean11,
        "counts11": counts11,
        "mean10":  mean10,
        "counts10": counts10,
    }
    return matrices

def _lower_index(df):
    # Make index robustly stringy before lowercasing
    if not pd.api.types.is_string_dtype(df.index):
        df.index = df.index.astype(str)
    df.index = df.index.str.lower()
    return df

def build_embeddings(matrices, save_file_embedding):
    # Safely lowercase indices where needed
    for k in ("mean11", "mean10", "count11", "count10"):
        if k in matrices and matrices[k] is not None:
            matrices[k] = _lower_index(matrices[k]) 
    cell_types_panel = [
        "podocyte",
        "bcell",
        "cd8tcell",
        "cd4tcell",
        "glomerularendothelium",
        "myofibroblast",
    ]
    results = {}
    
    print(matrices['mean11'].head())
    for ct in cell_types_panel:
        selected_cell_type_sum11 = matrices["mean11"].loc[ct]
        expanded_mean11 = pd.DataFrame.from_records(
            selected_cell_type_sum11.tolist(),
            index=matrices["mean11"].columns,
        ).transpose()
        expanded_mean11 = expanded_mean11.dropna(how="all").fillna(0)

        selected_cell_type_sum10 = matrices["mean10"].loc[ct]
        expanded_mean10 = pd.DataFrame.from_records(
            selected_cell_type_sum10.tolist(),
            index=matrices["mean10"].columns,
        ).transpose()
        expanded_mean10 = expanded_mean10.dropna(how="all").fillna(0)

        aligned_mean11, aligned_mean10 = expanded_mean11.align(
            expanded_mean10, join="outer", axis=0, fill_value=0
        )
        preds = aligned_mean11 + 0.5 * aligned_mean10
        results[ct] = preds.fillna(0)

    if not os.path.exists(save_file_embedding):
        with open(save_file_embedding, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved embeddings: {save_file_embedding}")
    else:
        print(f"[skip] Exists, not overwriting: {save_file_embedding}")

def main():
    parser = argparse.ArgumentParser(
        description="Collect ExpectoSC predictions into patient matrices and embeddings."
    )
    parser.add_argument("--scores_file", required=True, help="CSV of ExpectoSC scores")
    parser.add_argument("--pat_dict_file", required=True, help="Pickle of patient→variants dict")
    parser.add_argument("--save_file_full", required=True, help="Output pickle path for matrices")
    parser.add_argument("--save_file_embedding", required=True, help="Output pickle path for embeddings")
    args = parser.parse_args()

    print("Scores file:", args.scores_file)
    print("Patient dict file:", args.pat_dict_file)
    print("Save file (matrices):", args.save_file_full)

    start_time = time.time()
    patients_dict, scores, set_scores, cell_types = load_inputs(
        args.scores_file, args.pat_dict_file
    )
    print("Cell types found:", list(cell_types))
    matrices = build_patient_matrices(patients_dict, scores, set_scores, cell_types)

    if not os.path.exists(args.save_file_full):
        with open(args.save_file_full, "wb") as f:
            pickle.dump(matrices, f)
        print(f"Saved matrices: {args.save_file_full}")
    else:
        print(f"[skip] Exists, not overwriting: {args.save_file_full}")

    build_embeddings(matrices, args.save_file_embedding)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()