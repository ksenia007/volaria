"""
Volaria – Step 5: Combine AM (exonic) & ExpectoSC (non-coding) predictions
==========================================================================
Merges AlphaMissense-derived per-patient gene matrices with ExpectoSC
(non-coding) cell-type matrices and adds summaries. Includes additional 
features not used in the manuscript.

CLI
---
python step5_combine_all_predictions.py \
  --am_pickle /path/to/AM_matrices.pkl \
  --sc_pickle /path/to/expectosc_embeddings.pkl \
  --out_pickle /path/to/combined_mean_counts_weighted.pkl \
  [--drop_ids 001,002] \
  [--drop_ids_file /path/to/drop_ids.txt] \
  [--long_keys podocyte,cd4tcell,cd8tcell,glomerularendothelium,myofibroblast,weighted_AM_mean] \
  [--long_out_csv /path/to/long_matrix.transpose.csv] \
  [--gene_list /path/to/gene_subset.txt]


"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd


def process_predictions_all(loc_am: str,
                            loc_sc: str,
                            drop_IDs=None):
    with open(loc_am, "rb") as f:
        predsAM = pickle.load(f)
    with open(loc_sc, "rb") as f:
        predsSC = pickle.load(f)
    predsALL = predsAM.copy()
    for ct in predsSC.keys():
        predsALL[ct] = np.abs(predsSC[ct])

    if drop_IDs is not None and len(drop_IDs) > 0:
        for ct in predsALL.keys():
            try:
                predsALL[ct] = predsALL[ct].drop(drop_IDs, axis=1, errors="ignore")
            except Exception:
                # only DataFrame-like entries support drop; skip non-frames
                pass

    predsALL["weighted_AM_sum"] = predsALL["AM_scores_sum11"].add(
        0.5 * predsALL["AM_scores_sum11"], fill_value=0
    )
    predsALL["weighted_AM_mean"] = predsALL["AM_scores_mean11"].add(
        0.5 * predsALL["AM_scores_mean10"], fill_value=0
    )

    return predsALL


def one_long_matrix(preds_all: dict,
                    keys_use: list[str],
                    gene_subset: list[str] | None = None) -> pd.DataFrame:
    full_matrix = pd.DataFrame()
    gene_subset = set(gene_subset) if gene_subset else None

    for use_scores in keys_use:
        use_preds = preds_all[use_scores].T
        if gene_subset:
            overlap = list(set(use_preds.columns).intersection(gene_subset))
            use_preds = use_preds[overlap]
        use_preds.columns = [f"{col}_{use_scores}" for col in use_preds.columns]
        full_matrix = pd.concat([full_matrix, use_preds], axis=1)

    return full_matrix.astype(np.float32)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Combine AM and ExpectoSC predictions into a single patient×gene feature dictionary; optionally export a skinny long matrix."
    )
    ap.add_argument("--am_pickle", required=True, help="Path to AM matrices pickle.")
    ap.add_argument("--sc_pickle", required=True, help="Path to ExpectoSC embeddings pickle.")
    ap.add_argument("--out_pickle", required=True, help="Output pickle path for combined dict (no overwrite).")

    ap.add_argument("--drop_ids", default="", help="Comma-separated patient IDs to drop.")
    ap.add_argument("--drop_ids_file", default="", help="Optional file with one patient ID per line to drop.")

    ap.add_argument("--long_keys", default="", help="Comma-separated keys to include in long matrix.")
    ap.add_argument("--long_out_csv", default="", help="Output CSV path for the (TRANSPOSED) long matrix.")
    ap.add_argument("--gene_list", default="", help="Optional file with one gene per line to subset columns.")

    return ap.parse_args()


def main():
    args = parse_args()

    # collect drop IDs 
    drop_ids = []
    if args.drop_ids.strip():
        drop_ids.extend([x.strip() for x in args.drop_ids.split(",") if x.strip()])
    if args.drop_ids_file and os.path.exists(args.drop_ids_file):
        with open(args.drop_ids_file, "r") as fh:
            drop_ids.extend([ln.strip() for ln in fh if ln.strip()])

    # combine
    preds_all = process_predictions_all(
        loc_am=args.am_pickle,
        loc_sc=args.sc_pickle,
        drop_IDs=drop_ids if drop_ids else None,
    )

    # save combined dict 
    if os.path.exists(args.out_pickle):
        print(f"[skip] File exists, not overwriting: {args.out_pickle}")
    else:
        os.makedirs(os.path.dirname(args.out_pickle) or ".", exist_ok=True)
        with open(args.out_pickle, "wb") as f:
            pickle.dump(preds_all, f)
        print(f"[ok] Saved combined dict → {args.out_pickle}")

    # optional skinny matrix
    if args.long_keys and args.long_out_csv:
        keys_use = [k.strip() for k in args.long_keys.split(",") if k.strip()]
        gene_subset = None
        if args.gene_list and os.path.exists(args.gene_list):
            with open(args.gene_list, "r") as fh:
                gene_subset = [ln.strip() for ln in fh if ln.strip()]

        long_df = one_long_matrix(preds_all, keys_use, gene_subset or [])
        out_dir = os.path.dirname(args.long_out_csv) or "."
        os.makedirs(out_dir, exist_ok=True)
        long_df.T.to_csv(args.long_out_csv)
        print(f"[ok] Saved long matrix (transpose) → {args.long_out_csv}")


if __name__ == "__main__":
    main()
