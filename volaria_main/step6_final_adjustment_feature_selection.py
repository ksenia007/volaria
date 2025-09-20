"""
Creating final matrices: regress out relative to baseline GTEx, select top features, drop collinear features
"""

import os
import argparse
import pickle
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

def regress_out_pcs_together(standardized_matrix, n_components, model=None, pca=None):
    """
    Regress out
    """
    if pca is None:
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(standardized_matrix)
    else:
        pcs = pca.transform(standardized_matrix)
    
    if model is None:
        model = LinearRegression() 
    model.fit(pcs, standardized_matrix)
    residuals = standardized_matrix - model.predict(pcs)
    return model, pca, residuals
    
def regress_out_pcs(standardized_matrix, n_components, norm_func='standard', reg_all=None, pca=None):
    """
    Regress out the first N principal components (PCs) from a feature matrix.
    """
    if pca is None:
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(standardized_matrix)
    else:
        pcs = pca.transform(standardized_matrix)

    residuals = pd.DataFrame(index=standardized_matrix.index, columns=standardized_matrix.columns)
    print('Regressing the first', n_components, 'PCs')

    if reg_all is None:
        reg_all = {}
        for feature in standardized_matrix.columns:
            y = standardized_matrix[feature].values.reshape(-1, 1)
            X = pcs
            reg = LinearRegression().fit(X, y)
            reg_all[feature] = reg
            residuals[feature] = (y - reg.predict(X)).flatten()
    else:
        for feature in standardized_matrix.columns:
            y = standardized_matrix[feature].values.reshape(-1, 1)
            X = pcs
            reg = reg_all[feature]
            residuals[feature] = (y - reg.predict(X)).flatten()
    return reg_all, pca, residuals

def drop_collinear_features(X, threshold=0.8):
    """
    Identifies and drops collinear features from X based on correlation threshold.
    """
    print('**&&** Dropping collinear features')
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    drop_map = {}
    visited = set()

    for col in upper.columns:
        if col in visited:
            continue
        high_corr = upper[col][upper[col] > threshold].index.tolist()
        high_corr = [f for f in high_corr if f not in visited]
        if high_corr:
            drop_map[col] = high_corr
            visited.add(col)
            visited.update(high_corr)

    features_to_keep = sorted(X.columns.difference(set(f for group in drop_map.values() for f in group)))
    return features_to_keep, drop_map


def parse_args():
    p = argparse.ArgumentParser(
        description="Volaria Step 6 — Final adjustment & feature selection "
    )
    p.add_argument("--curegn-pkl", required=True,
                   help="Pickle with combined patient-level predictions (dict of matrices) for CureGN.")
    p.add_argument("--gtex-pkl", required=True,
                   help="Pickle with combined patient-level predictions (dict of matrices) for GTEx.")
    p.add_argument("--gtex-labels-csv", required=True,
                   help="GTEx subject phenotype CSV used to exclude renal/kidney conditions.")
    p.add_argument("--train-status-csv", required=True,
                   help="CureGN train status CSV containing WGS_ID column.")
    p.add_argument("--test-status-csv", required=True,
                   help="CureGN test status CSV containing WGS_ID column.")
    p.add_argument("--out-dir", required=True,
                   help="Output directory (e.g., outputs/temp). Files will be saved as regressed_v{version}.XXX.")
    p.add_argument("--version-tag", default="_main",
                   help="Version tag suffix for output files.")
    p.add_argument("--collinear-threshold", type=float, default=0.9,
                   help="Correlation threshold for collinearity drop (default: 0.9).")
    p.add_argument("--n-pcs", type=int, default=10,
                   help="Number of PCs to regress out (default: 10).")
    p.add_argument("--scores", nargs="+",
                   default=['podocyte','glomerularendothelium','myofibroblast','bcell','cd8tcell','cd4tcell','weighted_AM_mean'],
                   help="List of score keys to include.")
    # profile defaults: abs + abs + 99th percentile median + GTEx exclusion handling + collinear drop
    p.add_argument("--no-abs", action="store_true",
                   help="Disable taking absolute values BEFORE residualization (default: abs is ON).")
    p.add_argument("--no-abs-resid", action="store_true",
                   help="Disable taking absolute values AFTER residualization (default: abs is ON).")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load inputs
    with open(args.curegn_pkl, "rb") as f:
        curegnAM = pickle.load(f)
    with open(args.gtex_pkl, "rb") as f:
        gtexAM = pickle.load(f)

    gtex_labels = pd.read_csv(args.gtex_labels_csv, index_col=0)
    exclusion_list_cols = ['DIALYSIS','RENAL_FAILURE','NEPHRITIS','UREMIA','renal','KIDNEY_DISEASE']
    keep_gtex_ids = gtex_labels.index[gtex_labels[exclusion_list_cols].sum(1) == 0]
    kidney_gtex_ids = gtex_labels.index[gtex_labels[exclusion_list_cols].sum(1) != 0]

    train_indices = list(set(pd.read_csv(args.train_status_csv).WGS_ID.values))
    test_indices = list(set(pd.read_csv(args.test_status_csv).WGS_ID.values))
    assert not set(train_indices) & set(test_indices), "Train/Test index overlap detected."

    # Profile defaults
    take_abs = not args.no_abs  
    take_abs_residuals = not args.no_abs_resid 
    n_pcs = args.n_pcs
    regress_feats = True

    regressed_all = {}
    regressed_all_gtex = {}

    for use_scores in args.scores:
        print('*' * 10)
        print('Score', use_scores)

        curegn = curegnAM[use_scores].T.fillna(0)
        curegn = curegn.loc[:, (curegn != 0).any(axis=0)]
        curegn_original_cols = curegn.columns
        curegn.columns = [f'{col}_{use_scores}' for col in curegn.columns]

        curegn_train = curegn.loc[train_indices]
        curegn_test = curegn.loc[test_indices]

        gtex = gtexAM[use_scores].dropna().T.fillna(0)

        # align GTEx to CureGN (keep all CureGN features, pad GTEx with zeros if missing)
        missing = list(set(curegn_original_cols) - set(gtex.columns))
        for m in missing:
            gtex[m] = 0
        gtex = gtex[curegn_original_cols]
        gtex.columns = [f'{col}_{use_scores}' for col in gtex.columns]

        if take_abs:
            print('** TAKING ABS **')
            curegn_train = curegn_train.abs()
            curegn_test = curegn_test.abs()
            gtex  = gtex.abs()

        gtex_base  = gtex.loc[keep_gtex_ids]
        gtex_kidney = gtex.loc[kidney_gtex_ids]

        assert curegn_test.index.isin(test_indices).all() and (~curegn_train.index.isin(test_indices)).all()

        print('CureGN', curegn.shape, 'train', curegn_train.shape, 'test', curegn_test.shape)

        if regress_feats:
            scaler = StandardScaler()
            scaler.fit(gtex_base)
            gtex_norm = scaler.transform(gtex_base)
            model, pca, residuals_train = regress_out_pcs_together(
                pd.DataFrame(gtex_norm, index=gtex_base.index, columns=gtex_base.columns),
                n_components=n_pcs, model=None
            )
            gtex_base = pd.DataFrame(residuals_train, index=gtex_base.index, columns=gtex_base.columns)

        if regress_feats:
            X = scaler.transform(curegn_train)
            X_test = scaler.transform(curegn_test)
            X_gtex_kidney = scaler.transform(gtex_kidney)

            _, _, residuals_test = regress_out_pcs_together(
                pd.DataFrame(X_test, index=curegn_test.index, columns=curegn_test.columns),
                n_components=n_pcs, model=model, pca=pca
            )
            curegn_test = pd.DataFrame(residuals_test, index=curegn_test.index, columns=curegn_test.columns)

            _, _, residuals_train = regress_out_pcs_together(
                pd.DataFrame(X, index=curegn_train.index, columns=curegn_train.columns),
                n_components=n_pcs, model=model, pca=pca
            )
            curegn_train = pd.DataFrame(residuals_train, index=curegn_train.index, columns=curegn_train.columns)

            _, _, residuals_kidney = regress_out_pcs_together(
                pd.DataFrame(X_gtex_kidney, index=gtex_kidney.index, columns=gtex_kidney.columns),
                n_components=n_pcs, model=model, pca=pca
            )
            gtex_kidney = pd.DataFrame(residuals_kidney, index=gtex_kidney.index, columns=gtex_kidney.columns)

        assert curegn_test.index.isin(test_indices).all() and (~curegn_train.index.isin(test_indices)).all()
        assert gtex_kidney.index.intersection(gtex_base.index).empty, "Indexes overlap between gtex_kidney and gtex_base"

        gtex = pd.concat([gtex_base, gtex_kidney], axis=0)

        if take_abs_residuals:
            print('ABS residuals')
            curegn_train = curegn_train.abs()
            curegn_test = curegn_test.abs()
            gtex = gtex.abs()

        regressed_all[use_scores] = {'train': curegn_train, 'test': curegn_test}
        regressed_all_gtex[use_scores] = gtex

    # 99th percentile of |median| per feature on TRAIN
    feature_selection = True
    top_features = {}
    for score in regressed_all.keys():
        regr = regressed_all[score]['train']
        regr_gtex = regressed_all_gtex[score]
        if feature_selection:
            print('SELECTING FEATURES')
            regr_med = regr.median(0)
            regr_med = set(regr_med[(regr_med.abs() > regr_med.abs().quantile(0.99))].index)
        else:
            regr_med = regr.columns
        top_features[score] = list(regr_med)
        print('For score ', score, 'found', len(regr_med), 'top features')

    # Build super matrices
    super_matrix_train = pd.DataFrame()
    super_matrix_test = pd.DataFrame()
    paired_gtex = pd.DataFrame()

    for score in args.scores:
        print('Score', score)
        super_matrix_train = pd.concat([super_matrix_train, regressed_all[score]['train'][top_features[score]]], axis=1)
        super_matrix_test = pd.concat([super_matrix_test, regressed_all[score]['test'][top_features[score]]], axis=1)
        paired_gtex = pd.concat([paired_gtex, regressed_all_gtex[score][top_features[score]]], axis=1)

    # Drop collinear features using TRAIN; keep map
    feats_keep, drop_map = drop_collinear_features(super_matrix_train, threshold=args.collinear_threshold)
    super_matrix_train = super_matrix_train[feats_keep]
    super_matrix_test = super_matrix_test[feats_keep]
    paired_gtex = paired_gtex[feats_keep]

    print('Super matrix train', super_matrix_train.shape,
          'Super matrix test', super_matrix_test.shape,
          'GTEx', paired_gtex.shape)

    # Save
    tag = args.version_tag
    out_train = os.path.join(args.out_dir, f"regressed_v{tag}.TRAIN.csv")
    out_test = os.path.join(args.out_dir, f"regressed_v{tag}.TEST.csv")
    out_gtex = os.path.join(args.out_dir, f"regressed_v{tag}.GTEx.csv")
    out_map = os.path.join(args.out_dir, f"drop_map_v{tag}.pkl")

    super_matrix_train.T.to_csv(out_train)
    print('saved train ->', out_train)
    super_matrix_test.T.to_csv(out_test)
    print('saved test  ->', out_test)
    paired_gtex.T.to_csv(out_gtex)
    print('saved gtex  ->', out_gtex)

    with open(out_map, "wb") as f:
        pickle.dump(drop_map, f)
    print('saved drop map ->', out_map)

if __name__ == "__main__":
    main()
