
# # we pass in scores as is, and then take ABS of the residuals
# # use from step6_final_adjustment_feature_selection import regress_out_pcs_together, 
# # def regress_out_pcs_together(standardized_matrix, n_components, model=None, pca=None):
# # models regress are as "'podocyte': {'scaler': StandardScaler(),
# #   'model': LinearRegression(),
# #   'pca': PCA(n_components=10)},
# #  'glomerularendothelium': {'scaler': StandardScaler(),
# #   'model': LinearRegression(),
# #   'pca': PCA(n_components=10)}," ... 
# SCALE_LOC = '/Users/sokolova/Documents/research/volaria/resources/outcome_models/models_regress.pkl'

# # then using these predictions, we pass through the models
# # inside MODELS_BASE we have models like ESRD.pkl, etc
# MODELS_BASE = "/Users/sokolova/Documents/research/volaria/resources/outcome_models/outcomes/"

#!/usr/bin/env python3
"""
Get predictions from pre-trained models
"""
#!/usr/bin/env python3
from __future__ import annotations
import os, glob, argparse, pickle, json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from step6_final_adjustment_feature_selection import regress_out_pcs_together

# ---------- IO ----------
def read_long_table(path: str) -> pd.DataFrame:
    sep = ","
    df = pd.read_csv(path, sep=sep, index_col=0)
    samples = df.columns.tolist()
    return df, samples

def load_regress_pack(path: str) -> Dict[str, dict]:
    with open(path, "rb") as f:
        pack = pickle.load(f)
    if not isinstance(pack, dict):
        raise SystemExit("[ERR] regressors-pkl must be dict of ct -> {'scaler','pca','model'}")
    print(f"[info] regressors loaded: {len(pack)} CTs")
    return pack

def build_ct_vector(df: pd.DataFrame, ct_list: List[str]) -> np.ndarray:
    # aggregate values per feature_name; only CTs in pack are considered
    g = df.groupby("feature_name")["value"].sum()
    x = np.array([float(g.get(ct, 0.0)) for ct in ct_list], dtype=np.float64)  # shape (n_ct,)
    return x

def standardize_with_pack(x_ct: np.ndarray, ct_list: List[str], pack: Dict[str, dict]) -> np.ndarray:
    # apply per-CT scaler (fitted) to the single value; leave as-is if no scaler
    z = np.zeros_like(x_ct)
    for j, ct in enumerate(ct_list):
        sc = pack.get(ct, {}).get("scaler", None)
        v = x_ct[j]
        if sc is not None and hasattr(sc, "transform"):
            # use the scaler safely on 2D input
            z[j] = float(sc.transform([[v]])[0, 0])
        else:
            z[j] = v
    return z

def residual_abs_per_ct(z_ct: np.ndarray, ct_list: List[str], pack: Dict[str, dict]) -> Dict[str, float]:
    """
    For each CT:
      residual = regress_out_pcs_together(standardized_matrix, n_components, model=ct.model, pca=ct.pca)
      ABS(residual[ct])
    We pass the WHOLE standardized row (1 x n_ct) so PCA/model see the joint CT vector,
    then read back the residual for the target CT.
    """
    Z = z_ct.reshape(1, -1)  # shape (1, n_ct)
    print(Z)
    res_map: Dict[str, float] = {}
    for j, ct in enumerate(ct_list):
        cfg = pack[ct]
        pca = cfg.get("pca", None)
        mdl = cfg.get("model", None)
        n_comp = getattr(pca, "n_components", 0) if pca is not None else 0
        resid = regress_out_pcs_together(Z, n_comp, model=mdl, pca=pca)
        arr = np.asarray(resid).reshape(Z.shape) if not isinstance(resid, np.ndarray) else resid
        val = float(arr[0, j]) if arr.size == Z.size else float(np.ravel(arr)[0])
        res_map[ct] = abs(val)   
    return res_map

# ---------- Models ----------
def load_model_obj(pkl: str):
    with open(pkl, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "model" in obj and "features" in obj:
        return obj["model"], list(map(str, obj["features"]))
    raise SystemExit(f"[ERR] unsupported model format: {pkl}")

def align_X(inputs: pd.DataFrame, features_needed: List[str]) -> np.ndarray:
    return inputs[features_needed]

def predict_all(models_dir: str, inputs: pd.DataFrame, outdir: str) -> pd.DataFrame:
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame()
    for pkl in sorted(glob.glob(os.path.join(models_dir, "*.pkl"))):
        model, feats = load_model_obj(pkl)
        X = align_X(inputs, feats)     
        base = os.path.splitext(os.path.basename(pkl))[0]
        print(f"[info] {base}: need={len(feats)}; X.shape={X.shape}")
        
        proba = model.predict_proba(X)
        temp = pd.DataFrame({'ID': inputs.index, 'probability': proba[:, 1]})
        temp['model'] = base
        df = pd.concat([df, temp], axis=0, ignore_index=True)
    df = df.dropna(subset=['ID']).reset_index(drop=True)
    csv_path = os.path.join(outdir, "predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"[ok] saved {csv_path} (rows={len(df)})")
    return df

def main():
    ap = argparse.ArgumentParser(description="Residualize CTs (ABS) and predict with pretrained outcome models (no training).")
    ap.add_argument("--features", required=True, help="Step5 long table (combined_features_long.tsv).")
    ap.add_argument("--regressors-pkl", required=True, help="models_regress.pkl (dict: ct -> scaler/pca/model).")
    ap.add_argument("--models-dir", required=True, help="Directory with outcome model .pkl files.")
    ap.add_argument("--outdir", required=True, help="Where to write predictions.")
    args = ap.parse_args()

    long_df, samples = read_long_table(args.features)
    print(f"[info] long table: {long_df.shape}, samples: {len(samples)}")
    cts = long_df.index.str.split('_').str[1].unique().tolist()
    print('AVAILABLE ', cts)

    # 2) load regressors + pick CTs
    pack = load_regress_pack(args.regressors_pkl)
    ct_list = sorted(pack.keys())

    # 3) convert long into more common shape: 
    df = long_df.T
    res_map = pd.DataFrame()
    for ct in ct_list:
        print('[info] Processing:', ct)
        col_subset = [c for c in df.columns if ct in c]
        subset_vals = df[col_subset].copy()   # <- copy to avoid SettingWithCopy
        cfg = pack[ct]
        pca = cfg.get("pca", None)
        mdl = cfg.get("model", None)
        features = cfg.get("features", None)
        # limit to current ct
        features = [f for f in features if ct in f] if features is not None else None

        # Use 'features' from cfg; if missing in subset_vals, fill with 0 and keep order
        if features is not None:
            # add any missing cols with 0.0 and orders as in 'features'
            subset_vals = subset_vals.reindex(columns=features, fill_value=0.0)
            
        n_comp = 10
        _, _, resid = regress_out_pcs_together(subset_vals, n_comp, model=mdl, pca=pca)
        print(resid.max().max())
        res_map = pd.concat([res_map, pd.DataFrame(np.abs(resid))], axis=1)

    print(f"[info] residual ABS features: {len(res_map)}")

    # predict using ONLY features each model needs
    os.makedirs(args.outdir, exist_ok=True)
    predict_all(args.models_dir, res_map, args.outdir)

if __name__ == "__main__":
    main()
