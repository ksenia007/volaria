"""
This script is required to replicated structure contribution analysis, Figure 3 A-C, and Supp. Figure 1
"""

import os
os.environ["SCIPY_ARRAY_API"] = "1"

import pickle
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    auc,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from get_data_utils import get_subset


FIG_DIR = "outputs/figures_verify"
STATUS_TRAIN = "/data/curegn/curegn_genome/result/ksenia/processed_Ksenia/FINAL/status_train_2023.csv"
STATUS_TEST = "/data/curegn/curegn_genome/result/ksenia/processed_Ksenia/FINAL/status_test_2023.csv"
VERSION_USE = "_2023_3"
TRAIN_X = f"outputs/temp/regressed_v{VERSION_USE}.TRAIN.csv"
TEST_X = f"outputs/temp/regressed_v{VERSION_USE}.TEST.csv"
MODELS_PKL = "outputs/temp/models_vF2_2023_seeds.pkl"

RAND_SEEDS = [13, 37, 42, 73, 132, 57, 101]

COLOR_DICT_PRIMARY = {
    "ESRD": "#034C8C",
    "eGFR40": "#8C1C03",
    "Steroid_resistant": "#F29F05",
}

COLOR_DICT_PRIMARY_OFFLIGHT = {
    "ESRD": "#A6C3DE",
    "eGFR40": "#D99C91",
    "Steroid_resistant": "#F9DC9C",
}

LABEL_DICT = {
    "ESRD": "Volaria, main",
    "ESRD_base_SEX": "Sex only",
    "ESRD_base_ge_max": "Flat regulatory",
    "ESRD_base_AM_max": "Flat exonic",
    "ESRD_base_ALL_max": "Flat regulatory \n& exonic",
    "ESRD_base_diagnosis": "Diagnosis only",
    "eGFR40": "Volaria, main",
    "eGFR40_base_SEX": "Sex only",
    "eGFR40_base_ge_max": "Flat regulatory",
    "eGFR40_base_AM_max": "Flat exonic",
    "eGFR40_base_ALL_max": "Flat regulatory \n& exonic",
    "eGFR40_base_diagnosis": "Diagnosis only",
    "Steroid_resistant": "Volaria, main",
    "Steroid_resistant_base_SEX": "Sex only",
    "Steroid_resistant_base_ge_max": "Flat regulatory",
    "Steroid_resistant_base_AM_max": "Flat exonic",
    "Steroid_resistant_base_ALL_max": "Flat regulatory \n& exonic",
    "Steroid_resistant_base_diagnosis": "Diagnosis only",
}

PREDICT_FEATURES = [
    ("ESRD", True),
    ("eGFR40", True),
    ("Steroid_resistant", True),
]

PROVIDE_PARAMS = {
    "ESRD": {"n_estimators": 100, "max_depth": 2},
    "eGFR40": {"n_estimators": 1000, "max_depth": 2},
    "Steroid_resistant": {"n_estimators": 1000, "max_depth": 2},
}


def _silence_warnings() -> None:
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=DeprecationWarning)


def set_style(bg: str = "white") -> None:
    sns.set(font="serif")
    if bg == "black":
        sns.set(style="ticks", context="paper")
        sns.set_style("darkgrid", {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]})
        plt.style.use("dark_background")
        plt.rcParams.update({"grid.linewidth": 0.0, "grid.alpha": 0.5})
    else:
        sns.set(style="ticks", context="paper", font_scale=1.3)
        sns.set_style(
            "whitegrid",
            {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"], "font.size": 23},
        )
        plt.rcParams.update({"grid.linewidth": 0.2, "grid.alpha": 0.5})
        mpl.rcParams["axes.edgecolor"] = "black"
        mpl.rcParams["axes.labelcolor"] = "black"
        mpl.rcParams["xtick.color"] = "black"
        mpl.rcParams["ytick.color"] = "black"
        mpl.rcParams["text.color"] = "black"
    plt.rcParams["figure.dpi"] = 300


def load_labels_and_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_label = pd.read_csv(STATUS_TRAIN, index_col=0)
    test_label_full = pd.read_csv(STATUS_TEST, index_col=0)
    trainX = pd.read_csv(TRAIN_X, index_col=0).T
    testX = pd.read_csv(TEST_X, index_col=0).T
    return train_label, test_label_full, trainX, testX


def load_models(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# Baseline feature matrices
def build_baseline_matrices(trainX: pd.DataFrame, testX: pd.DataFrame, train_label: pd.DataFrame, test_label: pd.DataFrame):
    ge_cols = [c for c in trainX.columns if "_AM_mean" not in c]
    am_cols = [c for c in trainX.columns if "_AM_mean" in c]

    trainX_base = pd.DataFrame(np.abs(trainX[am_cols]).max(1), columns=["AM_max"])
    testX_base = pd.DataFrame(np.abs(testX[am_cols]).max(1), columns=["AM_max"])
    trainX_base["ge_max"] = np.abs(trainX[ge_cols]).max(1)
    testX_base["ge_max"] = np.abs(testX[ge_cols]).max(1)

    cols_use = ["SEX"]
    temp = train_label.loc[trainX.index][cols_use].fillna(0).copy()
    temp_test = test_label.loc[testX.index][cols_use].fillna(0).copy()
    temp["SEX"] = temp["SEX"] == "F"
    temp_test["SEX"] = temp_test["SEX"] == "F"

    trainX_base = pd.concat([trainX_base, temp], axis=1)
    testX_base = pd.concat([testX_base, temp_test], axis=1)

    return trainX_base, testX_base


def fit_eval_baselines(
    trainX_base: pd.DataFrame,
    testX_base: pd.DataFrame,
    train_label: pd.DataFrame,
    test_label: pd.DataFrame,
) -> dict:
    models_base: dict[str, dict[str, list[float]]] = {}
    disease_subset: list[str] = []
    ancestry_subset: list[str] = []

    for predict_feature, use_time_cutoff in PREDICT_FEATURES:
        for use_input in ["SEX", "ge_max", "AM_max", "ALL_max"]:
            key = f"{predict_feature}_{use_input}"
            models_base[key] = {"auc": [], "pr_auc": []}

            if predict_feature in PROVIDE_PARAMS:
                cfg = PROVIDE_PARAMS[predict_feature]
                max_depth = cfg["max_depth"]
                n_estimators = cfg["n_estimators"]
            else:
                max_depth = 3
                n_estimators = 10

            X, y, X_test, y_test, _ = get_subset(
                trainX_base,
                testX_base,
                predict_feature,
                disease_subset=disease_subset,
                ancestry_subset=ancestry_subset,
                use_time_cutoff=use_time_cutoff,
            )

            if use_input not in ["ALL", "ALL_max", "ALL_counts", "diagnosis", "diagnosis_MCD_FSGS"]:
                X = X[[use_input]]
                X_test = X_test[[use_input]]
            elif use_input == "ALL_max":
                X = X[["ge_max", "AM_max"]]
                X_test = X_test[["ge_max", "AM_max"]]
            elif use_input in ["diagnosis", "diagnosis_MCD_FSGS"]:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                if use_input == "diagnosis":
                    X_diag = encoder.fit_transform(train_label.loc[X.index][["DIAGNOSIS"]])
                    X_diag_test = encoder.transform(test_label.loc[X_test.index][["DIAGNOSIS"]])
                else:
                    mcd_fsgs = ["MCD", "FSGS"]
                    train_label_filtered = train_label[train_label["DIAGNOSIS"].isin(mcd_fsgs)]
                    test_label_filtered = test_label[test_label["DIAGNOSIS"].isin(mcd_fsgs)]
                    X = X.loc[train_label_filtered.index]
                    X_test = X_test.loc[test_label_filtered.index]
                    X_diag = encoder.fit_transform(train_label_filtered.loc[X.index][["DIAGNOSIS"]])
                    X_diag_test = encoder.transform(test_label_filtered.loc[X_test.index][["DIAGNOSIS"]])

                X = pd.DataFrame(X_diag, index=X.index, columns=encoder.get_feature_names_out(["DIAGNOSIS"]))
                X_test = pd.DataFrame(X_diag_test, index=X_test.index, columns=encoder.get_feature_names_out(["DIAGNOSIS"]))

            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X_test = scaler.transform(X_test)

            for seed in RAND_SEEDS:
                clf = RandomForestClassifier(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    random_state=seed,
                    class_weight="balanced",
                    criterion="entropy",
                ).fit(X, y)

                fpr_test, tpr_test, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
                roc_auc_test = auc(fpr_test, tpr_test)
                models_base[key]["auc"].append(roc_auc_test)

                avg_precision = average_precision_score(y_test, clf.predict_proba(X_test)[:, 1])
                baseline = float(np.mean(y_test))
                pr_gain = (avg_precision - baseline) / baseline if baseline > 0 else 0.0
                models_base[key]["pr_auc"].append(pr_gain)

    return models_base


def build_plot_table(models: dict, models_base: dict) -> pd.DataFrame:
    rand_seeds = RAND_SEEDS
    dict_plot: dict[str, dict[str, float]] = {}

    for key in models.keys():
        vals_auc = [models[key][rs]["roc_auc"] for rs in rand_seeds]
        vals_pr = [
            (models[key][rs]["avg_precision"] - models[key][rs]["baseline_pr"]) / models[key][rs]["baseline_pr"]
            for rs in rand_seeds
        ]
        dict_plot[key] = {
            "auc_mean": float(np.mean(vals_auc)),
            "auc_std": float(np.std(vals_auc)),
            "pr_mean": float(np.mean(vals_pr) * 100.0),
            "pr_std": float(np.std(vals_pr) * 100.0),
        }

        for feats in ["SEX", "ge_max", "AM_max", "ALL_max"]:
            k2 = f"{key}_base_{feats}"
            dict_plot[k2] = {
                "auc_mean": float(np.mean(models_base[f"{key}_{feats}"]["auc"])),
                "auc_std": float(np.std(models_base[f"{key}_{feats}"]["auc"])),
                "pr_mean": float(np.mean(models_base[f"{key}_{feats}"]["pr_auc"]) * 100.0),
                "pr_std": float(np.std(models_base[f"{key}_{feats}"]["pr_auc"]) * 100.0),
            }

    return pd.DataFrame.from_dict(dict_plot, orient="index")


def plot_pr_gain(plot_df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 6), dpi=300)
    for i, outcome in enumerate(["ESRD", "eGFR40", "Steroid_resistant"]):
        rows_keep = [ix for ix in plot_df.index if ix.startswith(outcome)]
        df = plot_df.loc[rows_keep].copy()
        df["Model"] = df.index
        df["is_base"] = df["Model"].str.contains("base")
        df = df.sort_values(by=["is_base", "pr_mean"], ascending=[True, False])

        if outcome == "Steroid_resistant":
            df = df[~df["Model"].str.contains("diagnosis")]

        x_labels = [LABEL_DICT.get(m, m) for m in df["Model"]]
        bar_colors = []
        for m in df["Model"]:
            if "base_SEX" in m:
                bar_colors.append("lightgrey")
            elif "base" in m:
                bar_colors.append(COLOR_DICT_PRIMARY_OFFLIGHT[outcome])
            else:
                bar_colors.append(COLOR_DICT_PRIMARY[outcome])

        axes[i].bar(x_labels, df["pr_mean"], yerr=df["pr_std"], capsize=5, color=bar_colors)
        axes[i].set_ylabel("AP (% gain over baseline)")
        axes[i].tick_params(axis="x", rotation=90)
        sns.despine(ax=axes[i])

    plt.tight_layout()
    if not os.path.exists(out_path):
        plt.savefig(out_path, format="svg", transparent=True, bbox_inches="tight", dpi=300)
    else:
        print(f"[skip] File exists, not overwriting: {out_path}")
    plt.show()


def plot_roc_auc(plot_df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 6), dpi=300)
    for i, outcome in enumerate(["ESRD", "eGFR40", "Steroid_resistant"]):
        rows_keep = [ix for ix in plot_df.index if ix.startswith(outcome)]
        df = plot_df.loc[rows_keep].copy()
        df["Model"] = df.index
        df["is_base"] = df["Model"].str.contains("base")
        df["is_base_sex"] = df["Model"].str.contains("_base_SEX")
        df = df.sort_values(by=["is_base_sex", "is_base", "auc_mean"], ascending=[False, False, True])

        if outcome == "Steroid_resistant":
            df = df[~df["Model"].str.contains("diagnosis")]

        x_labels = [LABEL_DICT.get(m, m) for m in df["Model"]]
        bar_colors = []
        for m in df["Model"]:
            if "base_SEX" in m:
                bar_colors.append("lightgrey")
            elif "base" in m:
                bar_colors.append(COLOR_DICT_PRIMARY_OFFLIGHT[outcome])
            else:
                bar_colors.append(COLOR_DICT_PRIMARY[outcome])

        axes[i].bar(x_labels, df["auc_mean"], yerr=df["auc_std"], capsize=5, color=bar_colors)
        axes[i].set_ylabel("ROC AUC")
        axes[i].tick_params(axis="x", rotation=90, labelsize=14)
        axes[i].tick_params(axis="y", labelsize=14)
        axes[i].set_ylim(0.4, df["auc_mean"].max() + df["auc_std"].max() * 1.2)
        sns.despine(ax=axes[i])

    plt.tight_layout()
    if not os.path.exists(out_path):
        plt.savefig(out_path, format="svg", transparent=True, bbox_inches="tight", dpi=300)
    else:
        print(f"[skip] File exists, not overwriting: {out_path}")
    plt.show()


def main() -> None:
    _silence_warnings()
    set_style(bg="white")
    os.makedirs(FIG_DIR, exist_ok=True)

    train_label, test_label_full, trainX, testX = load_labels_and_features()

    print(trainX.shape, testX.shape)
    print(
        "Before filtering, N pos and N neg",
        int(train_label.ESRD.sum()),
        int((train_label.ESRD == 0).sum()),
    )
    print(
        "Before filtering, N pos and N neg TEST",
        int(test_label_full.ESRD.sum()),
        int((test_label_full.ESRD == 0).sum()),
    )
    print("Total patients in both sets", int(train_label.shape[0] + test_label_full.shape[0]))

    trainX_base, testX_base = build_baseline_matrices(trainX, testX, train_label, test_label_full)

    models_base = fit_eval_baselines(trainX_base, testX_base, train_label, test_label_full)

    with open(MODELS_PKL, "rb") as f:
        models = pickle.load(f)

    plot_df = build_plot_table(models, models_base)

    pr_path = os.path.join(FIG_DIR, "pr_gain_volaria_vs_baselines.svg")
    plot_pr_gain(plot_df, pr_path)

    roc_path = os.path.join(FIG_DIR, "roc_auc_volaria_vs_baselines.svg")
    plot_roc_auc(plot_df, roc_path)


if __name__ == "__main__":
    main()

