
"""
PRS vs Volaria
Compare PGS vs Volaria (ESRD/eGFR40) ROC and AP gain.
This script is required to reproduce Figure 3 D-E and Supp. Figure 2 in the manuscript.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)

# Paths / Constants
FIG_DIR = "outputs/figures_verify"
MODELS_PKL = "outputs/temp/models_vF2_2023_seeds.pkl"

STATUS_TRAIN = "/data/curegn/curegn_genome/result/ksenia/processed_Ksenia/FINAL/status_train_2023.csv"
STATUS_TEST = "/data/curegn/curegn_genome/result/ksenia/processed_Ksenia/FINAL/status_test_2023.csv"

PGS_STUDIES: List[str] = [
    "PGS000708",
    "PGS004491",
    "PGS004492",
    "PGS004561",
    "PGS004562",
]

TARGET = 'ESRD' #"eGFR40"  
RAND_SEEDS = [13, 37, 42, 73, 132, 57, 101]

COLOR_DICT_PRIMARY = {
    "ESRD": "#034C8C",
    "eGFR40": "#8C1C03",
    "Steroid_resistant": "#F29F05",
}

def set_style(bg: str = "white") -> None:
    """Paper-friendly plotting style (logic preserved)."""
    sns.set(font="serif")
    if bg == "black":
        sns.set(style="ticks", context="paper")
        sns.set_style(
            "darkgrid",
            {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]},
        )
        plt.style.use("dark_background")
        plt.rcParams.update({"grid.linewidth": 0.0, "grid.alpha": 0.5})
    else:
        sns.set(style="ticks", context="paper", font_scale=1.3)
        sns.set_style(
            "whitegrid",
            {
                "font.family": "serif",
                "font.serif": ["Times", "Palatino", "serif"],
                "font.size": 23,
            },
        )
        plt.rcParams.update({"grid.linewidth": 0.2, "grid.alpha": 0.5})
        mpl.rcParams["axes.edgecolor"] = "black"
        mpl.rcParams["axes.labelcolor"] = "black"
        mpl.rcParams["xtick.color"] = "black"
        mpl.rcParams["ytick.color"] = "black"
        mpl.rcParams["text.color"] = "black"

    plt.rcParams["figure.dpi"] = 300


def load_models(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_labels() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_label = pd.read_csv(STATUS_TRAIN, index_col=0)
    test_label_full = pd.read_csv(STATUS_TEST, index_col=0)
    return train_label, test_label_full


def load_prs_scores(studies: Iterable[str]) -> pd.DataFrame:
    scores: pd.DataFrame | None = None
    for i, s in enumerate(studies):
        print(f"Processing {s} ({i+1}/{len(studies)})")
        fp = f"PRS/results/{s}/curegn/score/aggregated_scores.txt.gz"
        tmp = pd.read_csv(fp, sep="\t", compression="gzip")
        tmp["study"] = s
        scores = tmp if scores is None else pd.concat([scores, tmp], axis=0, ignore_index=True)
    return scores


def plot_pr_percent_gain_over_baseline(models: dict, feature: str, rand_seeds: Iterable[int], error: str = "se"):
    aucs = [models[feature][rs]["avg_precision"] for rs in rand_seeds]
    base = [models[feature][rs]["baseline_pr"] for rs in rand_seeds]
    gain = [(a - b) / b * 100 if b > 0 else 0 for a, b in zip(aucs, base)]
    mean_gain = float(np.mean(gain))
    err = float(np.std(gain) / np.sqrt(len(gain)) if error == "se" else np.std(gain))
    return mean_gain, err


def interp_roc(fpr: np.ndarray, tpr: np.ndarray, base_fpr: np.ndarray) -> np.ndarray:
    order = np.argsort(fpr)
    interp = np.interp(base_fpr, fpr[order], tpr[order])
    interp[0], interp[-1] = 0.0, 1.0
    return interp


def get_info_grouped_mean_roc(
    models: dict,
    feature: str,
    rand_seeds: Iterable[int],
    error: str = "se",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    tprs, aucs = [], []
    base_fpr = np.linspace(0, 1, 100)

    for rs in rand_seeds:
        fpr = models[feature][rs]["fpr"]
        tpr = models[feature][rs]["tpr"]
        auc_val = models[feature][rs]["roc_auc"]
        tprs.append(interp_roc(np.asarray(fpr), np.asarray(tpr), base_fpr))
        aucs.append(auc_val)

    tprs = np.asarray(tprs)
    mean_tpr = tprs.mean(axis=0)
    err_tpr = tprs.std(axis=0) / np.sqrt(len(tprs)) if error == "se" else tprs.std(axis=0)

    mean_auc = float(np.mean(aucs))
    err_auc = float(np.std(aucs) / np.sqrt(len(aucs)) if error == "se" else np.std(aucs))
    return base_fpr, mean_tpr, err_tpr, mean_auc, err_auc


def main() -> None:
    set_style(bg="white")
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load PRS scores and labels
    scores = load_prs_scores(PGS_STUDIES)
    train_label, test_label_full = load_labels()

    test_matched = scores.merge(test_label_full, left_on="FID", right_index=True, how="inner")
    train_matched = scores.merge(train_label, left_on="FID", right_index=True, how="inner")

    # Load Volaria 
    models = load_models(MODELS_PKL)

    # PRS evaluation per study 
    results_prs: Dict[str, Dict[str, object]] = {}

    for i, s in enumerate(scores.study.unique()):
        index_use_train = train_matched[train_matched.study == s].index
        index_use_test = test_matched[test_matched.study == s].index

        assert train_matched[train_matched.study == s].CureGNSTudyID.value_counts().max() == 1
        assert test_matched[test_matched.study == s].CureGNSTudyID.value_counts().max() == 1

        print(f"Processing {s} ({i+1}/{len(scores.study.unique())})")

        X_train = train_matched.loc[index_use_train]["SUM"].values.reshape(-1, 1)
        X_test = test_matched.loc[index_use_test]["SUM"].values.reshape(-1, 1)
        Y_train = train_matched.loc[index_use_train][TARGET].values
        Y_test = test_matched.loc[index_use_test][TARGET].values

        model = RandomForestClassifier(n_estimators=100, max_depth=2, criterion="entropy", random_state=42)
        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)
        precision, recall, _ = precision_recall_curve(Y_test, model.predict_proba(X_test)[:, 1])
        avg_precision = average_precision_score(Y_test, model.predict_proba(X_test)[:, 1])

        fpr_test, tpr_test, _ = roc_curve(Y_test, model.predict_proba(X_test)[:, 1])
        accuracy = accuracy_score(Y_test, Y_pred)
        roc_auc = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])
        conf_matrix = confusion_matrix(Y_test, Y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

        base_precision = average_precision_score(Y_test, np.zeros_like(Y_test))
        pr_gain = (avg_precision - base_precision) / base_precision * 100 if base_precision > 0 else 0

        results_prs[s] = {
            "fpr": fpr_test,
            "tpr": tpr_test,
            "roc_auc": roc_auc,
            "avg_precision": avg_precision,
            "baseline_pr": base_precision,
            "pr_gain": pr_gain,
        }

    base_fpr, mean_tpr, err_tpr, mean_auc, err_auc = get_info_grouped_mean_roc(
        models, TARGET, RAND_SEEDS, error="se"
    )

    # Figure: ROC (Volaria vs PRS studies)
    fig1, ax1 = plt.subplots(figsize=(6, 6), dpi=300)

    label = f"Volaria (AUC:{mean_auc:.2f}±{err_auc:.2f})"
    ax1.plot(base_fpr, mean_tpr, color=COLOR_DICT_PRIMARY[TARGET], lw=2, label=label)
    ax1.fill_between(base_fpr, mean_tpr - err_tpr, mean_tpr + err_tpr, color=COLOR_DICT_PRIMARY[TARGET], alpha=0.2)

    for s in scores.study.unique():
        ax1.plot(
            results_prs[s]["fpr"],
            results_prs[s]["tpr"],
            label=f'{s} (area = {results_prs[s]["roc_auc"]:.2f})',
            color="grey",
            ls="--",
            lw=2,
        )

    ax1.plot([0, 1], [0, 1], ls="--", lw=1, color="black")
    ax1.set_xlabel("False Positive Rate", fontsize=18)
    ax1.set_ylabel("True Positive Rate", fontsize=18)
    ax1.tick_params(labelsize=17)
    ax1.legend(loc="lower right", fontsize=15)
    sns.despine(ax=ax1)
    plt.tight_layout()

    out_roc = os.path.join(FIG_DIR, "roc_volaria_vs_prs.svg")
    if not os.path.exists(out_roc):
        plt.savefig(out_roc, format="svg", transparent=True, bbox_inches="tight", dpi=300)
    else:
        print(f"[skip] File exists, not overwriting: {out_roc}")
    plt.show()

    # Figure: AP Gain over baseline
    mean_gain, err = plot_pr_percent_gain_over_baseline(models, TARGET, rand_seeds=RAND_SEEDS, error="se")

    # Build the PRS gains dataframe 
    pr_gains: List[float] = []
    pr_labels: List[str] = []
    colors: List[str] = []

    for s in scores.study.unique():
        pr_gains.append(results_prs[s]["pr_gain"])
        pr_labels.append(s)
        colors.append("grey")

    pr_gains.append(mean_gain)
    pr_labels.append("Volaria")
    colors.append(COLOR_DICT_PRIMARY[TARGET])

    pr_gain_df = pd.DataFrame({"study": pr_labels, "gain": pr_gains}).sort_values(by="gain", ascending=True)

    # Align colors so 'Volaria' uses its color & others grey
    color_map = {name: ("grey" if name != "Volaria" else COLOR_DICT_PRIMARY[TARGET]) for name in pr_gain_df["study"]}

    width = 0.7
    fig2, ax2 = plt.subplots(figsize=(3.5, 5.5), dpi=300)
    ax2.bar(
        pr_gain_df["study"],
        pr_gain_df["gain"],
        width,
        yerr=[0] * len(pr_gain_df),
        error_kw={"ecolor": "black", "elinewidth": 1.5},
        edgecolor="white",
        color=[color_map[s] for s in pr_gain_df["study"]],
        capsize=4,
        alpha=0.9,
        label="PRS",
    )
    ax2.set_ylabel("AP Gain over Baseline (%)", fontsize=16)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.tick_params(labelsize=17)
    plt.xticks(rotation=90, ha="center")
    sns.despine(ax=ax2)
    plt.tight_layout()

    out_gain = os.path.join(FIG_DIR, "ap_gain_prs_vs_volaria.svg")
    if not os.path.exists(out_gain):
        plt.savefig(out_gain, format="svg", transparent=True, bbox_inches="tight", dpi=300)
    else:
        print(f"[skip] File exists, not overwriting: {out_gain}")
    plt.show()


if __name__ == "__main__":
    main()

