"""
Volaria - GTEx ROC (Evaluation)
This script is required to reproduce GTEx results and Figure 3F 
Loads prefit models and evaluates on GTEx labels, aggregating ROC across
random seeds. 
"""

from __future__ import annotations

import os
os.environ["SCIPY_ARRAY_API"] = "1"  

import pickle
import warnings
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, balanced_accuracy_score


FIG_DIR = "outputs/figures_verify"
MODELS_PKL = "outputs/temp/models_vF2_2023_seeds.pkl"

VERSION_USE = "_2023_3"
GTEX_X = f"outputs/temp/regressed_v{VERSION_USE}.GTEx.csv"
GTEX_LABELS = (
    "/pph/controlled/dbGaP/GTEx/input/phenotypes/"
    "phs000424.v10.pht002742.v9.p2.c1.GTEx_Subject_Phenotypes.GRU.cleaned.csv"
)
OUTPUT_FNAME = "roc_auc_gtex.svg"

RAND_SEEDS = [13, 37, 42, 73, 132, 57, 101]
TARGET_KEY = "ESRD"
GTEX_POS_COL = "RENALF_DIALYSIS"

COLOR_DICT_PRIMARY = {
    "ESRD": "#034C8C",
    "eGFR40": "#8C1C03",
    "Steroid_resistant": "#F29F05",
}


def _silence_warnings() -> None:
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=DeprecationWarning)

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


def load_models(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)

def load_gtex() -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = pd.read_csv(GTEX_X, index_col=0).T
    labels = pd.read_csv(GTEX_LABELS, index_col=0)
    labels[GTEX_POS_COL] = labels["DIALYSIS"] | labels["RENAL_FAILURE"]
    return X, labels


def evaluate_dataset(model, X: pd.DataFrame, y_true: np.ndarray) -> Dict[str, np.ndarray | float]:
    """Compute ROC, PR, and balanced accuracy for a fitted classifier."""
    y_probs = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc_val = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)
    baseline_pr_auc = float(np.mean(y_true))
    bacc = balanced_accuracy_score(y_true, y_pred)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc_val,
        "pr_auc": pr_auc,
        "bacc": bacc,
        "baseline_pr_auc": baseline_pr_auc,
    }

def evaluate_models(
    models_use: Dict,
    rand_seeds: Iterable[int],
    gtex: pd.DataFrame,
    gtex_labels: pd.DataFrame,
    columns_use: pd.Index,
    gtex_use_col_pos: str,
) -> Tuple[Dict[int, Dict[str, Dict[str, np.ndarray | float]]], Dict[int, float], list]:
    """
    Evaluate across seeds on GTEx.

    Returns:
      results: {seed: {"GTEx": eval_dict}}
      aucs:    {seed: roc_auc_on_GTEx}
      bacc:    [balanced_accuracy_over_seeds]
    """
    results: Dict[int, Dict[str, Dict[str, np.ndarray | float]]] = {}
    aucs: Dict[int, float] = {}
    bacc: list = []

    for rs in rand_seeds:
        model = models_use[rs]["model"]
        eval_gtex = evaluate_dataset(model, gtex[columns_use], gtex_labels[gtex_use_col_pos].values)

        results[rs] = {"GTEx": eval_gtex}
        aucs[rs] = float(eval_gtex["roc_auc"])
        bacc.append(eval_gtex["bacc"])

        print(f"[{rs}] AUC GTEx: {eval_gtex['roc_auc']:.3f}")

    return results, aucs, bacc

def compute_mean_se_roc(
    data_dict: Dict[int, Dict[str, Dict[str, np.ndarray]]],
    n_points: int = 100,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Aggregate ROC across seeds by interpolating TPR at common FPR points,
    returning mean TPR and standard error
    """
    interp_points = np.linspace(0.0, 1.0, n_points)
    tprs = []

    for seed_data in data_dict.values():
        fpr = np.asarray(seed_data["GTEx"]["fpr"])
        tpr = np.asarray(seed_data["GTEx"]["tpr"])
        tprs.append(np.interp(interp_points, fpr, tpr))

    tprs = np.asarray(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_error = tprs.std(axis=0, ddof=0) / np.sqrt(tprs.shape[0])

    return {"GTEx": (interp_points, mean_tpr, std_error)}


def plot_roc_curves_gtex(
    mean_std_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    color: str,
    out_path: str | None = None,
) -> None:
    """Plot mean ROC with SE ribbon; skip saving if file exists."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

    fpr, mean_tpr, se = mean_std_data["GTEx"]
    label = f"GTEx (mean AUC = {np.trapz(mean_tpr, fpr):.2f})"
    ax.plot(fpr, mean_tpr, linestyle="-", color=color, label=label, lw=2)
    ax.fill_between(fpr, mean_tpr - se, mean_tpr + se, alpha=0.2, color=color)

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate", fontsize=15)
    ax.set_ylabel("True Positive Rate", fontsize=15)
    ax.legend()
    ax.tick_params(labelsize=15)
    ax.grid(True)
    sns.despine(ax=ax)
    plt.tight_layout()

    if out_path:
        if not os.path.exists(out_path):  
            plt.savefig(out_path, dpi=300, bbox_inches="tight", format="svg", transparent=True)
        else:
            print(f"[skip] File exists, not overwriting: {out_path}")

    plt.show()


def main() -> None:
    _silence_warnings()
    set_style(bg="white")
    os.makedirs(FIG_DIR, exist_ok=True)

    models = load_models(MODELS_PKL)
    gtex, gtex_labels = load_gtex()

    models_use = models[TARGET_KEY]
    columns_use = models_use["trainX"].columns  

    gtex = gtex.loc[gtex_labels.index]

    results, _, _ = evaluate_models(
        models_use,
        RAND_SEEDS,
        gtex,
        gtex_labels,
        columns_use,
        GTEX_POS_COL,
    )

    mean_std_data = compute_mean_se_roc(results)

    out_path = os.path.join(FIG_DIR, OUTPUT_FNAME)
    plot_roc_curves_gtex(mean_std_data, color=COLOR_DICT_PRIMARY["ESRD"], out_path=out_path)

if __name__ == "__main__":
    main()



