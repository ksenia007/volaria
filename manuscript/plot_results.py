"""
This script is required to evaluate model results and generate Figure 1 and Supp. Table 2
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

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

FIG_DIR = "outputs/figures_verify"
MODELS_PKL = "outputs/models/models_outcomes.pkl" 
STATUS_TRAIN = "/data/curegn/curegn_genome/result/ksenia/processed_Ksenia/FINAL/status_train_2023.csv"
STATUS_TEST = "/data/curegn/curegn_genome/result/ksenia/processed_Ksenia/FINAL/status_test_2023.csv"
VERSION_USE = "_2023_3"
TRAIN_X = f"outputs/temp/regressed_v{VERSION_USE}.TRAIN.csv"
TEST_X  = f"outputs/temp/regressed_v{VERSION_USE}.TEST.csv"

COLOR_PRIMARY = {
    "ESRD": "#034C8C",
    "eGFR40": "#8C1C03",
    "Steroid_resistant": "#F29F05",
}
RENAME = {
    "eGFR40": "eGFR decline",
    "ESRD": "Kidney failure",
    "Steroid_resistant": "Steroid resistant",
}
RENAME_BOX = {
    "eGFR40": "eGFR\ndecline",
    "ESRD": "Kidney\nfailure",
    "Steroid_resistant": "Steroid\nresistant",
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
        sns.set_style("whitegrid", {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"], "font.size": 23})
        plt.rcParams.update({"grid.linewidth": 0.2, "grid.alpha": 0.5})
        mpl.rcParams["axes.edgecolor"] = "black"
        mpl.rcParams["axes.labelcolor"] = "black"
        mpl.rcParams["xtick.color"] = "black"
        mpl.rcParams["ytick.color"] = "black"
        mpl.rcParams["text.color"] = "black"
    plt.rcParams["figure.dpi"] = 300

def interp_roc(fpr: np.ndarray, tpr: np.ndarray, base_fpr: np.ndarray) -> np.ndarray:
    order = np.argsort(fpr)
    out = np.interp(base_fpr, fpr[order], tpr[order])
    out[0], out[-1] = 0.0, 1.0
    return out

def plot_grouped_mean_roc(
    models: dict,
    predict_features: list[tuple[str, bool]],
    rand_seeds: list[int],
    error: str = "se",
    line_styles: list = ("-", "--", ":", "-.", "-"),
    filename: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    base_fpr = np.linspace(0, 1, 100)

    for i, feat in enumerate(predict_features):
        feature, _ = feat
        tprs, aucs = [], []
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

        label = f"{RENAME[feature]} (AUC:{mean_auc:.2f}±{err_auc:.2f})"
        ax.plot(base_fpr, mean_tpr, color=COLOR_PRIMARY[feature], lw=2, label=label, ls=line_styles[i % len(line_styles)])
        ax.fill_between(base_fpr, mean_tpr - err_tpr, mean_tpr + err_tpr, color=COLOR_PRIMARY[feature], alpha=0.2)

    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="black")
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(loc="lower right", fontsize=13.5)
    sns.despine(ax=ax)
    plt.tight_layout()
    if filename:
        if not os.path.exists(filename):
            plt.savefig(filename, dpi=300, bbox_inches="tight", format="svg", transparent=True)
        else:
            print(f"[skip] File exists, not overwriting: {filename}")
    plt.show()

def plot_pr_percent_gain_over_baseline_box(
    models: dict,
    predict_features: list[tuple[str, bool]],
    rand_seeds: list[int],
    filename: str | None = None,
) -> None:
    records, used_features = [], []
    for feat in predict_features:
        f, _ = feat
        used_features.append(f)
        aucs = [models[f][rs]["avg_precision"] for rs in rand_seeds]
        bases = [models[f][rs]["baseline_pr"] for rs in rand_seeds]
        gains = [(a - b) / b * 100 if b > 0 else 0 for a, b in zip(aucs, bases)]
        records.extend([{"feature": f, "gain": g} for g in gains])

    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=500)
    palette = [COLOR_PRIMARY[f] for f in used_features]
    sns.boxplot(data=df, x="feature", y="gain", palette=palette, width=0.6, linewidth=1.3, fliersize=3, ax=ax)
    ax.set_xticklabels([RENAME_BOX.get(f, f) for f in used_features], fontsize=14)
    ax.set_ylabel("AP Gain over Baseline (%)", fontsize=16)
    ax.set_xlabel("", fontsize=16)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.yticks(fontsize=14)
    sns.despine(ax=ax)
    plt.tight_layout()
    if filename:
        if not os.path.exists(filename):
            plt.savefig(filename, format="svg", transparent=True, bbox_inches="tight", dpi=300)
        else:
            print(f"[skip] File exists, not overwriting: {filename}")
    plt.show()

def groupwise_metrics_table(
    models: dict,
    feature: str,
    seeds: list[int],
    test_label: pd.DataFrame,
    train_label: pd.DataFrame,
    group_by: str,
    min_events: int = 1,
) -> pd.DataFrame:
    X_test = models[feature]["testX"]
    y_test = models[feature]["testY"]
    y_train = models[feature]["trainY"]

    rows = []
    groups_check = list(test_label[group_by].dropna().unique())
    groups_check.append("ALL")

    for v in groups_check:
        if v != "ALL":
            idx = test_label.index[test_label[group_by] == v].intersection(X_test.index)
            idx_train = train_label.index[train_label[group_by] == v].intersection(y_train.index)
        else:
            idx = X_test.index
            idx_train = y_train.index

        n_total = len(idx)
        n_events = int(y_test.loc[idx].sum())
        prevalence = n_events / n_total if n_total else 0

        n_total_train = len(idx_train)
        n_events_train = int(y_train.loc[idx_train].sum())
        prevalence_train = n_events_train / n_total_train if n_total_train else 0

        if n_events < min_events:
            continue

        roc_aucs, pr_aucs, bal_accs, gain_pr = [], [], [], []
        for seed in seeds:
            try:
                clf = models[feature][seed]["model"]
                probs = clf.predict_proba(X_test.loc[idx])[:, 1]
            except Exception:
                continue

            y_true = y_test.loc[idx]
            roc_aucs.append(roc_auc_score(y_true, probs))
            pr_aucs.append(average_precision_score(y_true, probs))
            bal_accs.append(balanced_accuracy_score(y_true, (probs >= 0.5).astype(int)))
            gain_pr.append((average_precision_score(y_true, probs) - prevalence) * 100 / prevalence if prevalence > 0 else 0)

        if not roc_aucs:
            continue

        rows.append(
            {
                "group_by": group_by,
                "value": v,
                "n_total": n_total,
                "n_events": n_events,
                "event_rate%": round(100 * prevalence, 1),
                "ROC_AUC": f"{np.mean(roc_aucs):.2f} ± {np.std(roc_aucs):.2f}",
                "PR_AUC": f"{np.mean(pr_aucs):.2f} ± {np.std(pr_aucs):.2f}",
                "Balanced_acc": f"{np.mean(bal_accs):.2f} ± {np.std(bal_accs):.2f}",
                "n_total_train": n_total_train,
                "n_events_train": n_events_train,
                "event_rate_train%": round(100 * prevalence_train, 1),
                "gain_pr%": f"{np.mean(gain_pr):.1f} ± {np.std(gain_pr):.1f}",
            }
        )

    df = pd.DataFrame(rows).reset_index(drop=True)
    df = df.sort_values(by=["n_events"], ascending=[False])
    df.rename(
        columns={
            "n_total": "Total samples in test",
            "n_events": "Total events in test",
            "n_total_train": "Total samples in train",
            "n_events_train": "Total events in train",
            "event_rate%": "Event rate in test (%)",
            "ROC_AUC": "ROC AUC",
            "PR_AUC": "PR AUC",
            "Balanced_acc": "Balanced accuracy",
            "event_rate_train%": "Event rate in train (%)",
            "gain_pr%": "Gain in PR AUC (%)",
            "value": "Group",
        },
        inplace=True,
    )
    return df

def main() -> None:
    _silence_warnings()
    set_style(bg="white")
    os.makedirs(FIG_DIR, exist_ok=True)

    with open(MODELS_PKL, "rb") as f:
        models = pickle.load(f)

    train_label = pd.read_csv(STATUS_TRAIN, index_col=0)
    test_label_full = pd.read_csv(STATUS_TEST, index_col=0)
    trainX = pd.read_csv(TRAIN_X, index_col=0).T
    testX = pd.read_csv(TEST_X, index_col=0).T
    print(trainX.shape, testX.shape)

    predict_features = [("Steroid_resistant", True), ("eGFR40", True), ("ESRD", True)]
    rand_seeds = [13, 37, 42, 73, 132, 57, 101]

    box_path = os.path.join(FIG_DIR, "boxplot_pr_auc_gain_over_baseline.svg")
    plot_pr_percent_gain_over_baseline_box(
        models=models,
        predict_features=predict_features,
        rand_seeds=rand_seeds,
        filename=box_path,
    )

    roc_path = os.path.join(FIG_DIR, "roc_auc_SD.svg")
    plot_grouped_mean_roc(
        models,
        predict_features,
        rand_seeds,
        error="sd",
        line_styles=[":", "-", "--", "-.", "-", (0, (3, 10, 1, 15))],
        filename=roc_path,
    )

    features = ["eGFR40", "ESRD", "Steroid_resistant"]
    df_all = pd.DataFrame()
    for feat in features:
        temp = groupwise_metrics_table(
            models=models,
            feature=feat,
            seeds=rand_seeds,
            test_label=test_label_full,
            train_label=train_label,
            group_by="DIAGNOSIS",
        )
        temp["Predicted feature"] = feat
        df_all = pd.concat([df_all, temp], ignore_index=True)

    df_cols = [
        "Predicted feature",
        "Group",
        "Total events in test",
        "Event rate in test (%)",
        "Event rate in train (%)",
        "ROC AUC",
        "PR AUC",
        "Gain in PR AUC (%)",
        "Balanced accuracy",
    ]
    df_export = df_all[df_cols]

    csv_path = os.path.join(FIG_DIR, "groupwise_metrics_by_diagnosis.csv")
    if not os.path.exists(csv_path):
        df_export.to_csv(csv_path, index=False)
    else:
        print(f"[skip] File exists, not overwriting: {csv_path}")

if __name__ == "__main__":
    main()

