"""
Volaria - 2nd Module Model Training Script
===============================
Trains RandomForestClassifier models for one or more clinical outcomes using the
preprocessed TRAIN/TEST feature matrices and label CSVs. For each outcome and for
each fixed random seed, it fits a model on TRAIN and evaluates on TEST, storing
the fitted estimator and standard metrics (ROC/PR curves and AUCs). Results are
packed into a single pickle and saved under outputs/models/ without overwriting.

Dependencies and default values
---------------
- get_subset(...) from get_data_utils is used to select (X, y) for each outcome
  (and to apply optional time cutoff), returning aligned TRAIN/TEST splits.
- Outcomes (with use_time_cutoff=True by default):
    ESRD, eGFR40, Steroid_resistant
- Hyperparameters: 
    ESRD:             n_estimators=1000, max_depth=2
    eGFR40:           n_estimators=100,  max_depth=2
    Steroid_resistant:n_estimators=1000, max_depth=2
- Classifier: RandomForestClassifier(class_weight="balanced", criterion="entropy")

CLI flexibility
---------------
You can optionally override the targets via a comma-separated list:
    --targets ESRD,eGFR40
If omitted, the script trains the default three outcomes above. Assumes same file structure

Note: If outputs/models/models_outcomes.pkl already exists, it is not overwritten.
"""
import os
os.environ["SCIPY_ARRAY_API"] = "1"

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

import argparse
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from get_data_utils import get_subset  # uses your existing utility

STATUS_TRAIN = "/data/curegn/curegn_genome/result/ksenia/processed_Ksenia/FINAL/status_train_2023.csv"
STATUS_TEST = "/data/curegn/curegn_genome/result/ksenia/processed_Ksenia/FINAL/status_test_2023.csv"
VERSION_USE = "_2023_3"
TRAIN_X = f"outputs/temp/regressed_v{VERSION_USE}.TRAIN.csv"
TEST_X = f"outputs/temp/regressed_v{VERSION_USE}.TEST.csv"

OUT_DIR = "outputs/models"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PKL = os.path.join(OUT_DIR, "models_outcomes.pkl")

RAND_SEEDS = [13, 37, 42, 73, 132, 57, 101]

DEFAULT_PREDICT_FEATURES = [
    ("ESRD", True),
    ("eGFR40", True),
    ("Steroid_resistant", True),
]

PROVIDE_PARAMS = {
    "ESRD": {"n_estimators": 1000, "max_depth": 2},
    "eGFR40": {"n_estimators": 100,  "max_depth": 2},
    "Steroid_resistant": {"n_estimators": 1000, "max_depth": 2},
}

DISEASE_SUBSET: list[str]  = []
ANCESTRY_SUBSET: list[str] = []


def train_for_outcome(
    X_train_full: pd.DataFrame,
    X_test_full: pd.DataFrame,
    outcome: str,
    use_time_cutoff: bool,
    params: dict,
) -> dict:
    """Train per-seed RFs for a single outcome and return the packed result dict."""
    max_depth     = params["max_depth"]
    n_estimators  = params["n_estimators"]

    X, y, X_test, y_test, _ = get_subset(
        X_train_full,
        X_test_full,
        outcome,
        disease_subset=DISEASE_SUBSET,
        ancestry_subset=ANCESTRY_SUBSET,
        use_time_cutoff=use_time_cutoff,
    )

    result: dict = {
        "trainX": X,
        "testX": X_test,
        "trainY": y,
        "testY": y_test,
    }

    for rs in RAND_SEEDS:
        clf = RandomForestClassifier(
            class_weight="balanced",
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=rs,
            criterion="entropy",
        ).fit(X, y)

        probs = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc_val = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_test, probs)
        avg_precision = average_precision_score(y_test, probs)
        baseline_pr   = float(np.mean(y_test))

        result[rs] = {
            "model": clf,
            "roc_auc": roc_auc_val,
            "fpr": fpr,
            "tpr": tpr,
            "avg_precision": avg_precision,
            "baseline_pr": baseline_pr,
            "precision": precision,
            "recall": recall,
        }
        print(f"[{outcome}] seed={rs} ROC AUC (TEST): {roc_auc_val:.4f}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Volaria RF models for outcomes.")
    parser.add_argument(
        "--targets",
        type=str,
        default="",
        help="Comma-separated outcomes to train (e.g., 'ESRD, eGFR40'). If empty, use defaults.",
    )
    return parser.parse_args()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load labels and matrices  
    train_label = pd.read_csv(STATUS_TRAIN, index_col=0)
    test_label  = pd.read_csv(STATUS_TEST, index_col=0)
    trainX = pd.read_csv(TRAIN_X, index_col=0).T
    testX  = pd.read_csv(TEST_X,  index_col=0).T
    print(trainX.shape, testX.shape)

    # Determine targets  
    args = parse_args()
    if args.targets.strip():
        targets = [(t.strip(), True) for t in args.targets.split(",") if t.strip()]
    else:
        targets = DEFAULT_PREDICT_FEATURES

    # Train
    models: dict = {}
    for outcome, use_time_cutoff in targets:
        if outcome not in PROVIDE_PARAMS:
            raise ValueError(f"No params provided for outcome '{outcome}'")
        print("*" * 10, outcome)
        models[outcome] = train_for_outcome(
            X_train_full=trainX,
            X_test_full=testX,
            outcome=outcome,
            use_time_cutoff=use_time_cutoff,
            params=PROVIDE_PARAMS[outcome],
        )

    if os.path.exists(OUT_PKL):
        print(f"[skip] File exists, not overwriting: {OUT_PKL}")
    else:
        with open(OUT_PKL, "wb") as f:
            pickle.dump(models, f)
        print(f"[ok] Saved: {OUT_PKL}")


if __name__ == "__main__":
    main()

