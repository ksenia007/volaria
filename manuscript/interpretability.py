"""
Volaria – SHAP Importances, Gene Overlaps, and Mutation Effects

This script is required to reproduce interpretability results, including Figure 4, as well as Supp. Figure 3
Computes per-feature SHAP (mean |value| across samples) across seeds for
eGFR40/ESRD/Steroid_resistant; exports gene lists and per-outcome weights,
plots heatmap for genes shared by all three outcomes, and evaluates mutation
effects (feature zeroing / scaling) across TRAIN/TEST.
"""
import os
os.environ["SCIPY_ARRAY_API"] = "1"

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind


FIG_DIR = "outputs/figures_verify"
os.makedirs(FIG_DIR, exist_ok=True)

MODELS_PKL = "outputs/temp/models_vF2_2023_seeds.pkl"
STATUS_TRAIN = "/data/curegn/curegn_genome/result/ksenia/processed_Ksenia/FINAL/status_train_2023.csv"
STATUS_TEST = "/data/curegn/curegn_genome/result/ksenia/processed_Ksenia/FINAL/status_test_2023.csv"
VERSION_USE = "_2023_3"
TRAIN_X = f"outputs/temp/regressed_v{VERSION_USE}.TRAIN.csv"
TEST_X  = f"outputs/temp/regressed_v{VERSION_USE}.TEST.csv"
DROP_MAP = f"outputs/temp/drop_map_v{VERSION_USE}.pkl"

color_dict_primary = {
    "ESRD": "#034C8C",
    "eGFR40": "#8C1C03",
    "Steroid_resistant": "#F29F05",
}

rand_seeds = [13, 37, 42, 73, 132, 57, 101]

def set_style(bg="white"):
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


def main() -> None:
    set_style("white")

    with open(MODELS_PKL, "rb") as f:
        models = pickle.load(f)

    train_label = pd.read_csv(STATUS_TRAIN, index_col=0)
    test_label_full = pd.read_csv(STATUS_TEST, index_col=0)

    trainX = pd.read_csv(TRAIN_X, index_col=0).T
    testX  = pd.read_csv(TEST_X,  index_col=0).T

    print(trainX.shape, testX.shape)

    with open(DROP_MAP, "rb") as f:
        coll_feats = pickle.load(f)

    # SHAP feature importances
    dict_pds = {}
    use_shap = True
    predict_features = [
        ("eGFR40", True),
        ("ESRD", True),
        ("Steroid_resistant", True),
    ]

    for predict_feature in predict_features:
        feature_imps = []
        predict_feature, _ = predict_feature
        print("*" * 10, predict_feature)
        for rs in rand_seeds:
            print(rs)
            if not use_shap:
                feature_imps.append(models[predict_feature][rs]["model"].feature_importances_)
            else:
                background_sample = models[predict_feature]["trainX"]
                explainer = shap.TreeExplainer(
                    models[predict_feature][rs]["model"],
                    background_sample,
                    model_output="probability",
                    feature_perturbation="interventional",
                )
                explanation = explainer(background_sample)
                shap_values = explanation.values
                feature_imps.append(np.abs(shap_values[:, :, 1]).mean(0))

        df = pd.DataFrame(feature_imps,
                          columns=models[predict_feature]["trainX"].columns,
                          index=rand_seeds).T
        dict_pds[predict_feature] = df


    feats = ["ESRD", "eGFR40", "Steroid_resistant"]
    for f in feats:
        print(f)
        feat_weights = dict_pds[f].min(1).sort_values(ascending=False)
        print(feat_weights.shape)

        names_idx = feat_weights[feat_weights > 0].index
        names = list(set([i.split("_")[0] for i in names_idx]))
        print("Total of {} genes".format(len(names)))
        print(",".join(names))

        p_all = os.path.join(FIG_DIR, f"{f}_weights_min_all_seeds_ALL.csv")
        if not os.path.exists(p_all):
            feat_weights.to_csv(p_all)

        p_nonzero = os.path.join(FIG_DIR, f"{f}_weights_min_all_seeds_above_zero.csv")
        if not os.path.exists(p_nonzero):
            feat_weights[feat_weights > 0].to_csv(p_nonzero)

        p_genes = os.path.join(FIG_DIR, f"{f}_weights_min_all_seeds_above_zero_GENES.txt")
        if not os.path.exists(p_genes):
            with open(p_genes, "w") as fh:
                fh.write(",".join(names))


    def get_top_genes_nonzero(model_name):
        feat_weights = dict_pds[model_name].min(1)
        top_feats = feat_weights[feat_weights > 0].index
        genes = set([i.split("_")[0] for i in top_feats])
        return genes

    genes_eGFR40 = get_top_genes_nonzero("eGFR40")
    genes_ESRD = get_top_genes_nonzero("ESRD")
    genes_3 = get_top_genes_nonzero("Steroid_resistant")

    overlap_three = genes_eGFR40 & genes_ESRD & genes_3
    for t in overlap_three:
        try:
            print(t, coll_feats[t])
        except Exception:
            print(t, "Not in coll_feats")

    overlap_two = genes_eGFR40 & genes_ESRD
    print("Genes overlapping in all ESRD & eGFR ({} genes):".format(len(overlap_two)))
    if overlap_two:
        print(", ".join(sorted(overlap_two)))
    else:
        print("None")

    # Save all-three as list (txt) AND as a CSV with per-outcome weights (no overwrite)
    p_three_list = os.path.join(FIG_DIR, "overlap_All_three.txt")
    if overlap_three and not os.path.exists(p_three_list):
        with open(p_three_list, "w") as fh:
            fh.write(",".join(sorted(overlap_three)))

    # per-outcome weights per gene in the all-three set (use min across seeds per feature, then max within gene)
    if overlap_three:
        out_rows = []
        for g in sorted(overlap_three):
            row = {"gene": g}
            for m in ["ESRD", "eGFR40", "Steroid_resistant"]:
                s_min = dict_pds[m].min(1)  # per-feature (across seeds) importance
                feats_for_gene = [ix for ix in s_min.index if ix.split("_")[0] == g]
                row[m] = float(s_min.loc[feats_for_gene].max()) if feats_for_gene else 0.0
            out_rows.append(row)
        df_all_three = pd.DataFrame(out_rows)
        p_three_csv = os.path.join(FIG_DIR, "overlap_All_three_with_per_outcome_weights.csv")
        if not os.path.exists(p_three_csv):
            df_all_three.to_csv(p_three_csv, index=False)


    overlap_three = genes_eGFR40 & genes_ESRD & genes_3
    model_names = ["ESRD", "eGFR40", "Steroid_resistant"]

    median_for_ovelap = {}
    for model_name in model_names:
        feat_weights = dict_pds[model_name].median(1) / dict_pds[model_name].median(1).max()
        # aggregate to gene-level (max across features for the gene), then keep overlap genes
        feat_weights_gene = feat_weights.groupby(lambda ix: ix.split("_")[0]).max()
        median_for_ovelap[model_name] = feat_weights_gene[feat_weights_gene.index.isin(overlap_three)]

    median_for_ovelap = pd.DataFrame(median_for_ovelap)
    if not median_for_ovelap.empty:
        median_for_ovelap = median_for_ovelap.sort_values("eGFR40", ascending=False)

        plt.figure(figsize=(3, 4), dpi=300)
        sns.heatmap(
            median_for_ovelap,
            cmap="Reds",
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Scaled Importance"},
        )
        plt.xlabel("")
        plt.ylabel("")
        out_heat = os.path.join(
            FIG_DIR, "heatmap_genes_overlapping_all_models_median_scaled_median_max.svg"
        )
        if not os.path.exists(out_heat):
            plt.savefig(out_heat, dpi=300, bbox_inches="tight", format="svg", transparent=True)
        plt.show()
        plt.close()

    # Mutation effect: loop over splits and matches
    feature = "eGFR40"
    for split in ["trainX", "testX"]:
        for match in ["RPS6KA1_weighted_AM", "EYA2_glomerularendothelium"]:
            pred_feature_vary = [i for i in models[feature][split].columns if match in i]
            print(pred_feature_vary)
            if not pred_feature_vary:
                continue
            rs = 13

            X_base = models[feature][split].copy()
            ground_truth_preds = models[feature][rs]["model"].predict_proba(X_base)[:, 1]

            X_low = X_base.copy()
            X_low[pred_feature_vary] = 0
            new_preds_low = models[feature][rs]["model"].predict_proba(X_low)[:, 1]

            X_high = X_base.copy()
            X_high[pred_feature_vary] = X_high[pred_feature_vary] * 100
            new_preds_high = models[feature][rs]["model"].predict_proba(X_high)[:, 1]

            plot_df = pd.DataFrame(
                {
                    "ground_truth_preds": ground_truth_preds,
                    "Low": new_preds_low,
                    "High": new_preds_high,
                }
            )
            plot_df["Low"]  = (plot_df["Low"]  - plot_df["ground_truth_preds"]) * 100 / plot_df["ground_truth_preds"]
            plot_df["High"] = (plot_df["High"] - plot_df["ground_truth_preds"]) * 100 / plot_df["ground_truth_preds"]

            stat, p_value = mannwhitneyu(plot_df["Low"], plot_df["High"])
            print(f"[{split} | {match}] Mann-Whitney U: {stat}, p={p_value}")

            plot_df_reshape = pd.melt(
                plot_df,
                id_vars=["ground_truth_preds"],
                value_vars=["Low", "High"],
                var_name="Mutated group",
                value_name="new prediction",
            )

            fig, ax = plt.subplots(figsize=(4, 6), dpi=300)
            sns.pointplot(
                data=plot_df_reshape,
                x="Mutated group",
                y="new prediction",
                errorbar="sd",
                join=False,
                capsize=0.1,
                markers="o",
                color=color_dict_primary[feature],
                ax=ax,
                scale=5.5,
                errwidth=4,
            )
            if p_value < 0.05:
                ax.annotate("***", xy=(0.5, 0.98), xycoords="axes fraction", fontsize=20, ha="center")
                ax.plot([0.25, 0.75], [0.97, 0.97], transform=ax.transAxes, color="black", lw=1.5)
            ax.plot([-0.5, 1.5], [0, 0], color="grey", lw=1.5, ls="--")
            plt.xlabel("")
            plt.ylabel("% change in predicted probability of outcome", fontsize=14)
            plt.yticks(fontsize=17)
            plt.xticks(fontsize=17)
            sns.despine(ax=ax)
            plt.tight_layout()
            plt.xlim([-0.5, 1.5])

            out_mut = os.path.join(FIG_DIR, f"{match}_mutation_{feature}_{split}.svg")
            if not os.path.exists(out_mut):
                plt.savefig(out_mut, dpi=300, bbox_inches="tight", format="svg", transparent=True)
            plt.show()
            plt.close()

    # plot eGFR distribution in low vs high groups
    feature = "eGFR40"
    pred_features_vary = ["EYA2_glomerularendothelium", "RPS6KA1_weighted_AM",]
    for pred_feature_vary in pred_features_vary:
        trainX_mut = models[feature]["trainX"].copy()
        print('requested feature:', pred_feature_vary)
        pred_feature_vary = [i for i in trainX_mut.columns if pred_feature_vary in i][0]
        print('matched features:', pred_feature_vary)
        vals = trainX_mut[pred_feature_vary].values
        cutoff_low = np.quantile(vals, 0.1)
        cutoff_high = np.quantile(vals, 0.9)
        print(cutoff_low, cutoff_high)

        bottom_n = trainX_mut[pred_feature_vary][trainX_mut[pred_feature_vary] <= cutoff_low]
        top_n = trainX_mut[pred_feature_vary][trainX_mut[pred_feature_vary] >= cutoff_high]

        bottom_n = bottom_n.index
        top_n = top_n.index
        bottom_n = train_label.loc[train_label.index.isin(bottom_n)].dropna(subset=["eGFR"])
        top_n = train_label.loc[train_label.index.isin(top_n)].dropna(subset=["eGFR"])

        print("N people in bottom group:", len(bottom_n))
        print("N people in top group:", len(top_n))

        stat, p_value = mannwhitneyu(bottom_n.eGFR, top_n.eGFR)
        print(f"Mann-Whitney U statistic: {stat}, p-value: {p_value}")

        stat, p_value = ttest_ind(bottom_n.eGFR, top_n.eGFR)
        print(f"T-test statistic: {stat}, p-value: {p_value}")

        plt.figure(figsize=(4, 5), dpi=600)
        sns.kdeplot(top_n.eGFR, label="High (top 10%)", color="black", lw=2, ls="--")
        sns.kdeplot(bottom_n.eGFR, label="Low (bottom 10%)", color="grey", lw=2)
        plt.xlabel("eGFR at intake")
        plt.ylabel("Density")
        plt.text(
            0.95,
            0.95,
            f"p = {p_value:.2e}",
            horizontalalignment="right",
            verticalalignment="center",
            transform=plt.gca().transAxes,
            fontsize=14,
        )
        plt.legend()
        out_kde = os.path.join(FIG_DIR, f"{pred_feature_vary}_low_high_eGFR_{feature}.svg")
        if not os.path.exists(out_kde):
            plt.savefig(out_kde, dpi=300, bbox_inches="tight", format="svg", transparent=True)
        sns.despine()
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()



