import os
os.environ["SCIPY_ARRAY_API"] = "1"
import pandas as pd

import warnings; warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_recall_curve, average_precision_score

import numpy as np
import pandas as pd

import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

from get_data_utils import get_subset 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


rename_dict = {
    'eGFR40': 'eGFR decline',
    'ESRD': 'Kidney disease', 
    'Steroid_resistant': 'Steroid resistant',
}

# import train and test data
train_label = pd.read_csv('status_train.csv', index_col=0)
test_label_full = pd.read_csv('status_test.csv', index_col=0)

version_use = '_3' 
trainX = pd.read_csv('outputs/temp/regressed_v{}.TRAIN.csv'.format(version_use), index_col=0).T
testX = pd.read_csv('outputs/temp/regressed_v{}.TEST.csv'.format(version_use), index_col=0).T

print(trainX.shape, testX.shape)

predict_feature = 'ESRD' 
disease_subset = [] 
ancestry_subset = [] 

X, y, X_test, y_test, _ = get_subset(trainX, testX, predict_feature, 
                                  use_time_cutoff=False)

n_bootstraps = 50
bootstrap_ids = list(range(n_bootstraps))

predict_features = [('ESRD', True),
                    ('eGFR40', True), 
                    ('Steroid_resistant', True),
                    ]

provide_params = {
    'ESRD': {
        'n_estimators': 1000,
        'max_depth': 2,       
    },     
    'eGFR40': {
        'n_estimators': 100,
        'max_depth': 2,    
    },    
    'Steroid_resistant': {
        'n_estimators': 1000,
        'max_depth': 2,       
    },   
}
                    

disease_subset = [] 
ancestry_subset = []  

models = {}

for i, predict_feature in enumerate(predict_features):
    predict_feature, use_time_cutoff = predict_feature
    
    if predict_feature in provide_params:
        max_depth = provide_params[predict_feature]['max_depth']
        n_estimators = provide_params[predict_feature]['n_estimators']        
    else:
        raise ValueError

    print('*'*10, predict_feature)
    
    models[predict_feature] = {}
    X, y, X_test, y_test, _ = get_subset(
        trainX, testX, predict_feature,
        use_time_cutoff=use_time_cutoff
    )

    models[predict_feature]['trainX'] = X
    models[predict_feature]['testX'] = X_test
    models[predict_feature]['trainY'] = y
    models[predict_feature]['testY'] = y_test

    n_test = len(y_test)
    rng = np.random.default_rng(12345)

    # optional: one full-fit model for reference on the full test set
    clf_full = RandomForestClassifier(
        class_weight='balanced',
        max_depth=max_depth, 
        n_estimators=n_estimators,
        random_state=42, 
        criterion='entropy'
    ).fit(X, y)

    full_score = clf_full.predict_proba(X_test)[:, 1]
    fpr_full, tpr_full, _ = roc_curve(y_test, full_score)
    roc_auc_full = auc(fpr_full, tpr_full)
    avg_precision_full = average_precision_score(y_test, full_score)

    models[predict_feature]['full_test'] = {
        'roc_auc': roc_auc_full,
        'fpr': fpr_full,
        'tpr': tpr_full,
        'avg_precision': avg_precision_full,
        'baseline_pr': np.mean(y_test),
    }

    print('ROC AUC TEST (full test set)', roc_auc_full)

    for b in bootstrap_ids:
        if b % 25 == 0:
            print('*** Replicate:', b)

        models[predict_feature][b] = {}

        model_seed = 1000 + b
        boot_idx = rng.choice(np.arange(n_test), size=n_test, replace=True)

        X_test_boot = X_test.iloc[boot_idx]
        y_test_boot = y_test[boot_idx]

        if len(np.unique(y_test_boot)) < 2:
            continue

        clf = RandomForestClassifier(
            class_weight='balanced',
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=model_seed,
            criterion='entropy'
        ).fit(X, y)

        y_score = clf.predict_proba(X_test_boot)[:, 1]

        fpr_test, tpr_test, _ = roc_curve(y_test_boot, y_score)
        roc_auc_test = auc(fpr_test, tpr_test)
        precision, recall, _ = precision_recall_curve(y_test_boot, y_score)
        avg_precision = average_precision_score(y_test_boot, y_score)
        baseline_pr = np.mean(y_test_boot)

        models[predict_feature][b]['roc_auc'] = roc_auc_test
        models[predict_feature][b]['fpr'] = fpr_test
        models[predict_feature][b]['tpr'] = tpr_test
        models[predict_feature][b]['avg_precision'] = avg_precision
        models[predict_feature][b]['baseline_pr'] = baseline_pr
        models[predict_feature][b]['precision'] = precision
        models[predict_feature][b]['recall'] = recall
        models[predict_feature][b]['boot_idx'] = boot_idx
        models[predict_feature][b]['model_seed'] = model_seed
        
def make_robustness_summary_table(models, predict_features, rep_ids, rename_dict=None):
    records = []

    for feat in predict_features:
        feat, _ = feat if isinstance(feat, tuple) else (feat, None)

        aucs = []
        aps = []
        baselines = []
        gains = []

        for r in rep_ids:
            if r not in models[feat]:
                continue
            if 'roc_auc' not in models[feat][r]:
                continue

            auc_val = models[feat][r]['roc_auc']
            ap_val = models[feat][r]['avg_precision']
            base_val = models[feat][r]['baseline_pr']
            gain_val = (ap_val - base_val) / base_val * 100 if base_val > 0 else 0

            aucs.append(auc_val)
            aps.append(ap_val)
            baselines.append(base_val)
            gains.append(gain_val)

        if len(aucs) == 0:
            continue

        aucs = np.array(aucs)
        aps = np.array(aps)
        baselines = np.array(baselines)
        gains = np.array(gains)

        feature_name = rename_dict.get(feat, feat) if rename_dict is not None else feat

        full_auc = models[feat]['full_test']['roc_auc'] if 'full_test' in models[feat] else np.nan
        full_ap = models[feat]['full_test']['avg_precision'] if 'full_test' in models[feat] else np.nan
        full_base = models[feat]['full_test']['baseline_pr'] if 'full_test' in models[feat] else np.nan
        full_gain = (full_ap - full_base) / full_base * 100 if full_base > 0 else np.nan

        records.append({
            'feature': feat,
            'label': feature_name,
            'n_reps': len(aucs),

            'full_auc': full_auc,
            'mean_auc': aucs.mean(),
            'sd_auc': aucs.std(ddof=1),
            'auc_mean_sd': f'{aucs.mean():.3f} ± {aucs.std(ddof=1):.3f}',

            'full_ap': full_ap,
            'mean_ap': aps.mean(),
            'sd_ap': aps.std(ddof=1),
            'ap_mean_sd': f'{aps.mean():.3f} ± {aps.std(ddof=1):.3f}',

            'mean_baseline_pr': baselines.mean(),
            'sd_baseline_pr': baselines.std(ddof=1),

            'full_pr_gain': full_gain,
            'mean_pr_gain': gains.mean(),
            'sd_pr_gain': gains.std(ddof=1),
            'pr_gain_mean_sd': f'{gains.mean():.2f} ± {gains.std(ddof=1):.2f}',
        })

    df = pd.DataFrame(records)

    return df[['feature', 'label', 'n_reps',
               'full_auc', 'auc_mean_sd',
               'full_ap', 'ap_mean_sd',
               'mean_baseline_pr',
               'full_pr_gain', 'pr_gain_mean_sd',
               'mean_auc', 'sd_auc',
               'mean_ap', 'sd_ap',
               'mean_pr_gain', 'sd_pr_gain']].sort_values('feature')
    
rep_ids = [r for r in range(n_bootstraps)]
summary_df = make_robustness_summary_table(
    models,
    predict_features,
    rep_ids,
    rename_dict=rename_dict
)

summary_df[['feature', 'label', 'n_reps', 'auc_mean_sd', 'ap_mean_sd']].to_csv('outputs/temp/robustness_summary_table.csv', index=False)
