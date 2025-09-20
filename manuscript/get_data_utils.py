import pandas as pd 

train_label = pd.read_csv('/data/curegn/curegn_genome/result/ksenia/processed_Ksenia/FINAL/status_train_2023.csv', index_col=0)
test_label_full = pd.read_csv('/data/curegn/curegn_genome/result/ksenia/processed_Ksenia/FINAL/status_test_2023.csv', index_col=0)

def get_subset(trainX, testX, predict_feature, disease_subset=[], ancestry_subset=[], egfr_r2=-1,
               time_cutoff = 2000, use_time_cutoff=False):
    
    trainX_subset = trainX
    testX_subset = testX

    subset_label_train = train_label 
    # always eval on confident observations
    subset_label_test = test_label_full[(test_label_full[predict_feature]==1) | (test_label_full['followup_time']>2000)] 
    
    if use_time_cutoff:
        if predict_feature+'_time' in subset_label_train.columns:
            subset_label_train = subset_label_train[(subset_label_train[predict_feature]==1) | (train_label[predict_feature+'_time']>time_cutoff)]
        else:
            subset_label_train = subset_label_train[(subset_label_train[predict_feature]==1) | (train_label['followup_time']>time_cutoff)]

    if disease_subset!=[]:
        subset_label_train = subset_label_train[subset_label_train['DIAGNOSIS'].isin(disease_subset)]
        subset_label_test = subset_label_test[subset_label_test['DIAGNOSIS'].isin(disease_subset)]
        
    if ancestry_subset!=[]:
        subset_label_train = subset_label_train[subset_label_train['ANCESTRY'].isin(ancestry_subset)]
        subset_label_test = subset_label_test[subset_label_test['ANCESTRY'].isin(ancestry_subset)]
    
    y = subset_label_train[predict_feature].copy().dropna()
    X = trainX_subset.loc[y.index].copy()

    if use_time_cutoff and predict_feature+'_time' in subset_label_train.columns:
        y_test = subset_label_test[[predict_feature, predict_feature+'_time']].copy().dropna(subset=[predict_feature, predict_feature+'_time'])
        time_to_feat = y_test[predict_feature+'_time']
        y_test = y_test[predict_feature].copy()
    else:
        y_test = subset_label_test[predict_feature].copy().dropna()
        time_to_feat = []
    
    if testX is not None:
        X_test = testX_subset.loc[y_test.index].copy()
    else:
        X_test = []
    
    
    return X, y, X_test, y_test, time_to_feat