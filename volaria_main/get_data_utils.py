import pandas as pd 



def get_subset(trainX, testX, predict_feature, 
               time_cutoff = 2000, use_time_cutoff=False, 
               train_loc='status_train.csv', test_loc='status_test.csv'):
    
    train_label = pd.read_csv(train_loc, index_col=0)
    test_label_full = pd.read_csv(test_loc, index_col=0)
    
    trainX_subset = trainX
    testX_subset = testX

    subset_label_train = train_label 
    subset_label_test = test_label_full[(test_label_full[predict_feature]==1) | (test_label_full['followup_time']>2000)] 
    
    if use_time_cutoff:
        if predict_feature+'_time' in subset_label_train.columns:
            subset_label_train = subset_label_train[(subset_label_train[predict_feature]==1) | (train_label[predict_feature+'_time']>time_cutoff)]
        else:
            subset_label_train = subset_label_train[(subset_label_train[predict_feature]==1) | (train_label['followup_time']>time_cutoff)]
    
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