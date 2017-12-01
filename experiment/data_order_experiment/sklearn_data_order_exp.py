import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import constant
import sklearn.metrics as metrics
import math


def get_isclick_num_and_all_num_of_data(df):
    all_num = df.shape[0]
    is_click_num = df[df.is_click == 1].shape[0]
    return (is_click_num, all_num)


def get_features_labels_from_df(df):
    features = df[constant.FEATURES_COLUMNS_NAMES]
    labels = df[[constant.IS_CLICK_COLUMN_NAME]]
    return features, labels


def check_na(df):
    return df.isnull().any()


def drop_ng_row(df):
    return df.dropna(subset=['prob_man', 'ctr_user', 'inview_ratio'])


def get_df_from_list_csv_files(filenames):
    print('Loading data ...')
    return pd.concat(map(pd.read_csv, filenames))


if __name__ == '__main__':
    # df_train = pd.read_csv(constant.TRAIN_FILE)
    df_train = get_df_from_list_csv_files(constant.get_train_files(2,10,6))
    df_test = get_df_from_list_csv_files(constant.get_test_files(2,10,6))

    features_column_names = constant.FEATURES_COLUMNS_NAMES
    label_column_name = 'is_click'

    print('Train size: ', df_train.shape[0])
    print('Test size: ', df_test.shape[0])

    df_train = drop_ng_row(df_train)
    df_test = drop_ng_row(df_test)

    print('Train size clear ng row: ', df_train.shape[0])
    print('Test size clear ng row: ', df_test.shape[0])

    rf = RandomForestClassifier(n_estimators=40, max_depth=10, max_features='sqrt', random_state=0, bootstrap=True, oob_score=False, n_jobs=-1)

    train_features = df_train[features_column_names]
    train_labels = df_train[label_column_name]

    print('Start training ...')
    rf.fit(train_features, train_labels)
    print('Finish training ...')

    # test_features, test_labels = get_features_labels_from_df(df_test)
    print('Predicting ...')
    predicts = [x[1] for x in rf.predict_proba(df_test[features_column_names])]
    
    auroc = metrics.roc_auc_score(y_true=df_test[label_column_name], y_score=predicts)
    print(auroc)




