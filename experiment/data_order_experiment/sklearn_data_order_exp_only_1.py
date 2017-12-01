import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import constant
import sklearn.metrics as metrics
import experiment.data_order_experiment.data_order_strategy as data_order_strategy
import gc


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
    # print('Loading data ...')
    return pd.concat(map(pd.read_csv, filenames))


if __name__ == '__main__':
    train_size = 100
    test_size = 10
    num_tree = 40
    max_depth=10
    sub_feature = 'sqrt'
    random_state=0
    bootstrap=False
    oob_score=False

    str_detail = '{}_{}_{}_{}_{}_{}_{}_{}'.format(train_size, test_size, num_tree, max_depth, sub_feature, random_state, bootstrap, oob_score)

    log_file = constant.get_data_order_change_experiment_log_file(addition_infor=str_detail)
    with open(log_file, 'w+') as fp:
        fp.write('strategy_name, auroc\n')

    df_train = get_df_from_list_csv_files(constant.get_train_files(2,train_size,6))
    df_test = get_df_from_list_csv_files(constant.get_test_files(2,test_size,6))
    print('Train size: ', df_train.shape[0])
    print('Test size: ', df_test.shape[0])

    df_train = drop_ng_row(df_train)
    df_test = drop_ng_row(df_test)
    print('Train size clear ng row: ', df_train.shape[0])
    print('Test size clear ng row: ', df_test.shape[0])

    features_column_names = constant.FEATURES_COLUMNS_NAMES
    label_column_name = 'is_click'

    for col_name in (['nothing'] + list(df_train.columns)):
        gc.collect()
        print('=== {}'.format(col_name))
        if col_name == 'nothing':
            df_train_sorted = df_train
        else:
            df_train_sorted = df_train.sort_values(col_name)

        rf = RandomForestClassifier(
            n_estimators=num_tree, max_depth=max_depth,
            max_features=sub_feature, random_state=random_state,
            bootstrap=bootstrap, oob_score=oob_score, n_jobs=-1
        )

        train_features = df_train_sorted[features_column_names]
        train_labels = df_train_sorted[label_column_name]

        rf.fit(train_features, train_labels)
        del df_train_sorted
        predicts = [x[1] for x in rf.predict_proba(df_test[features_column_names])]

        auroc = metrics.roc_auc_score(y_true=df_test[label_column_name], y_score=predicts)
        with open(log_file, 'a') as fp:
            fp.write('{},{}\n'.format(col_name,auroc))

        print(col_name, ':', auroc)

