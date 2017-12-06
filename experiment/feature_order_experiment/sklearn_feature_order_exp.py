import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import constant
import sklearn.metrics as metrics
import experiment.data_order_experiment.data_order_strategy as data_order_strategy
import gc
import itertools


def drop_ng_row(df):
    return df.dropna(subset=['prob_man', 'ctr_user', 'inview_ratio'])


def get_df_from_list_csv_files(filenames):
    return pd.concat(map(pd.read_csv, filenames))


def experiment(train_size, test_size, num_tree, max_depth, sub_feature, random_state, bootstrap, oob_score, max_permutation=200):
    str_detail = '{}_{}_{}_{}_{}_{}_{}_{}'.format(train_size, test_size, num_tree, max_depth, sub_feature, random_state,
                                                  bootstrap, oob_score)

    log_file = constant.get_feature_order_change_experiment_log_file(addition_infor=str_detail)
    with open(log_file, 'w+') as fp:
        fp.write('feature_order|auroc\n')

    df_train = get_df_from_list_csv_files(constant.get_train_files(2, train_size, 6))
    df_test = get_df_from_list_csv_files(constant.get_test_files(2, test_size, 6))
    print('Train size: ', df_train.shape[0])
    print('Test size: ', df_test.shape[0])

    df_train = drop_ng_row(df_train)
    df_test = drop_ng_row(df_test)
    print('Train size clear ng row: ', df_train.shape[0])
    print('Test size clear ng row: ', df_test.shape[0])

    # features_column_names = constant.FEATURES_COLUMNS_NAMES
    label_column_name = 'is_click'

    permutations_of_features_cols = itertools.permutations(constant.FEATURES_COLUMNS_NAMES)

    i = 0
    for features_column_names in permutations_of_features_cols:
        features_column_names = list(features_column_names)
        if i > max_permutation:
            break

        rf = RandomForestClassifier(
            n_estimators=num_tree, max_depth=max_depth,
            max_features=sub_feature, random_state=random_state,
            bootstrap=bootstrap, oob_score=oob_score, n_jobs=-1
        )

        train_features = df_train[features_column_names]
        train_labels = df_train[label_column_name]

        rf.fit(train_features, train_labels)

        predicts = [x[1] for x in rf.predict_proba(df_test[features_column_names])]
        auroc = metrics.roc_auc_score(y_true=df_test[label_column_name], y_score=predicts)

        print(features_column_names, ':', auroc)
        with open(log_file, 'a') as fp:
            fp.write('{}|{}\n'.format(features_column_names, auroc))
        i = i + 1
    print('EXIT SUCCESS !!!')


if __name__ == '__main__':
    train_size = 10
    test_size = 10
    # num_tree = 50
    max_depth = 13
    sub_feature = 'sqrt'
    random_state = 0
    bootstrap = False
    oob_score = False
    max_permutation = 100

    for num_tree in range(10, 260, 10):
        experiment(train_size, test_size, num_tree, max_depth, sub_feature, random_state, bootstrap, oob_score, max_permutation)
