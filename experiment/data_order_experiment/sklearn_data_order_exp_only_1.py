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
    log_file = constant.get_data_order_change_experiment_log_file()
    # with open(log_file, 'w+') as fp:
    #     fp.write('strategy_name, auroc\n')

    df_train = get_df_from_list_csv_files(constant.get_train_files(2,10,6))
    df_test = get_df_from_list_csv_files(constant.get_test_files(2,10,6))
    print('Train size: ', df_train.shape[0])
    print('Test size: ', df_test.shape[0])

    df_train = drop_ng_row(df_train)
    df_test = drop_ng_row(df_test)
    print('Train size clear ng row: ', df_train.shape[0])
    print('Test size clear ng row: ', df_test.shape[0])

    features_column_names = constant.FEATURES_COLUMNS_NAMES
    label_column_name = 'is_click'

    # for strategy_name, strategy_function in data_order_strategy.strategies.items():
    data_order_strategies = data_order_strategy._sort_strategy_generator(df_train)
    for strategy in data_order_strategies:
        gc.collect()
        strategy_name = strategy[0]
        strategy_function = strategy[1]
        print('=== {}'.format(strategy_name))

        df_train_sorted = strategy_function(df_train)
        # df_train_sorted = drop_ng_row(df_train_sorted)

        rf = RandomForestClassifier(
            n_estimators=40, max_depth=10,
            max_features='sqrt', random_state=0,
            bootstrap=True, oob_score=False, n_jobs=-1
        )

        train_features = df_train_sorted[features_column_names]
        train_labels = df_train_sorted[label_column_name]

        rf.fit(train_features, train_labels)
        del df_train_sorted
        predicts = [x[1] for x in rf.predict_proba(df_test[features_column_names])]

        auroc = metrics.roc_auc_score(y_true=df_test[label_column_name], y_score=predicts)
        # with open(log_file, 'a') as fp:
        #     fp.write('{},{}\n'.format(strategy_name,auroc))

        print(strategy_name, ':', auroc)

