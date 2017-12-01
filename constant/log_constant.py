LOG_FOLDER = 'D:/GitProject/RandomForestExperiment/log/'
DATA_ORDER_CHANGE_EXPERIMENT_FOLDER = 'data_order_change_exp/'
import time


def get_data_order_change_experiment_log_folder():
    return LOG_FOLDER + DATA_ORDER_CHANGE_EXPERIMENT_FOLDER


def get_data_order_change_experiment_log_file(spid=6968, addition_infor=''):
    current_time = time.strftime('%Y_%m_%d_%H_%M_%S')
    return get_data_order_change_experiment_log_folder() + str(spid) + '_' + addition_infor + '_' + current_time + '.csv'