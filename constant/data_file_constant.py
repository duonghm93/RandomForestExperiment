TRAIN_TEST_DATA = 'D:/ReeMo/20171023_20171029/part-r-00008-70d2b416-a459-4c9d-9021-98552fc7b501.csv'

TRAIN_FILE = 'D:/ReeMo/20171016_20171022/part-r-00002-312b8391-c703-4d03-ac1a-111b9b33ec3b.csv'
TESTING_FILE = 'D:/ReeMo/20171023_20171029/part-r-00002-70d2b416-a459-4c9d-9021-98552fc7b501.csv'

TRAIN_FOLDER = 'D:/ReeMo/20171016_20171022/'
TEST_FOLDER = 'D:/ReeMo/20171023_20171029/'


def get_train_files(start=2, end=100, step=6):
    return [TRAIN_FOLDER + 'part-r-' + format(i, '05d') + '-312b8391-c703-4d03-ac1a-111b9b33ec3b.csv' for i in range(start, end, step)]


def get_test_files(start=2, end=20, step=6):
    return [TEST_FOLDER + 'part-r-' + format(i,'05d') + '-70d2b416-a459-4c9d-9021-98552fc7b501.csv' for i in range(start, end, step)]



