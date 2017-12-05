from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import constant
from pyspark.sql.functions import rand

def drop_ng_row(df):
    return df.na.drop()


if __name__ == '__main__':
    spark = SparkSession.builder.appName("train_ctr_model") \
        .config('spark.kryoserializer.buffer.max', '1600m') \
        .getOrCreate()
    df_test = spark.read.csv('D:/ReeMo/20171016_20171022/', header=True, inferSchema=True)
    df_train = spark.read.csv('D:/ReeMo/20171023_20171029/', header=True, inferSchema=True)
    print('Train data size:',df_train.count())
    print('Test data size:',df_test.count())

    df_train = drop_ng_row(df_train)
    df_test = drop_ng_row(df_test)
    print('Train data after clear ng size:', df_train.count())
    print('Test data after clear ng size:', df_test.count())

    assembler = VectorAssembler(inputCols=constant.FEATURES_COLUMNS_NAMES, outputCol='features')
    features_test = assembler.transform(df_test)

    for shuffle_seed in range(85, 100, 5):
        if shuffle_seed == 0:
            df_train_sort = df_train
        else:
            df_train_sort = df_train.orderBy(rand(shuffle_seed))

        features = assembler.transform(df_train_sort)

        rf = RandomForestClassifier(
            numTrees=40, maxDepth=10, featuresCol='features',
            labelCol='is_click', featureSubsetStrategy='sqrt',
            seed=0, impurity='gini'
        )
        model = rf.fit(features)

        features.unpersist()
        df_train_sort.unpersist()

        predictions = model.transform(features_test)

        preds = predictions.select('probability','is_click').rdd.map(lambda row: (row[0][1], row[1])).collect()

        evaluator = BinaryClassificationEvaluator(labelCol='is_click', metricName='areaUnderROC')
        auroc = evaluator.evaluate(predictions)
        print('shuffle seed ', str(shuffle_seed), ':', auroc)

        predictions.unpersist()
