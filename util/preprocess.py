# def preproc(df):
#     df = _drop_ng_rows(df)
#     df = _fill_na(df)
#     if df.count() != df.na.drop().count():
#         print('There are NULL values in the processed data. Something wrong.')
#         exit(1)
#     return df
#
#
# def _drop_ng_rows(df):
#     not_rtg = df['recency'] < 1e-10
#     null_gender = df['prob_man'].isNull()
#     null_ctr_user = df['ctr_user'].isNull() | (df['ctr_user'] < 1e-10)
#     unknown_slot = df['inview_ratio'].isNull()
#     mask_drop = (not_rtg & null_gender & null_ctr_user) | unknown_slot
#     return df.filter(~mask_drop)
#
#
# def _fill_na(df):
#     df = df.na.fill({'prob_man': -1.0, 'ctr_user': -1.0})
#     df = df.withColumn('recency', F.when(df['recency'] < 1e-10, -1).otherwise(df['recency'])) \
#         .withColumn('inview_recency', F.when(df['inview_recency'] < 1e-10, -1).otherwise(df['inview_recency']))
#     return df