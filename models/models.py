'''
Reco Models
'''

import pandas as pd

K = 10
DAYS = 14


def pop_items_last_14d(df_inter):
    '''
    1st - return a list of top10 most popular items for last 14 days
    '''

    recommendations = []
    df = df_inter.copy()

    min_date = df["last_watch_dt"].max().normalize() - pd.DateOffset(DAYS)
    recommendations = (
        df.loc[df["last_watch_dt"] > min_date, "item_id"]
        .value_counts()
        .head(K)
        .index.values
    )

    return list(recommendations)
