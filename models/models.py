import pandas as pd

# Models
# 1st - return a list of top10 most popular items for last 14 days

def pop_14d(df_iter):

  K = 10
  days = 14
  recommendations = []
  df_1 = df_iter.copy()

  min_date = df_1['last_watch_dt'].max().normalize() - pd.DateOffset(days)
  recommendations = df_1.loc[df_1['last_watch_dt'] > min_date, 'item_id'].value_counts().head(K).index.values

  return list(recommendations)