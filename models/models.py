"""
Reco Models
"""

from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd
import scipy as sp
from implicit.nearest_neighbours import ItemItemRecommender

K = 10
DAYS = 14


def pop_items_last_14d(df_inter: pd.DataFrame) -> list:
    """
    1st - return a list of top10 most popular items for last 14 days
    """

    recommendations = []
    df = df_inter.copy()

    min_date = df["datetime"].max().normalize() - pd.DateOffset(DAYS)
    recommendations = (
        df.loc[df["datetime"] > min_date, "item_id"]
          .value_counts().head(K).index.values
    )

    return list(recommendations)


def mix_popular_items(df_inter: pd.DataFrame, number: int) -> list:
    """
    Return list consisting of input number(max=10) mix of popular items.
    near 50% of items - from list top 10 last 30 days
    near 30% of items - from list top 10 last 90 days
    near 20% of items - from list top 10 for all time
    """
    df = df_inter.copy()

    # np.array top@k pop for different periods
    def popoular_number_of_items_days(
        df: pd.DataFrame, k: int = 10, days: int = 14, all_time: bool = False
    ) -> np.array:
        """
        Return a np.array of top@k most popular items for last N days
        """
        recommendations = []

        if all_time is True:
            recommendations = df.loc[:, "item_id"].value_counts().head(k).index.values
        else:
            min_date = df["datetime"].max().normalize() - pd.DateOffset(days)
            recommendations = (
                df.loc[df["datetime"] > min_date, "item_id"]
                .value_counts()
                .head(k)
                .index.values
            )
        return recommendations

    # all duplicates will be deleting and adding some items from pop_all_time
    def del_repeat_items(arr_mix_pop: np.array) -> np.array:
        """
        Delete all duplicates items in array
        """
        if len(set(arr_mix_pop)) == number:
            return arr_mix_pop
        else:
            i = number - len(set(arr_mix_pop))
            arr_mix_pop = np.concatenate(
                (arr_mix_pop, np.random.choice(pop_all_time, i, replace=False))
            )
            return del_repeat_items(arr_mix_pop)

    arr_mix_pop = np.array([])
    pop_30d = popoular_number_of_items_days(df, k=10, days=30)
    pop_90d = popoular_number_of_items_days(df, k=10, days=90)
    pop_all_time = popoular_number_of_items_days(df, k=10, all_time=True)

    if number == 1:
        i, j, k = 1, 0, 0
    elif number == 2:
        i, j, k = 1, 1, 0
    elif number == 3:
        i, j, k = 2, 1, 0
    elif number == 4:
        i, j, k = 2, 1, 1
    elif number == 5:
        i, j, k = 3, 1, 1
    elif number == 6:
        i, j, k = 3, 2, 1
    elif number == 7:
        i, j, k = 4, 2, 1
    elif number == 8:
        i, j, k = 4, 2, 2
    elif number == 9:
        i, j, k = 5, 2, 2
    elif number == 10:
        i, j, k = 5, 3, 2
    else:
        arr_mix_pop = np.array([])
        return list(arr_mix_pop)

    arr_mix_pop = np.concatenate(
        (
            np.random.choice(pop_30d, i, replace=False),
            np.random.choice(pop_90d, j, replace=False),
            np.random.choice(pop_all_time, k, replace=False),
        )
    )

    arr_mix_pop = del_repeat_items(arr_mix_pop)

    return list(set(arr_mix_pop))


class my_UserKnn:
    """Class for fit-perdict UserKNN model
    based on ItemKNN model from implicit.nearest_neighbours
    """

    def __init__(self, model: ItemItemRecommender, N_users: int = 50):
        self.N_users = N_users
        self.model = model
        self.is_fitted = False

    def get_mappings(self, train):
        self.users_inv_mapping = dict(enumerate(train["user_id"].unique()))
        self.users_mapping = {v: k for k, v in self.users_inv_mapping.items()}

        self.items_inv_mapping = dict(enumerate(train["item_id"].unique()))
        self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}

    def get_matrix(
        self,
        df: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        weight_col: str = None,
        users_mapping: Dict[int, int] = None,
        items_mapping: Dict[int, int] = None,
    ):

        if weight_col:
            weights = df[weight_col].astype(np.float32)
        else:
            weights = np.ones(len(df), dtype=np.float32)

        interaction_matrix = sp.sparse.coo_matrix(
            (
                weights,
                (
                    df[user_col].map(self.users_mapping.get),
                    df[item_col].map(self.items_mapping.get),
                ),
            )
        )

        self.watched = df.groupby(user_col).agg({item_col: list})
        return interaction_matrix

    def idf(self, n: int, x: float):
        return np.log((1 + n) / (1 + x) + 1)

    def _count_item_idf(self, df: pd.DataFrame):
        item_cnt = Counter(df["item_id"].values)
        item_idf = pd.DataFrame.from_dict(
            item_cnt, orient="index", columns=["doc_freq"]
        ).reset_index()
        item_idf["idf"] = item_idf["doc_freq"].apply(lambda x: self.idf(self.n, x))
        self.item_idf = item_idf

    def fit(self, train: pd.DataFrame):
        self.user_knn = self.model
        self.get_mappings(train)
        self.weights_matrix = self.get_matrix(
            train, users_mapping=self.users_mapping, items_mapping=self.items_mapping
        )

        self.n = train.shape[0]
        self._count_item_idf(train)

        self.user_knn.fit(self.weights_matrix)
        self.is_fitted = True

    def _generate_recs_mapper(
        self,
        model: ItemItemRecommender,
        user_mapping: Dict[int, int],
        user_inv_mapping: Dict[int, int],
        N: int,
    ):
        def _recs_mapper(user):
            user_id = user_mapping[user]
            recs = model.similar_items(user_id, N=N)
            return [user_inv_mapping[user] for user, _ in recs], [
                sim for _, sim in recs
            ]

        return _recs_mapper

    def predict(self, train: pd.DataFrame, test: pd.DataFrame, N_recs: int = 10):

        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        mapper = self._generate_recs_mapper(
            model=self.user_knn,
            user_mapping=self.users_mapping,
            user_inv_mapping=self.users_inv_mapping,
            N=self.N_users,
        )

        recs = pd.DataFrame({"user_id": test["user_id"].unique()})
        recs["sim_user_id"], recs["sim"] = zip(*recs["user_id"].map(mapper))
        recs = recs.set_index("user_id").apply(pd.Series.explode).reset_index()

        recs = (
            recs[~(recs["sim"] >= 1)]
            .merge(
                self.watched, left_on=["sim_user_id"], right_on=["user_id"], how="left"
            )
            .explode("item_id")
            .sort_values(["user_id", "sim"], ascending=False)
            .drop_duplicates(["user_id", "item_id"], keep="first")
            .merge(self.item_idf, left_on="item_id", right_on="index", how="left")
        )

        recs["score"] = recs["sim"] * recs["idf"]
        recs = recs.sort_values(["user_id", "score"], ascending=False)
        recs["rank"] = recs.groupby("user_id").cumcount() + 1
        recs = recs[recs["rank"] <= 10]

        final_reco = recs.groupby("user_id").agg({"item_id": list})
        # очень долгое выполнение около 5 часов для полного датасета
        final_reco["item_id"] = final_reco["item_id"].apply(
            lambda x: x + mix_popular_items(train, N_recs - len(x))
        )
        final_reco.to_pickle("final_reco.pickle")

        # преобразование таблицы для метрик
        my_reco = final_reco.explode("item_id")
        my_reco["rank"] = my_reco.groupby("user_id").cumcount() + 1
        my_reco = my_reco.reset_index()

        return my_reco
