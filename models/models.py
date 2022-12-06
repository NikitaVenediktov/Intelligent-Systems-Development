"""
Reco Models
"""
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).parent
BASE_DIR = CURRENT_DIR.parents[0]
MODELS_DIR = CURRENT_DIR
DATA_DIR = BASE_DIR / "data_original"


# For HW1********************************************************************
def popular_number_of_items_days(
    df: pd.DataFrame, k: int = 10, days: int = 14, all_time: bool = False
) -> list:
    """
    Return a list of top@k most popular items for last N days
    """
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
    return list(recommendations)


# For HW3********************************************************************
def full_reco_items_list(
    arr_reco_after_model: np.array, pop_array: np.array, number: int
) -> list:
    """
    Add number of items from pop_array to arr_reco_after_model.
    Return array of 10 unique items
    """

    CONST_K = 10

    size_of_array = np.unique(arr_reco_after_model).size
    if size_of_array == CONST_K:
        return arr_reco_after_model

    # all duplicates will be deleting and adding some items from pop_array
    def del_repeat_items(full_arr_mix_pop: np.array, k: int = CONST_K) -> np.array:
        """
        Delete all duplicates items in array
        """

        size_of_array = np.unique(full_arr_mix_pop).size
        if size_of_array == CONST_K:
            return full_arr_mix_pop
        else:
            # delete duplicates and save the order of items
            full_arr_mix_pop = full_arr_mix_pop[
                np.sort(np.unique(full_arr_mix_pop, return_index=True)[1])
            ]
            # add new items from pop_array
            i = k - size_of_array
            full_arr_mix_pop = np.concatenate(
                (full_arr_mix_pop, np.random.choice(pop_array, i, replace=False))
            )
            return del_repeat_items(full_arr_mix_pop)

    full_arr_mix_pop = np.array([])
    full_arr_mix_pop = np.concatenate((arr_reco_after_model, pop_array[:number]))
    full_arr_mix_pop = del_repeat_items(full_arr_mix_pop)

    return list(full_arr_mix_pop)


def knn_make_predict(model, data: pd.DataFrame, list_pop_items: list) -> list:
    """
    Gets predict after bas-model and add lacking items(to 10)
    for each user if it needed
    """

    predict = model.predict(data, 10)
    predict = predict["item_id"].values
    final_reco = full_reco_items_list(
        predict, np.array(list_pop_items), (10 - predict.size)
    )

    return list(final_reco)


# For HW4********************************************************************
