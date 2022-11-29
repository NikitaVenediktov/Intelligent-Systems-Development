"""
Starting service of recomendation via FastAPI
"""

import os
from typing import List

import pandas as pd
import numpy as np
import uvicorn
import dill
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security.api_key import APIKey, APIKeyHeader, APIKeyQuery
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.testclient import TestClient
from pydantic import BaseModel
from rectools import Columns

from models.models import popoular_number_of_items_days, full_reco_items_list

API_KEY = "reco_mts_best"
MODEL_LIST = ["pop14d", "user_knn_v1"]


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


# Load table of interactions and lists of popular items for last 14 days
df_inter = pd.read_csv(
    "./data_original/interactions.csv", parse_dates=["last_watch_dt"]
)
df_inter = df_inter.rename(
    columns={"last_watch_dt": Columns.Datetime, "total_dur": Columns.Weight}
)
recolist_pop14d = popoular_number_of_items_days(df_inter)
recolist_for_cold_users = recolist_pop14d

# Load fitted model for online predictions
with open("./models/user_knn/model_user_knn.dill", "rb") as f:
    model_user_knn = dill.load(f)


api_key_query = APIKeyQuery(name="api_key", auto_error=False)
api_key_header = APIKeyHeader(name="api_key", auto_error=False)
token_bearer = HTTPBearer(auto_error=False)


async def get_api_key(
    api_key_from_query: str = Security(api_key_query),
    api_key_from_header: str = Security(api_key_header),
    token: HTTPAuthorizationCredentials = Security(token_bearer),
) -> str:
    if api_key_from_query == API_KEY:
        return api_key_from_query
    elif api_key_from_header == API_KEY:
        return api_key_from_header
    elif token is not None and token.credentials == API_KEY:
        return token.credentials
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Forbidden"
        )


app = FastAPI()
client = TestClient(app)


@app.get("/health")
async def root():
    return "Im still alive"


@app.get(path="/reco/{model_name}/{user_id}", response_model=RecoResponse)
async def get_reco(
    model_name: str, user_id: int, api_key: APIKey = Depends(get_api_key)
) -> RecoResponse:
    if model_name not in MODEL_LIST:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="This model is not exist"
        )
    else:
        if model_name == "pop14d":
            reco_list = recolist_pop14d
        elif model_name == "user_knn_v1":
            if user_id in df_inter["user_id"].unique():
                print()
                # In case for online predictions
                one_user_df = df_inter[df_inter["user_id"] == user_id][
                    ["user_id", "item_id"]
                ]
                reco_list = model_user_knn.predict(one_user_df, 10)
            else:
                reco_list = recolist_for_cold_users

    reco = RecoResponse(user_id=user_id, items=reco_list)

    return reco


if __name__ == "__main__":

    host = os.getenv("HOST", "192.168.89.10")  # Вариант для сервера Кирилла
    # host = os.getenv("HOST", "127.0.0.1")  # Если у себя тестить
    port = int(os.getenv("PORT", "8080"))

    uvicorn.run(app, host=host, port=port)
