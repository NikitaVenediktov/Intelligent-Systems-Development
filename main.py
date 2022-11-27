"""
Starting service of recomendation via FastAPI
"""

import os
from typing import List

import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status, Security
from fastapi.security.api_key import APIKey, APIKeyHeader, APIKeyQuery
from fastapi.testclient import TestClient
from pydantic import BaseModel

from models.models import pop_items_last_14d

API_KEY = "reco_mts_best"
MODEL_LIST = ["pop14d"]


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


df_inter = pd.read_csv("./data_original/interactions.csv",
                       parse_dates=["last_watch_dt"])
recolist_pop14d = pop_items_last_14d(df_inter)

api_key_query = APIKeyQuery(name="api_key", auto_error=False)
api_key_header = APIKeyHeader(name="api_key", auto_error=False)


async def get_api_key(
    api_key_from_query: str = Security(api_key_query),
    api_key_from_header: str = Security(api_key_header),
) -> str:
    if api_key_from_query == API_KEY:
        return api_key_from_query
    elif api_key_from_header == API_KEY:
        return api_key_from_header
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Forbidden"
        )


app = FastAPI()
client = TestClient(app)


@app.get("/health")
async def root():
    return "Im still alive"


@app.get(
    path="/reco/{model_name}/{user_id}",
    response_model=RecoResponse,
)
async def get_reco(
    model_name: str,
    user_id: int,
    api_key: APIKey = Depends(get_api_key)
) -> RecoResponse:
    if model_name not in MODEL_LIST:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This model is not exist"
        )
    else:
        if model_name == "pop14d":
            reco_list = recolist_pop14d

    reco = RecoResponse(user_id=user_id, items=reco_list)

    return reco


if __name__ == "__main__":

    # host = os.getenv("HOST", "192.168.89.10")  # Вариант для сервера Кирилла
    host = os.getenv("HOST", "127.0.0.1")  # Если у себя тестить
    port = int(os.getenv("PORT", "8080"))

    uvicorn.run(app, host=host, port=port)
