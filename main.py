"""
Starting service of recomendation via FastAPI
"""

import asyncio
import os
from pathlib import Path
from typing import List

import dill
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKey, APIKeyHeader, APIKeyQuery
from fastapi.testclient import TestClient
from pydantic import BaseModel
from rectools import Columns

from models.models import knn_make_predict, popular_number_of_items_days

load_dotenv()

API_KEY = os.getenv("API-KEY")
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data_original"
MODEL_LIST = ["pop14d", "user_knn_v1"]


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


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


# Load table of interactions and lists of popular items for last 14 days
df_inter = pd.read_csv(DATA_DIR / "interactions.csv", parse_dates=["last_watch_dt"])
df_inter = df_inter.rename(
    columns={"last_watch_dt": Columns.Datetime, "total_dur": Columns.Weight}
)
recolist_pop14d = popular_number_of_items_days(df_inter)
recolist_for_cold_users = recolist_pop14d


class CustomUnpickler(dill.Unpickler):
    """
    Это специальный класс для того, чтобы наследовать
    все зависимости обученной модели.
    Фишка вся в том, что без такой загрузки uvicorn/gunicorn
    не будет работать. Можно запустить python main.py, но так
    работает только с 1 воркером, при большем кол-ве воркеров, а также
    при запуске с командной строки uvicorn/gunicorn теряются зависимости.
    Чтобы найти причину и эти 7 строчек ушла почти неделя.
    Запомнить раз и навсегда!!!
    """

    def find_class(self, module, name):
        if name == "my_UserKnn":
            from models.train_knn import my_UserKnn

            return my_UserKnn
        return super().find_class(module, name)


# Load fitted base model
knn_model = CustomUnpickler(open(MODELS_DIR / "full_model_user_knn.dill", "rb")).load()


@app.get("/health")
async def root():
    return "Im still alive"


@app.get(path="/reco/{model_name}/{user_id}", response_model=RecoResponse)
def get_reco(
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
                # In case for online predictions
                one_user_df = df_inter[df_inter["user_id"] == user_id][
                    ["user_id", "item_id"]
                ]
                reco_list = knn_make_predict(knn_model, one_user_df, recolist_pop14d)
            else:
                reco_list = recolist_for_cold_users

    reco = RecoResponse(user_id=user_id, items=reco_list)

    return reco


async def main():
    # host = os.getenv("HOST", "192.168.89.10")  # Вариант для сервера Кирилла
    host = os.getenv("HOST", "127.0.0.1")  # Если у себя тестить
    port = int(os.getenv("PORT", "8080"))
    config = uvicorn.Config("main:app", host=host, port=port, workers=4)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
