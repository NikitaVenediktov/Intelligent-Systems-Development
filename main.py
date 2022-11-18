'''
Starting service of recomendation via FastAPI
'''

import os
from typing import List

import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from models.models import *

API_KEY = "reco_mts_best"
MODEL_LIST = ["pop14d"]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def api_key_auth(api_key: str = Depends(oauth2_scheme)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Forbidden"
        )


app = FastAPI()

df_iter = pd.read_csv("./data_original/interactions.csv", parse_dates=["last_watch_dt"])

recolist_pop14d = pop_14d(df_iter)


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


@app.get(
    "/health", 
    dependencies=[Depends(api_key_auth)]
)
async def root():
    return "Im still alive"


@app.get(
    "/reco/{model_name}/{user_id}",
    response_model=RecoResponse,
    dependencies=[Depends(api_key_auth)],
)
async def get_reco(model_name: str, user_id: int) -> RecoResponse:
    if model_name not in MODEL_LIST:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="This model is not exist"
        )
    else:
        if model_name == "pop14d":
            reco_list = recolist_pop14d

    reco = RecoResponse(user_id=user_id, items=reco_list)

    return reco


if __name__ == "__main__":

    host = os.getenv("HOST", "192.168.89.10")
    port = int(os.getenv("PORT", "8080"))

    uvicorn.run(app, host=host, port=port)
