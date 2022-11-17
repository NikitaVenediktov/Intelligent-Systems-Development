import os
import pandas as pd
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from models.models import *

app = FastAPI()

df_iter = pd.read_csv('./data_original/interactions.csv', parse_dates=['last_watch_dt'])

recolist_pop14d = pop_14d(df_iter)

class RecoResponse(BaseModel):
    user_id: int
    items: List[int]

@app.get("/health")
async def root():
    return "Живее всех живых"


@app.get("/reco/{model_name}/{user_id}", response_model=RecoResponse)
async def get_reco(model_name: str, user_id: int) -> RecoResponse:
    if model_name == 'pop14d':
        reco_list = recolist_pop14d
    else:
        reco_list = list(range(10))[::-1]
    
    reco = RecoResponse(
        user_id = user_id,
        items = reco_list
    )

    return reco


if __name__ == "__main__":

    host = os.getenv("HOST", "192.168.89.10")
    port = int(os.getenv("PORT", "8080"))

    uvicorn.run(app, host=host, port=port)