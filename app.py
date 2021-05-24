from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import ccb_model

app = FastAPI()
app.mount("/demo", StaticFiles(directory="static", html=True), name="static")

class RecommendationRequest(BaseModel):
    userFeatures: dict
    actionFeatures: list
    slotCount: int

class RewardRequest(BaseModel):
    predictionId: str
    slotIndex: int
    reward: float

class LearnRequest(BaseModel):
    predictionId: str


@app.post("/recommend")
async def recommend(body: RecommendationRequest):
    print(type(body))
    print(body)
    prediction_id, recommendations_list = ccb_model.ccb_predict(dict(body))
    return {
        "predictionId": prediction_id,
        "rank": recommendations_list
    }


@app.post("/reward")
async def reward(body: RewardRequest):
    ccb_model.ccb_save_reward(
        body.predictionId,
        body.slotIndex,
        body.reward
    )


@app.post("/learn")
async def learn(body: LearnRequest):
    ccb_model.ccb_learn(body.predictionId)


@app.get("/save")
async def save():
    ccb_model.save_bandit_data_to_disk()


@app.get("/get_options")
async def get_options():
    return ccb_model.get_options()
