from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import random

app = FastAPI()

# Define request schema
class InputData(BaseModel):
    features: list[float]

model_A = joblib.load("model_v1.pkl")
model_B = joblib.load("model_v3.pkl")

TRAFFIC_SPLIT = 0.7

@app.post("/predict")
def predict(data: InputData):

    features = np.array(data.features).reshape(1, -1)

    if random.random() < TRAFFIC_SPLIT:
        model = model_A
        model_version = "A"
    else:
        model = model_B
        model_version = "B"

    prediction = model.predict(features)[0]

    return {
        "model_used": model_version,
        "prediction": int(prediction)
    }