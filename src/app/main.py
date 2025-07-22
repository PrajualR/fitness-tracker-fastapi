from http.client import HTTPException

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the model
model = joblib.load('../../src/models/xgb_model.pkl')

# Define input schema
class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(input_data: InputData):
    try:
        features = input_data.features
        if len(features) != model.n_features_in_:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {model.n_features_in_} features, got {len(features)}"
            )
        prediction = model.predict([features])
        return {"prediction": int(prediction[0])}
    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))