"""
main.py
--------
FastAPI app serving LSTM-based real estate price predictions.
- Supports multiple regions (trained separately)
- Supports multi-step prediction (e.g., 1, 3, 6 months ahead)
"""



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = FastAPI()
SEQ_LEN = 12

# Load dataset
df = pd.read_csv("real_estate_prices.csv", parse_dates=["date"])
df = df.sort_values(["region", "date"])

# Group by region
grouped = {region: group["price"].values for region, group in df.groupby("region")}

# Load trained models
models = {}
for file in os.listdir("models"):
    if file.startswith("model_") and file.endswith(".h5"):
        region = file.replace("model_", "").replace(".h5", "")
        models[region] = load_model(os.path.join("models", file))

print(f"Loaded models: {list(models.keys())}")

# Request schema
class PredictRequest(BaseModel):
    region: str
    months_ahead: int = 1   # default = 1 month

@app.post("/predict")
async def predict(req: PredictRequest):
    region = req.region
    steps = req.months_ahead

    if region not in models:
        raise HTTPException(status_code=404, detail=f"Region '{region}' not available")

    prices = grouped.get(region)
    if prices is None or len(prices) < SEQ_LEN:
        raise HTTPException(status_code=400, detail="Not enough data for prediction")

    # Normalize
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    # Start with the last window
    window = list(prices_scaled[-SEQ_LEN:])
    preds = []

    # Recursive prediction
    for _ in range(steps):
        X_input = np.array(window[-SEQ_LEN:]).reshape(1, SEQ_LEN, 1)
        pred_scaled = models[region].predict(X_input)[0, 0]
        window.append(pred_scaled)
        preds.append(scaler.inverse_transform([[pred_scaled]])[0, 0])

    return {
        "region": region,
        "months_ahead": steps,
        "predictions": preds
    }
