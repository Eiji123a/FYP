"""
train_lstm.py
--------------
Train an LSTM model for each region in the dataset.
Each trained model will be saved as model_{region}.h5
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Parameters
SEQ_LEN = 12  # number of time steps (e.g., 12 months)

# Load dataset (CSV format: region,date,price)
df = pd.read_csv("real_estate_prices.csv", parse_dates=["date"])
df = df.dropna(subset=["region", "date", "price"])
df = df.sort_values(["region", "date"])

# Create "models" directory if not exists
os.makedirs("models", exist_ok=True)

# Train a model for each region
for region, group in df.groupby("region"):
    print(f"=== Training model for region: {region} ===")
    
    prices = group["price"].values.astype(float).reshape(-1, 1)
    
    # Skip regions with insufficient data
    if len(prices) <= SEQ_LEN:
        print(f"  Skipped: not enough data ({len(prices)} samples)")
        continue
    
    # Normalize prices
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices).flatten()
    
    # Build sequences
    X, y = [], []
    for i in range(len(prices_scaled) - SEQ_LEN):
        X.append(prices_scaled[i:i+SEQ_LEN])
        y.append(prices_scaled[i+SEQ_LEN])
    X = np.array(X).reshape(-1, SEQ_LEN, 1)
    y = np.array(y)
    
    # Define LSTM model
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(SEQ_LEN,1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    
    # Train model
    model.fit(X, y, epochs=20, batch_size=4, verbose=0)
    
    # Save model per region
    model_path = os.path.join("models", f"model_{region}.h5")
    model.save(model_path)
    print(f"  Saved model: {model_path}")
