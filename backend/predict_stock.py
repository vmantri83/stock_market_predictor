import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import os 
from sklearn.preprocessing import MinMaxScaler

# Dummy CSV loader for last 60 days' prices of all stocks
def load_stock_data(ticker):
    path = f"data/{ticker}.csv"  # Make sure these exist
    df = pd.read_csv(path)
    return df['Close'].values[-60:]  # Last 60 days

# Predict stock prices using pre-trained .h5 model
def predict_stock_price(ticker, days):
    try:
        # Load model
        model_path = os.path.join("models", f"{ticker}.NS_lstm.h5")
        if not os.path.exists(model_path):
            return {"error": f"Model file for {ticker} not found."}

        model = load_model(model_path, compile=False)

       # Get last 60 days of data for prediction (buffered for safety)
        ticker_full = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
        df = yf.download(ticker_full, period="120d")
        print(f"Downloading data for: {ticker_full}")
        print(f"{ticker}: {len(df)} days downloaded")
        if df.empty or 'Close' not in df.columns or len(df['Close'].dropna()) < 30:
            return [f"No valid data found for {ticker}. Check if the ticker is active and correct."]

        close_prices = df['Close'].values[-90:]

        # Normalize prices between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_close = scaler.fit_transform(close_prices.reshape(-1, 1))

        predictions = []
        input_sequence = scaled_close.copy()

        for _ in range(days):
            input_reshaped = input_sequence[-30:].reshape(1, 30, 1)
            scaled_pred = model.predict(input_reshaped, verbose=0)
            predictions.append(scaler.inverse_transform(scaled_pred)[0][0])

            # Append prediction for next round (recursive forecasting)
            input_sequence = np.append(input_sequence, scaled_pred, axis=0)

        return [float(p) for p in predictions]
    
    except Exception as e:
        return [str(e)]