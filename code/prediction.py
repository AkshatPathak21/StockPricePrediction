# predict.py
from train_model import train_model
import numpy as np

def make_prediction(ticker):
    model, X_test, y_test = train_model(ticker)
    predictions = model.predict(X_test)
    return predictions, y_test

if __name__ == "__main__":
    ticker = "AAPL"  # Example ticker
    predictions, y_test = make_prediction(ticker)
    print("Predictions:", predictions)
    print("Actual:", y_test)
