# train_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import preprocess_data
import fetch_stock_data

def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(ticker, epochs=50, batch_size=32):
    data = fetch_stock_data.get_stock_data(ticker)
    sequences, scaler = preprocess_data.preprocess_data(data)
    X, y = sequences[:-1], sequences[1:, -1]
    X = np.expand_dims(X, axis=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = create_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    
    model.save(f"C:\Python Program\Online_Projects\StockPricePrediction\Models\{ticker}_model.h5")
    return model, X_test,y_test

if __name__ == "__main__":
    ticker = input("Enter the stock ticker symbol: ")
    model, scaler = train_model(ticker)
    print(f"Model trained and saved as {ticker}_model.h5")
