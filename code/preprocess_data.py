# preprocess_data.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data, feature='Close', seq_len=60):
    # Select the feature for prediction
    data = data[[feature]].values
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    sequences = []
    for i in range(seq_len, len(scaled_data)):
        sequences.append(scaled_data[i-seq_len:i, 0])
    sequences = np.array(sequences)
    return sequences, scaler

if __name__ == "__main__":
    import fetch_stock_data
    ticker = input("Enter the stock ticker symbol: ")
    data = fetch_stock_data.get_stock_data(ticker)
    sequences, scaler = preprocess_data(data)
    print(sequences.shape)
