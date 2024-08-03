# load_model.py
import tensorflow as tf

def load_model(ticker):
    model_path=f"C:\Python Program\Online_Projects\StockPricePrediction\Models\{ticker}_model.h5"
    model = tf.keras.models.load_model(model_path)
    return model

if __name__ == "__main__":
    ticker = input("Enter the stock ticker symbol: ")
    model = load_model(ticker)
    print(model.summary())
    print(model)
