import yfinance as yf

def get_stock_data(ticker):
    data = yf.download(ticker,period = '1y')
    return data

if __name__ == "__main__":
    ticker = input("Enter the stock ticker symbol: ")
    data = get_stock_data(ticker)
    print(type(data))
    