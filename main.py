from ingestion import load_data

tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'FB', 'NVDA', 'BABA', 'AMD', 'PYPL',
    'NFLX', 'NVDA', 'TWTR', 'SNAP', 'V', 'MA', 'DIS', 'INTC', 'ASML', 'T', 'PFE',
    'MRK', 'JNJ', 'GSK', 'ABT', 'VZ', 'KO', 'WMT', 'NKE', 'MCD', 'UNH', 'HD',
    'ORCL', 'CRM', 'SPY', 'QQQ', 'AMET', 'CAT', 'XOM', 'CVX', 'ADBE', 'PYPL',
    'F', 'BA', 'PEP', 'GS', 'JPM', 'BTC-USD', 'ETH-USD', 'DOGE-USD'
]

TICKER_DATA_PATH = 'data'

for ticker in tickers:

    # Download data if not exists
    load_data(ticker, TICKER_DATA_PATH)




def calculate_bollinger_bands(df, window=20):
    df['SMA'] = df['Close'].rolling(window=window).mean()
    df['Upper Band'] = df['SMA'] + (df['Close'].rolling(window=window).std() * 2)
    df['Lower Band'] = df['SMA'] - (df['Close'].rolling(window=window).std() * 2)
    return df
