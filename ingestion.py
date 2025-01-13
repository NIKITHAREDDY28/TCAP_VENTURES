import os
import yfinance as yf


def load_data(ticker: str, data_path: str = 'data'):

    # Ticker persistent storage path

    ticker_path = f'{data_path}/{ticker}.csv'

    if not os.path.exists(ticker_path):
        print(f'{ticker_path} does not exist. So Downloading data')

        # Get prev 1 year data
        data = yf.download(ticker, period='1y')

        # Save data to csv
        data.to_csv(ticker_path)

        print(f'{ticker} data downloaded')

