import pandas as pd

from datetime import datetime
from binance.client import Client

def extract_data(ticker: str, start_date: str, interval: str = '5m') -> None:
    client = Client()            
    df = pd.DataFrame(client.get_historical_klines(ticker, interval, start_date, datetime.now().strftime("%Y-%m-%d")))
    df = df[[0, 4]]
    df.columns = ['timestamp', 'close']
    df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index, unit='ms')

    df.to_csv(f'../data/validation/{ticker}_5_MIN_INTERVAL.csv')

if __name__ == '__main__':
    ticker = 'ETHUSDT'
    start_date = '2017-01-01'

    extract_data(ticker, start_date)