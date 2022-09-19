from email.mime import base
import os
import sys
sys.path.append('../')
sys.path.append('./')
print(sys.path)
import settings
import pandas as pd

from datetime import date, datetime, timedelta
from binance.client import Client
import alpaca_trade_api as api

class ExtractData:

    def __init__(self):
        self._alpaca = api.REST(settings.CREDENTIALS['ALPACA']['API_KEY'], settings.CREDENTIALS['ALPACA']['SECRET_KEY'])
        self._binance = Client()

    def _validate_interval(self, interval: str) -> None:
        if interval not in settings.ACCEPTED_INTERVALS.keys():
            raise Exception(
                f'Interval value {interval} is not accepted by APIs. Please, use one of the next formats: {settings.keys()}'
                )

    def _validate_tickers(self, tickers: list):
        not_contemplated_tickers = []
        for ticker in tickers:
            if ticker not in settings.TICKERS['CRYPTO'] and ticker not in settings.TICKERS['STOCKS']:
                not_contemplated_tickers.append(ticker)

        if len(not_contemplated_tickers) > 0:
            raise Exception(
                f'Ticker/s {not_contemplated_tickers} not contemplated. Please, add them to settings.py file in TICKERS dictionary.'
            )

    def _validate_mode(self, mode: str):
        if mode not in settings.MODE:
            raise Exception(
                f'Mode {mode} not contemplated. Please, add it to settings.py file in MODES list.'
            )

    def _split_ticker_types(self, tickers: list):
        crypto_tickers = []
        stock_new_tickers = []

        for ticker in tickers:
            if ticker in settings.TICKERS['CRYPTO']:
                crypto_tickers.append(ticker)
            
            elif ticker in settings.TICKERS['STOCKS']:
                stock_new_tickers.append(ticker)

        return stock_new_tickers, crypto_tickers

    def extract_crypto_data(self, tickers: list, period: str, from_date, to_date):
        i = 0
        for ticker in tickers:
            filename = f'{ticker}_{period}.csv'
            i+=1
            print(f'{i}/{len(tickers)} cryptos extracted.')
            if filename not in os.listdir('./data/crypto'):
                # TODO: Follow here: Differ incremental with full extraction
                df = pd.DataFrame(self._binance.get_historical_klines(ticker, period, from_date, to_date))
                df = df[[0, 4]]
                df.columns = ['timestamp', 'close']
                df = df.set_index('timestamp')
                df.index = pd.to_datetime(df.index, unit='ms')

                df.to_csv(f'./data/crypto/{filename}')
        

    def extract_stock_data(self, tickers: list, period: str, from_date, to_date):
        to_date = (datetime.strptime(to_date, "%Y-%m-%d") - timedelta(minutes=15)).strftime('%Y-%m-%d')
        i = 0
        for ticker in tickers:
            filename = f'{ticker}_{period}.csv'
            i+=1
            print(f'{i}/{len(tickers)} stocks extracted.')
            if filename not in os.listdir('./data/stocks'):
                df = self._alpaca.get_bars(ticker, settings.ACCEPTED_INTERVALS[period], from_date, to_date).df['close']
                df.to_csv(f'./data/stocks/{filename}')
            # df.query('ticker==@ticker').to_csv(f'./data/stocks/{ticker}_{period}.csv')

    def merge_tickers_data(self) -> pd.DataFrame:
        base_path = './data/'
        active_types = os.listdir(base_path)
        data = None
        for active_type in active_types:
            if '.csv' in active_type:
                continue

            type_path = base_path + active_type
            for ticker in os.listdir(base_path + active_type):
                ticker_df = pd.read_csv(f'{type_path}/{ticker}')
                ticker_df['ticker'] = ticker.split('_')[0]
                ticker_df.set_index('timestamp', inplace=True)
                ticker_df.index = pd.to_datetime(ticker_df.index).tz_localize(None)
                ticker_df = ticker_df.pivot(columns='ticker', values='close')
                if data is None:
                    data = ticker_df
                else:
                    data = data.merge(ticker_df, left_index=True, right_index=True, how='left')
        data.to_csv('./data/all_tickers.csv')
        return data

    def extract_data(self, tickers: list, interval: str='1D', from_date: date='2017-01-01', to_date: date=datetime.now().strftime("%Y-%m-%d"), mode: str='save'):
        self._validate_interval(interval)
        self._validate_tickers(tickers)
        self._validate_mode(mode)


        stock_tickers, crypto_tickers = self._split_ticker_types(tickers)
        self.extract_crypto_data(crypto_tickers, interval, from_date, to_date)
        self.extract_stock_data(stock_tickers, interval, from_date, to_date)
        self.df = self.merge_tickers_data()
        """
            from_date: None,
            mode: save, retrieve
        
        """
        

if __name__ == '__main__':
    ed = ExtractData()
    ed.extract_data(["TSLA", "AAPL", "SPOT", "NIO", "BABA", "SPY", "BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"], '5m')