import os
import config
import pandas as pd

from datetime import date, datetime, timedelta
from binance.client import Client
import alpaca_trade_api as api
import logging
logger = logging.getLogger()
logger.setLevel(level=logging.INFO)

class ExtractData:

    def __init__(self):
        self._alpaca = api.REST(config.CREDENTIALS['ALPACA']['API_KEY'], config.CREDENTIALS['ALPACA']['SECRET_KEY'])
        self._binance = Client()

    def _validate_interval(self, interval: str) -> None:
        if interval not in config.ACCEPTED_INTERVALS.keys():
            raise Exception(
                f'Interval value {interval} is not accepted by APIs. Please, use one of the next formats: {config.ACCEPTED_INTERVALS.keys()}'
                )

    def _validate_tickers(self, tickers: list):
        not_contemplated_tickers = []
        for ticker in tickers:
            if ticker not in config.TICKERS['CRYPTO'] and ticker not in config.TICKERS['STOCKS']:
                not_contemplated_tickers.append(ticker)

        if len(not_contemplated_tickers) > 0:
            raise Exception(
                f'Ticker/s {not_contemplated_tickers} not contemplated. Please, add them to config.py file in TICKERS dictionary.'
            )

    def _validate_mode(self, mode: str):
        if mode not in config.MODE:
            raise Exception(
                f'Mode {mode} not contemplated. Please, add it to config.py file in MODES list.'
            )
    
    def _create_folders(self):
        for asset_type in config.ASSET_TYPE:
            for interval in config.ACCEPTED_INTERVALS:
                try:
                    os.mkdir(os.path.join(os.getcwd(), f'data/{asset_type}/{interval}'))
                except FileExistsError:
                    pass


    def _split_ticker_types(self, tickers: list):
        crypto_tickers = []
        stock_new_tickers = []

        for ticker in tickers:
            if ticker in config.TICKERS['CRYPTO']:
                crypto_tickers.append(ticker)
            
            elif ticker in config.TICKERS['STOCKS']:
                stock_new_tickers.append(ticker)

        return stock_new_tickers, crypto_tickers

    def extract_crypto_data(self, tickers: list, period: str, from_date, to_date, overwrite=False):
        i = 0
        for ticker in tickers:
            filename = f'{ticker}_{period}.csv'
            i+=1
            if filename not in os.listdir(os.path.join(os.getcwd(), f'data/crypto/{period}')) or overwrite:
                logger.info(f'Extracting {i}/{len(tickers)} crypto data.')
                df = pd.DataFrame(self._binance.get_historical_klines(ticker, period, from_date, to_date))
                df = df[[0, 4]]
                df.columns = ['timestamp', 'close']
                df = df.set_index('timestamp')
                df.index = pd.to_datetime(df.index, unit='ms')
                df.to_csv(os.path.join(os.getcwd(), f'data/crypto/{period}/{filename}'))
        

    def extract_stock_data(self, tickers: list, period: str, from_date, to_date, overwrite=False):
        to_date = (datetime.strptime(to_date, "%Y-%m-%d") - timedelta(minutes=15)).strftime('%Y-%m-%d')  # last 15 minutes are not available in API
        tickers_extracted = 0
        for ticker in tickers:
            tickers_extracted += 1
            filename = f'{ticker}_{period}.csv'
            if filename not in os.listdir(os.path.join(os.getcwd(), f'data/stocks/{period}')) or overwrite:
                logger.info(f'Extracting data of {tickers_extracted}/{len(tickers)} stock data.')
                df = self._alpaca.get_bars(ticker, config.ACCEPTED_INTERVALS[period], from_date, to_date).df['close']
                df.to_csv(os.path.join(os.getcwd(), f'data/stocks/{period}/{filename}'))

    def merge_tickers_data(self, tickers: list, period: str) -> pd.DataFrame:
        
        data = None
        for asset_type in config.ASSET_TYPE:
            base_path = os.path.join(os.getcwd(), f'data/{asset_type}/{period}/')
            for ticker in os.listdir(base_path):
                raw_ticker = ticker.split('_')[0]
                if raw_ticker in tickers:
                    ticker_df = pd.read_csv(f'{base_path}/{ticker}')
                    ticker_df['ticker'] = raw_ticker
                    ticker_df.set_index('timestamp', inplace=True)
                    ticker_df.index = pd.to_datetime(ticker_df.index).tz_localize(None)
                    ticker_df = ticker_df.pivot(columns='ticker', values='close')
                    if data is None:
                        data = ticker_df
                    else:
                        data = data.merge(ticker_df, left_index=True, right_index=True, how='outer')

        data.to_csv(os.path.join(os.getcwd(), 'data/all_tickers.csv'))
        return data

    def extract_data(self, tickers: list, interval: str='1d', from_date: date='2017-01-01', to_date: date=datetime.now().strftime("%Y-%m-%d"), overwrite: bool=False):
        self._validate_interval(interval)
        self._validate_tickers(tickers)
        self._create_folders()
        stock_tickers, crypto_tickers = self._split_ticker_types(tickers)
        self.extract_crypto_data(crypto_tickers, interval, from_date, to_date, overwrite)
        self.extract_stock_data(stock_tickers, interval, from_date, to_date, overwrite)
        self.df = self.merge_tickers_data(tickers, interval)
        return self.df

    

if __name__ == '__main__':
    ed = ExtractData()
    ed.extract_data(['TSLA'], '1d', overwrite=True)
