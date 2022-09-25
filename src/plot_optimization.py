from turtle import title
import pandas as pd

from strategy import apply_strategy
from utils import train_test_split, get_best_trial_parameters, prepare_asset_dataframe_format
import config

if __name__ == '__main__':
    interval, lowess_fraction, velocity_up, velocity_down, acceleration_up, acceleration_down, rsi_window, lower_rsi, upper_rsi = get_best_trial_parameters()
    
    tickers = ['SOLUSDT', 'BTCUSDT', 'ETHUSDT']
    #tickers = ['BTCUSDT']
    ticker_price = prepare_asset_dataframe_format(tickers=tickers)
    close_interval = ticker_price.resample(interval).last()
    close_interval.dropna(axis=0, inplace=True)
    price_train, _ = train_test_split(data=close_interval, test_months=config.OPTIMIZATION['TEST_MONTHS'])

    for ticker in tickers:
        ticker_price_train = price_train[ticker]
        portfolios = apply_strategy(ticker_price_train, interval, lowess_fraction, velocity_up, velocity_down, acceleration_up, acceleration_down, rsi_window, lower_rsi, upper_rsi, use_folds=False)
        portfolio = portfolios[0]

        returns = portfolio.total_return()
        print(f'TICKER {ticker}')
        print('ALL RETURNS')
        print(returns)
        print(portfolio.stats())
        portfolio.plot(subplots=[
            'cum_returns',
            'trades',
            'drawdowns'
        ], title=f'{ticker}').show()
        
