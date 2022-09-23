import pandas as pd

from strategy import apply_strategy
from utils import train_test_split, get_best_trial_parameters
import config

if __name__ == '__main__':
    interval, lowess_fraction, velocity_up, velocity_down, acceleration_up, acceleration_down, rsi_window, lower_rsi, upper_rsi = get_best_trial_parameters()

    tickers = ['BTCUSDT']
    ticker_price = pd.read_csv(config.OPTIMIZATION['PATH_DATA'])
    timestamp = pd.DatetimeIndex(ticker_price.timestamp.values)
    ticker_price.index = timestamp
    ticker_price = ticker_price[tickers]
    ticker_price[ticker_price.index >= '2020-11-20']
    close_interval = ticker_price.resample(interval).last()
    close_interval.dropna(axis=0, inplace=True)
    ticker_price_train, _ = train_test_split(data=close_interval, test_months=config.OPTIMIZATION['TEST_MONTHS'])
    portfolios = apply_strategy(ticker_price_train, interval, lowess_fraction, velocity_up, velocity_down, acceleration_up, acceleration_down, rsi_window, lower_rsi, upper_rsi, use_folds=False)
    portfolio = portfolios[0]
    
    returns = portfolio.total_return()
    print('ALL RETURNS')
    print(returns)
    print(portfolio.stats())
    portfolio.plot(subplots=[
        'cum_returns',
        'trades',
        'drawdowns'
    ]).show()
        
