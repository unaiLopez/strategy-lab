import pandas as pd

from strategy import apply_strategy
from utils import train_test_split
import config

if __name__ == '__main__':
    df_optimization_trials = pd.read_csv(config.PATH_OPTIMIZATION_RESULTS)
    ticker_price = pd.read_csv(config.PATH_DATA)
    
    index = pd.DatetimeIndex(ticker_price.timestamp.values)
    ticker_price = pd.Series(data=ticker_price.close.values, index=index)

    best_trial = df_optimization_trials.iloc[0,:]
    interval = best_trial.params_interval
    lowess_fraction = best_trial.params_lowess_fraction
    velocity_up = best_trial.params_velocity_up
    velocity_down = best_trial.params_velocity_down
    acceleration_up = best_trial.params_acceleration_up
    acceleration_down = best_trial.params_acceleration_down

    close_interval = ticker_price.resample(interval).last()
    close_interval.dropna(axis=0, inplace=True)

    ticker_price_train, _ = train_test_split(data=close_interval, test_months=config.TEST_MONTHS)

    portfolio = apply_strategy(ticker_price_train, lowess_fraction, velocity_up, velocity_down, acceleration_up,
                   acceleration_down)
    
    returns = portfolio.total_return()
    print('ALL RETURNS')
    print(returns)
    print(portfolio.stats())
    portfolio.plot(subplots=[
        'cum_returns',
        'trades',
        'drawdowns'
    ]).show()
        
