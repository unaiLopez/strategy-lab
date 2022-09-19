import pandas as pd
import optuna
import logging

from strategy import apply_strategy
from utils import train_test_split
import config

def objective(trial, close):
    interval = trial.suggest_categorical('interval', ['1h', '2h', '4h', '6h', '12h'])
    lowess_fraction = trial.suggest_int('lowess_fraction', 20, 60, step=5)
    velocity_up = trial.suggest_float('velocity_up', 0.05, 1.0, step=0.05)
    velocity_down = trial.suggest_float('velocity_down', -1.0, -0.05, step=0.05)
    acceleration_up = trial.suggest_float('acceleration_up', 0.05, 1.0, step=0.05)
    acceleration_down = trial.suggest_float('acceleration_down', -1.0, -0.05, step=0.05)

    close_interval = close.resample(interval).last()
    close_interval.dropna(axis=0, inplace=True)
    
    mean_returns = apply_strategy(close_interval, interval, lowess_fraction, velocity_up, velocity_down, acceleration_up, acceleration_down)
    
    return mean_returns

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)

    ticker_price = pd.read_csv(config.PATH_DATA)
    index = pd.DatetimeIndex(ticker_price.timestamp.values)
    ticker_price = pd.Series(data=ticker_price.close.values, index=index)
    ticker_price_train, _ = train_test_split(data=ticker_price, test_months=config.TEST_MONTHS)

    func = lambda trial: objective(trial, ticker_price_train)
    study = optuna.create_study(direction='maximize')
    study.optimize(func, timeout=100, n_jobs=1)

    print()
    print(f'BEST PARAMS: {study.best_params}')
    print(f'BEST RETURNS: {study.best_value}')
    print('ALL TRIALS')
    df_trials = study.trials_dataframe().sort_values('value', ascending=False)
    print(df_trials)

    df_trials.to_csv(config.PATH_OPTIMIZATION_RESULTS, index=False)