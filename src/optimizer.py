import pandas as pd
import numpy as np
import optuna
import logging

from strategy import apply_strategy
from utils import train_test_split
import config

def objective(trial: object, close: pd.DataFrame) -> float:
    interval = trial.suggest_categorical('interval', ['1h', '2h', '4h', '6h', '12h'])
    lowess_fraction = trial.suggest_int('lowess_fraction', 20, 60, step=5)
    velocity_up = trial.suggest_float('velocity_up', 0.05, 1.0, step=0.05)
    velocity_down = trial.suggest_float('velocity_down', -1.0, -0.05, step=0.05)
    acceleration_up = trial.suggest_float('acceleration_up', 0.05, 1.0, step=0.05)
    acceleration_down = trial.suggest_float('acceleration_down', -1.0, -0.05, step=0.05)
    rsi_window = trial.suggest_int('rsi_window', 10, 50, step=5)
    lower_rsi = trial.suggest_int('lower_rsi', 15, 40, step=5)
    upper_rsi = trial.suggest_int('upper_rsi', 55, 90, step=5)

    close_interval = close.resample(interval).last()
    close_interval.dropna(axis=0, inplace=True)
    
    portfolios = apply_strategy(
        close_interval, interval, lowess_fraction, velocity_up,
        velocity_down, acceleration_up, acceleration_down,
        rsi_window, lower_rsi, upper_rsi, use_folds=config.OPTIMIZATION['USE_FOLDS_IN_OPTIMIZATION']
    )
    returns = [portfolio.total_return() for portfolio in portfolios]

    return np.mean(returns)

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)

    ticker_price = pd.read_csv(config.OPTIMIZATION['PATH_DATA'])
    index = pd.DatetimeIndex(ticker_price.timestamp.values)
    ticker_price = pd.Series(data=ticker_price.close.values, index=index)
    ticker_price_train, _ = train_test_split(data=ticker_price, test_months=config.OPTIMIZATION['TEST_MONTHS'])

    func = lambda trial: objective(trial, ticker_price_train)
    study = optuna.create_study(direction=config.OPTIMIZATION['DIRECTION'])
    study.optimize(func, timeout=config.OPTIMIZATION['OPTIMIZATION_TIME'], n_jobs=config.OPTIMIZATION['N_JOBS'])

    print()
    print(f'BEST PARAMS: {study.best_params}')
    print(f'BEST RETURNS: {study.best_value}')
    print('ALL TRIALS')
    df_trials = study.trials_dataframe().sort_values('value', ascending=False)
    print(df_trials)

    df_trials.to_csv(config.OPTIMIZATION['PATH_OPTIMIZATION_RESULTS'], index=False)