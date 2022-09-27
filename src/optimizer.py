import pandas as pd
import numpy as np
import optuna
import logging

from strategy import apply_strategy
from utils import train_test_split, prepare_asset_dataframe_format
from early_stopping import EarlyStoppingCallback
import config



def objective(trial: object, close: pd.DataFrame) -> float:
    interval = trial.suggest_categorical('interval', ['1h', '2h', '4h', '6h', '12h', '24h'])
    #interval = trial.suggest_categorical('interval', ['15min', '30min', '1h', '2h', '4h', '6h', '12h', '24h'])
    lowess_fraction = trial.suggest_int('lowess_fraction', 20, 60, step=5)
    velocity_up = trial.suggest_float('velocity_up', 0.05, 1.0, step=0.05)
    velocity_down = trial.suggest_float('velocity_down', -1.0, -0.05, step=0.05)
    acceleration_up = trial.suggest_float('acceleration_up', 0.05, 1.0, step=0.05)
    acceleration_down = trial.suggest_float('acceleration_down', -1.0, -0.05, step=0.05)
    take_profit = trial.suggest_float('take_profit', 0.01, 0.05, step=0.01)
    stop_loss = trial.suggest_float('stop_loss', 0.01, 0.05, step=0.01)


    close_interval = close.resample(interval).last()
    close_interval.dropna(axis=0, inplace=True)
    
    portfolios = apply_strategy(
        close_interval, interval, lowess_fraction, velocity_up,
        velocity_down, acceleration_up, acceleration_down, take_profit, stop_loss,
        use_folds=config.OPTIMIZATION['USE_FOLDS_IN_OPTIMIZATION']
    )
    
    if close.shape[1] > 1:
        returns = [np.mean(portfolio.total_return()) for portfolio in portfolios]
    else:
        returns = [portfolio.total_return() for portfolio in portfolios]

    return np.mean(returns)

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)

    ticker_price = prepare_asset_dataframe_format(tickers=config.OPTIMIZATION['TICKERS'])
    ticker_price_train, _ = train_test_split(data=ticker_price, test_months=config.OPTIMIZATION['TEST_MONTHS'])
    func = lambda trial: objective(trial, ticker_price_train)
    study = optuna.create_study(direction=config.OPTIMIZATION['DIRECTION'])
    early_stopping = EarlyStoppingCallback(config.OPTIMIZATION['EARLY_STOPPING_ITERATIONS'], direction=config.OPTIMIZATION['DIRECTION'])
    study.optimize(func, callbacks=[early_stopping], timeout=config.OPTIMIZATION['OPTIMIZATION_TIME'], n_jobs=config.OPTIMIZATION['N_JOBS'])

    print()
    print(f'BEST PARAMS: {study.best_params}')
    print(f'BEST RETURNS: {study.best_value}')
    print('ALL TRIALS')
    df_trials = study.trials_dataframe().sort_values('value', ascending=False)
    print(df_trials)

    df_trials.to_csv(config.OPTIMIZATION['PATH_OPTIMIZATION_RESULTS'], index=False)