import vectorbt as vbt
import pandas as pd
import datetime
import optuna
import logging

from typing import List
from strategy import apply_strategy

def get_ticker_prices(ticker: List[str], days: int, interval: str) -> pd.DataFrame:
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)

    ticker_price = vbt.YFData.download(
        ticker,
        missing_index='drop',
        start=start,
        end=end,
        interval=interval
    ).get('Close')

    return ticker_price

def objective(trial, close):
    interval = trial.suggest_categorical('interval', ['5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h'])
    lowess_fraction = trial.suggest_int('lowess_fraction', 20, 60, step=5)
    take_profit = trial.suggest_float('take_profit', 0.01, 0.12, step=0.005)
    stop_loss = trial.suggest_float('stop_loss', 0.05, 0.1, step=0.005)
    velocity_up = trial.suggest_float('velocity_up', 0.05, 1.0, step=0.05)
    velocity_down = trial.suggest_float('velocity_down', -1.0, -0.05, step=0.05)
    acceleration_up = trial.suggest_float('acceleration_up', 0.05, 1.0, step=0.05)
    acceleration_down = trial.suggest_float('acceleration_down', -1.0, -0.05, step=0.05)

    close_interval = close.resample(interval).last()
    close_interval.dropna(axis=0, inplace=True)
    
    portfolio = apply_strategy(
        close_interval,
        lowess_fraction,
        velocity_up,
        velocity_down,
        acceleration_up,
        acceleration_down,
        take_profit,
        stop_loss
    )
    
    return portfolio.total_return()

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)

    #ticker_price = get_ticker_prices(ticker=['BTC-USD'], days=120, interval='1h')
    ticker_price = pd.read_csv('../data/validation/BTCEUR_1_MIN_INTERVAL.csv')
    index = pd.DatetimeIndex(ticker_price.Time.values)
    ticker_price = pd.Series(data=ticker_price.BTCEUR.values, index=index)

    func = lambda trial: objective(trial, ticker_price)
    study = optuna.create_study(direction='maximize')
    study.optimize(func, timeout=720, n_jobs=-1)

    print()
    print(f'BEST PARAMS: {study.best_params}')
    print(f'BEST RETURNS: {study.best_value}')
    print('ALL TRIALS')
    df_trials = study.trials_dataframe().sort_values('value', ascending=False)
    print(df_trials)

    df_trials.to_csv('../data/results/optimization_trials.csv', index=False)