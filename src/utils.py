import datetime
import pandas as pd

from typing import Dict, Tuple, List
import config

def train_test_split(data: pd.DataFrame, test_months: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    last_date = data.index.max()
    cut_date = last_date - datetime.timedelta(days=test_months*30)
    data_train = data[data.index <= cut_date]
    data_test = data[data.index > cut_date]

    return data_train, data_test

def get_best_trial_parameters() -> dict:   # Dict[str, int, float, float, float, float]: --> Bug don't letting type for dict. TODO: Search a workaround
    df_optimization_trials = pd.read_csv(config.OPTIMIZATION['PATH_OPTIMIZATION_RESULTS'])

    best_trial = df_optimization_trials.iloc[0,:]
    parameters_dict = {
        'interval': best_trial.params_interval,
        'lowess_fraction': best_trial.params_lowess_fraction,
        'velocity_up': best_trial.params_velocity_up,
        'velocity_down': best_trial.params_velocity_down,
        'acceleration_up': best_trial.params_acceleration_up,
        'acceleration_down': best_trial.params_acceleration_down,
        'take_profit': best_trial.params_take_profit,
        'stop_loss': best_trial.params_stop_loss
    }

    return parameters_dict

def prepare_asset_dataframe_format(tickers: List[str]) -> pd.DataFrame:
    ticker_price = pd.read_csv(config.OPTIMIZATION['PATH_DATA'])
    timestamp = pd.DatetimeIndex(ticker_price.timestamp.values)
    ticker_price.index = timestamp
    ticker_price = ticker_price[tickers]
    ticker_price[ticker_price.index >= config.OPTIMIZATION['START_DATE']]
    return ticker_price


def check_multiasset(data):
    return not isinstance(data, pd.Series)