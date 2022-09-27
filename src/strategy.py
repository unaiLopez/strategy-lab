import vectorbt as vbt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import config
from create_folds import create_folds
from typing import Tuple

def calculate_indicators_single_ticker(close_interval: pd.DataFrame, lowess_fraction: int) -> Tuple[np.array, np.array, np.array]:
    lowess = sm.nonparametric.lowess

    #do data denoising
    y = close_interval.to_numpy().flatten()
    x = np.array(range(len(y)))
    denoised_ys = lowess(y, x, 1/lowess_fraction, is_sorted=True)[:,1] # the smaller the third parameter, the smoothest the fit

    #calculate first and second derivatives
    first_derivative = np.gradient(denoised_ys)
    second_derivative = np.gradient(first_derivative)
    first_derivative = (((first_derivative - np.min(first_derivative)) / (np.max(first_derivative) - np.min(first_derivative))) * 2) - 1
    second_derivative = (((second_derivative - np.min(second_derivative)) / (np.max(second_derivative) - np.min(second_derivative))) * 2) - 1

    return first_derivative, second_derivative

def calculate_indicators_multiple_tickers(close_interval: pd.DataFrame, lowess_fraction: int) -> Tuple[np.array, np.array, np.array]:
    tickers = list(close_interval.columns)
    
    #do data denoising per asset
    lowess = sm.nonparametric.lowess
    denoised_list = list()
    for ticker in tickers:
        y = close_interval[ticker].to_numpy().flatten()
        x = np.array(range(len(y)))
        denoised_y = lowess(y, x, 1/lowess_fraction, is_sorted=True)[:,1] # the smaller the third parameter, the smoothest the fit
        denoised_list.append(denoised_y)
    denoised_ys = np.stack(denoised_list, axis=1)

    #calculate first and second derivatives
    first_derivative = np.gradient(denoised_ys, axis=0)
    second_derivative = np.gradient(first_derivative, axis=0)
    first_derivative = ((np.divide((first_derivative - np.min(first_derivative, axis=0, keepdims=True)),
                                    np.max(first_derivative, axis=0, keepdims=True) - np.min(first_derivative, axis=0, keepdims=True))) * 2) - 1
    second_derivative = ((np.divide((second_derivative - np.min(second_derivative, axis=0, keepdims=True)),
                                        np.max(second_derivative, axis=0, keepdims=True) - np.min(second_derivative, axis=0, keepdims=True))) * 2) - 1


    return first_derivative, second_derivative

def custom_indicator(close_interval: pd.DataFrame, lowess_fraction: int, velocity_up: float, velocity_down: float,
                     acceleration_up: float, acceleration_down: float) -> np.array:

    tickers = list(close_interval.columns)
    if len(tickers) == 1:
        first_derivative, second_derivative = calculate_indicators_single_ticker(close_interval, lowess_fraction)
    elif len(tickers) > 1:
        first_derivative, second_derivative = calculate_indicators_multiple_tickers(close_interval, lowess_fraction)
    
    #-----signal conditions----
    buy_condition = ((
        (first_derivative > velocity_down) & (first_derivative < velocity_up) & (second_derivative >= acceleration_up)
    ))
    sell_condition = ((
        (first_derivative > velocity_down) & (first_derivative < velocity_up) & (second_derivative <= acceleration_down)
    ))    

    signals = np.where(buy_condition, 1, 0)
    signals = np.where(sell_condition, -1, signals)
    #-------------------------

    return signals

def vectorize_strategy(close: pd.DataFrame, lowess_fraction: int, velocity_up: float, velocity_down: float,
             acceleration_up: float, acceleration_down: float, take_profit: float, stop_loss: float):

    indicator = vbt.IndicatorFactory(
        class_name='first_and_second_order_derivatives',
        short_name='derivatives',
        input_names=['close'],
        param_names=['lowess_fraction', 'velocity_up', 'velocity_down', 'acceleration_up', 'acceleration_down'],
        output_names=['signals']
    ).from_apply_func(
        custom_indicator,
        lowess_fraction=lowess_fraction,
        velocity_up=velocity_up,
        velocity_down=velocity_down,
        acceleration_up=acceleration_up,
        acceleration_down=acceleration_down,
        keep_pd=True
    )

    res = indicator.run(
        close
    )

    entries = res.signals == 1.0
    exits = res.signals == -1.0

    pf = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        fees=config.OPTIMIZATION['FEE_RATE'],
        tp_stop=take_profit,
        sl_stop=stop_loss
    )

    return pf

def apply_strategy(close_interval: pd.DataFrame, interval: str, lowess_fraction: int, velocity_up: float, velocity_down: float,
                   acceleration_up: float, acceleration_down: float, take_profit: float, stop_loss: float, use_folds: float) -> float:

    portfolios = list()
    if use_folds:
        folds = create_folds(close_interval, interval)
        for fold_indexes in folds:
            close_fold = close_interval.iloc[fold_indexes]
            portfolio = vectorize_strategy(close_fold, lowess_fraction, velocity_up, velocity_down, acceleration_up, acceleration_down, take_profit, stop_loss)
            portfolios.append(portfolio)
    else:
        portfolio = vectorize_strategy(close_interval, lowess_fraction, velocity_up, velocity_down, acceleration_up, acceleration_down, take_profit, stop_loss)
        portfolios.append(portfolio)

    return portfolios
