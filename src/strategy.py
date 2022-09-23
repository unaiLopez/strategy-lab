import vectorbt as vbt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import config

from create_folds import create_folds

def custom_indicator(close_interval: pd.DataFrame, lowess_fraction: int, velocity_up: float, velocity_down: float,
                     acceleration_up: float, acceleration_down: float, rsi_window: int, lower_rsi: int, upper_rsi: int) -> np.array:
    # loess denoising---------
    lowess = sm.nonparametric.lowess
    y = close_interval.to_numpy().flatten()
    x = np.array(range(len(y)))
    denoised_ys = lowess(y, x, 1/lowess_fraction, is_sorted=True)[:,1] # the smaller the third parameter, the smoothest the fit
    """
    if len(tickers) == 1:
        y = close_fold.to_numpy().flatten()
        x = np.array(range(len(y)))
        denoised_ys = lowess(y, x, 1/lowess_fraction, is_sorted=True)[:,1] # the smaller the third parameter, the smoothest the fit
    elif len(tickers) > 1:
        denoised_list = list()merge master into branch
        for ticker in tickers:
            y = close_fold[ticker].to_numpy().flatten()
            x = np.array(range(len(y)))
            denoised_y = lowess(y, x, 1/lowess_fraction, is_sorted=True)[:,1] # the smaller the third parameter, the smoothest the fit
            denoised_list.append(denoised_y)
        denoised_ys = np.stack(denoised_list, axis=1)
        print(denoised_ys)
    """
    #-------------------------
    
    # derivation calculation----------
    first_derivative = np.gradient(denoised_ys)
    second_derivative = np.gradient(first_derivative)

    first_derivative = (((first_derivative - np.min(first_derivative)) / (np.max(first_derivative) - np.min(first_derivative))) * 2) - 1
    second_derivative = (((second_derivative - np.min(second_derivative)) / (np.max(second_derivative) - np.min(second_derivative))) * 2) - 1
    #----------------------------------

    rsi = vbt.RSI.run(close_interval, rsi_window).rsi.to_numpy().flatten()

    #-----signal conditions----
    buy_condition = ((
        (first_derivative > velocity_down) & (first_derivative < velocity_up) & (second_derivative >= acceleration_up) & (rsi < lower_rsi)
    ))
    sell_condition = ((
        (first_derivative > velocity_down) & (first_derivative < velocity_up) & (second_derivative <= acceleration_down) & (rsi > upper_rsi)
    ))    

    signals = np.where(buy_condition, 1, 0)
    signals = np.where(sell_condition, -1, signals)
    #-------------------------

    return signals

def vectorize_strategy(close: pd.DataFrame, lowess_fraction: int, velocity_up: float, velocity_down: float,
             acceleration_up: float, acceleration_down: float, rsi_window: int, lower_rsi: int, upper_rsi: int):

    indicator = vbt.IndicatorFactory(
        class_name='first_and_second_order_derivatives',
        short_name='derivatives',
        input_names=['close'],
        param_names=['lowess_fraction', 'velocity_up', 'velocity_down', 'acceleration_up', 'acceleration_down', 'rsi_window', 'lower_rsi', 'upper_rsi'],
        output_names=['signals']
    ).from_apply_func(
        custom_indicator,
        lowess_fraction=lowess_fraction,
        velocity_up=velocity_up,
        velocity_down=velocity_down,
        acceleration_up=acceleration_up,
        acceleration_down=acceleration_down,
        rsi_window=rsi_window,
        lower_rsi=lower_rsi,
        upper_rsi=upper_rsi,
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
        fees=config.FEE_RATE
    )

    return pf

def apply_strategy(close_interval: pd.DataFrame, interval: str, lowess_fraction: int, velocity_up: float, velocity_down: float,
                   acceleration_up: float, acceleration_down: float, rsi_window: int, lower_rsi: int, upper_rsi: int, use_folds: float) -> float:

    portfolios = list()
    if use_folds:
        folds = create_folds(close_interval, interval)
        for fold_indexes in folds:
            close_fold = close_interval.iloc[fold_indexes]
            portfolio = vectorize_strategy(close_fold, lowess_fraction, velocity_up, velocity_down, acceleration_up, acceleration_down, rsi_window, lower_rsi, upper_rsi)
            portfolios.append(portfolio)
    else:
        portfolio = vectorize_strategy(close_interval, lowess_fraction, velocity_up, velocity_down, acceleration_up, acceleration_down, rsi_window, lower_rsi, upper_rsi)
        portfolios.append(portfolio)

    return portfolios
