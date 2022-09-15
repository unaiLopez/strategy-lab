import vectorbt as vbt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime
import logging

from typing import List, Callable
from genopt.environment import Environment

def custom_indicator(close: pd.DataFrame, interval: str, velocity_up: float, velocity_down: float,
                     acceleration_up: float, acceleration_down: float) -> np.array:
    close_interval = close.resample(interval).last()
    # loess denoising---------
    lowess = sm.nonparametric.lowess
    y = close_interval.to_numpy().flatten()
    x = np.array(range(len(y)))
    denoised_ys = lowess(y, x, 1/20, is_sorted=True)[:,1] # the smaller the third parameter, the smoothest the fit
    """
    if len(tickers) == 1:
        y = close_interval.to_numpy().flatten()
        x = np.array(range(len(y)))
        denoised_ys = lowess(y, x, 1/20, is_sorted=True)[:,1] # the smaller the third parameter, the smoothest the fit
    elif len(tickers) > 1:
        denoised_list = list()
        for ticker in tickers:
            y = close_interval[ticker].to_numpy().flatten()
            x = np.array(range(len(y)))
            denoised_y = lowess(y, x, 1/20, is_sorted=True)[:,1] # the smaller the third parameter, the smoothest the fit
            denoised_list.append(denoised_y)
        denoised_ys = np.stack(denoised_list, axis=1)
        print(denoised_ys)
    """
    #-------------------------
    
    # derivation calculation----------
    first_derivative = np.gradient(denoised_ys)
    second_derivative = np.gradient(first_derivative)
    
    first_derivative = first_derivative / np.max(first_derivative)
    second_derivative = second_derivative / np.max(second_derivative)
    #----------------------------------
    
    #df generation and aligning for resample----------
    df_first_derivative = pd.DataFrame(first_derivative, columns=close.columns, index=close_interval.index)
    df_second_derivative = pd.DataFrame(second_derivative, columns=close.columns, index=close_interval.index)
    
    df_first_derivative, _ = df_first_derivative.align(
        close,
        broadcast_axis=0,
        method='ffill',
        join='right'
    )
    df_second_derivative, _ = df_second_derivative.align(
        close,
        broadcast_axis=0,
        method='ffill',
        join='right'
    )
    #-----------------------

    #-----signal conditions----
    first_derivative = df_first_derivative.to_numpy()
    second_derivative = df_second_derivative.to_numpy()
    buy_condition = (
                    (
                        ((first_derivative > velocity_down) & (first_derivative < velocity_up)) &
                        (second_derivative >= acceleration_up)
                    )
                )
    sell_condition = (
        (
            ((first_derivative > velocity_down) & (first_derivative < velocity_up)) &
            (second_derivative <= acceleration_down)
        )
    )

    signals = np.where(buy_condition, 1, 0)
    signals = np.where(sell_condition, -1, signals)
    #-------------------------

    return signals

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

def apply_strategy(ticker_price: pd.DataFrame, interval: str, velocity_up: float, velocity_down: float,
                   acceleration_up: float, acceleration_down: float, take_profit: float, stop_loss: float, fee_rate=0.001) -> float:
    indicator = vbt.IndicatorFactory(
        class_name='first_and_second_order_derivatives',
        short_name='derivatives',
        input_names=['close'],
        param_names=['interval', 'velocity_up', 'velocity_down', 'acceleration_up', 'acceleration_down'],
        output_names=['signals']
    ).from_apply_func(
        custom_indicator,
        interval=interval,
        velocity_up=velocity_up,
        velocity_down=velocity_down,
        acceleration_up=acceleration_up,
        acceleration_down=acceleration_down,
        keep_pd=True
    )

    res = indicator.run(
        ticker_price
    )

    entries = res.signals == 1.0
    exits = res.signals == -1.0

    pf = vbt.Portfolio.from_signals(
        ticker_price,
        entries,
        exits,
        sl_stop=stop_loss,
        tp_stop=take_profit,
        fees=fee_rate
    )

    return pf.total_return()

def define_search_space() -> dict:
    params = {
        'interval': ['1h', '2h', '4h', '6h', '12h'],
        'take_profit': np.arange(0.01, 0.12, step=0.05, dtype=float),
        'stop_loss': np.arange(0.05, 0.1, step=0.05, dtype=float),
        'velocity_up': np.arange(0.005, 1.0, step=0.005, dtype=float),
        'velocity_down': np.arange(-0.005, -1.0, step=-0.005, dtype=float),
        'acceleration_up': np.arange(0.005, 1.0, step=0.005, dtype=float),
        'acceleration_down': np.arange(-0.005, -1.0, step=-0.005, dtype=float)
    }

    return params

def objective(individual: dict) -> float:
    interval = individual['interval']
    take_profit = individual['take_profit']
    stop_loss = individual['stop_loss']
    velocity_up = individual['velocity_up']
    velocity_down = individual['velocity_down']
    acceleration_up = individual['acceleration_up']
    acceleration_down = individual['acceleration_down']
    
    returns = apply_strategy(
        ticker_price,
        interval,
        velocity_up,
        velocity_down,
        acceleration_up,
        acceleration_down,
        take_profit,
        stop_loss
    )
    
    return returns

def optimize(params: dict, objective: Callable[[dict], float], timeout: int, n_jobs: int, num_generations: int, num_population: int, direction: str) -> dict:
    environment = Environment(
        params=params,
        num_population=num_population,
        selection_type='ranking',
        selection_rate=0.8,
        crossover_type='two-point',
        mutation_type='single-gene',
        prob_mutation=0.25,
        verbose=1,
        random_state=42
    )

    results = environment.optimize(
        objective=objective,
        direction=direction,
        num_generations=num_generations,
        timeout=timeout,
        n_jobs=n_jobs
    )

    return results


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(level=logging.ERROR)

    ticker_price = get_ticker_prices(['BTC-USD'], 10, '1h')
    print('DATA READ...')
    
    params = define_search_space()
    results = optimize(
        params=params,
        objective=objective,
        timeout=30,
        n_jobs=1,
        num_generations=99999,
        num_population=1000,
        direction='maximize'
    )

    print(f'BEST INDIVIDUAL: {results.best_individual}')
    print(f'BEST RETURNS: {results.best_score}')
    print('BEST PER GENERATION')
    print(results.best_per_generation_dataframe)
    print('LAST GENERATION INDIVIDUALS')
    print(results.last_generation_individuals_dataframe)

    
"""
returns = pf.total_return()
print('ALL RETURNS')
print(returns)

print('MAX RETURNS')
print(returns.max())
print('MAX INDEX')
print(returns.idxmax())


print(pf.stats())
print(pf.total_return())

pf.plot(subplots=[
    'cum_returns',
    'trades',
    'drawdowns'
]).show()

"""
    
