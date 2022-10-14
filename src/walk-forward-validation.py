import pandas as pd
import math
import vectorbt as vbt
import strategy
import config
import optimizer
import json
import optuna
import numpy as np
from extract import ExtractData
from strategy import custom_indicator
from sklearn.model_selection import TimeSeriesSplit
from early_stopping import EarlyStoppingCallback
from tqdm import tqdm

class WalkForwardValidation:

    def __init__(self, ticker_list: list) -> None:
        self.data = ExtractData().extract_data(ticker_list, from_date='2016-01-01')
        self._chunk_division = (0.70, 0.30)  # Left train, right test

    def _create_rolling_folds(self, num_folds: int=12):
        print(self.data.shape)
        mod_data = len(self.data) % num_folds
        if mod_data != 0:
            self.data = self.data.iloc[mod_data:]

        test_size = 30
        
        window_len = int(len(self.data) / num_folds) + test_size

        return self.data.vbt.rolling_split(n=num_folds, window_len=window_len, set_lens=(test_size,), left_to_right=False)


    def _create_accumulated_folds(self, num_folds: int=10):
        splitter = TimeSeriesSplit(n_splits=num_folds)
        # (train_df, train_indexes), (test_df, test_indexes) = self.data.vbt.split(splitter)

        fig = self.data.vbt.split(splitter, plot=True, trace_names=['train', 'test'])
        fig.update_layout(width=1280, height=720)
        fig.show()

    def _get_signals(self, fast_ma, slow_ma):
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        return entries, exits
    """ 
    def _optimize(self):
        windows = np.arange(10, 50)

        (is_prices, is_dates), (oos_prices, oos_dates) = self._create_rolling_folds()
        
        fast_ma, slow_ma = vbt.MA.run_combs(is_prices, windows)
        entries, exits = self._get_signals(fast_ma, slow_ma)

        portfolio = vbt.Portfolio.from_signals(is_prices, entries, exits, freq='1d', direction='both')
        ret = portfolio.total_return()
        print(ret[ret.groupby('split_idx').idxmax()].index)
    """


    def vectorize_strategy(self, close: pd.DataFrame, lowess_fraction: int, velocity_up: float, velocity_down: float,
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

    def apply_strategy(self, close_interval: pd.DataFrame, lowess_fraction: int, velocity_up: float, velocity_down: float,
                   acceleration_up: float, acceleration_down: float, take_profit: float, stop_loss: float) -> float:

        portfolios = list()
        portfolio = self.vectorize_strategy(close_interval, lowess_fraction, velocity_up, velocity_down, acceleration_up, acceleration_down, take_profit, stop_loss)
        portfolios.append(portfolio)

        return portfolios

    def objective(self, trial: object, close: pd.DataFrame) -> float:
        #interval = trial.suggest_categorical('interval', ['1h', '2h', '4h', '6h', '12h', '24h'])
        #interval = trial.suggest_categorical('interval', ['15min', '30min', '1h', '2h', '4h', '6h', '12h', '24h'])
        lowess_fraction = trial.suggest_int('lowess_fraction', 20, 60, step=5)
        velocity_up = trial.suggest_float('velocity_up', 0.05, 1.0, step=0.05)
        velocity_down = trial.suggest_float('velocity_down', -1.0, -0.05, step=0.05)
        acceleration_up = trial.suggest_float('acceleration_up', 0.05, 1.0, step=0.05)
        acceleration_down = trial.suggest_float('acceleration_down', -1.0, -0.05, step=0.05)
        take_profit = trial.suggest_float('take_profit', 0.01, 0.05, step=0.01)
        stop_loss = trial.suggest_float('stop_loss', 0.01, 0.05, step=0.01)


        # close_interval = close.resample(interval).last()
        # close_interval.dropna(axis=0, inplace=True)
        
        portfolios = self.apply_strategy(
            close, lowess_fraction, velocity_up,
            velocity_down, acceleration_up, acceleration_down, take_profit, stop_loss
            # use_folds=config.OPTIMIZATION['USE_FOLDS_IN_OPTIMIZATION']
        )
        
        if close.shape[1] > 1:
            returns = [np.mean(portfolio.total_return()) for portfolio in portfolios]
        else:
            returns = [portfolio.total_return() for portfolio in portfolios]

        return np.mean(returns)
    
    def _apply_model(self, test_df, best_params):
        portfolios = self.apply_strategy(
                test_df,
                best_params['lowess_fraction'],
                best_params['velocity_up'],
                best_params['velocity_down'],
                best_params['acceleration_up'],
                best_params['acceleration_down'],
                best_params['take_profit'],
                best_params['stop_loss']
            )
            
        if test_df.shape[1] > 1:
            test_value = [np.mean(portfolio.total_return()) for portfolio in portfolios]
        else:
            test_value = [portfolio.total_return() for portfolio in portfolios]

        test_value = np.mean(test_value)
        return test_value

    def _optimize(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        (is_prices, is_dates), (oos_prices, oos_dates) = self._create_rolling_folds()
        print(len(is_prices), len(is_dates), len(oos_prices), len(oos_dates))
        best_chunk_params = dict(
            idx=[],
            best_params=[],
            best_value_train=[],
            value_test=[]
        )
        for chunk_index in tqdm(range(0, len(is_dates))):
            print(min(is_dates[chunk_index]), max(is_dates[chunk_index]))
            train_df = self.data.loc[is_dates[chunk_index]]
            test_df = self.data.loc[oos_dates[chunk_index]]

            func = lambda trial: self.objective(trial, train_df)
            study = optuna.create_study(direction=config.OPTIMIZATION['DIRECTION'])
            early_stopping = EarlyStoppingCallback(config.OPTIMIZATION['EARLY_STOPPING_ITERATIONS'], direction=config.OPTIMIZATION['DIRECTION'])
            study.optimize(func, callbacks=[early_stopping], timeout=config.OPTIMIZATION['OPTIMIZATION_TIME'], n_jobs=config.OPTIMIZATION['N_JOBS'])

            test_value = self._apply_model(test_df, study.best_params)
            
            print(f'BEST CHUNK {chunk_index} PARAMS: {study.best_params}')
            print(f'BEST CHUNK {chunk_index} RETURNS: {study.best_value}')
            print(f'BEST CHUNK {chunk_index} TEST-RET: {test_value}')

            best_chunk_params['idx'].append(chunk_index)
            best_chunk_params['best_params'].append(study.best_params)
            best_chunk_params['best_value_train'].append(study.best_value)
            best_chunk_params['value_test'].append(test_value)
      

        with open("best_params_per_chunk.json", "w") as outfile:
            json.dump(best_chunk_params, outfile, indent=4, sort_keys=False)


    def tutorial(self):
        # figure = self.data.vbt.plot(trace_names=['Price'], width = 1280)
        # figure.show()
        self._optimize()



if __name__ == '__main__':
    wfv = WalkForwardValidation(['SPY'])
    wfv.tutorial()