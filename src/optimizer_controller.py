import os
import sys
sys.path.append(f'{os.getcwd()}/strategies/')
import optuna
# optuna.logging.set_verbosity(optuna.logging.WARNING)
import config
import numpy as np
import pandas as pd
from extract import ExtractData
from early_stopping import EarlyStoppingCallback
from strategy_derivatives import StrategyDerivatives
from typing import List

class Optimizer:
    # TODO: Recently IPOed stocks such as Palantir (PLTR) not working due to different sizes on data
    def __init__(self, data, objective='maximize', metric='roi', method='derivate') -> None:
        self._data = data
        self._objective = objective
        self._metric = metric
        self._method = method

    def _get_metrics(self, data, portfolios):
        if data.shape[1] > 1:
            returns = [np.mean(portfolio.total_return()) for portfolio in portfolios]
        else:
            returns = [portfolio.total_return() for portfolio in portfolios]

        return np.mean(returns)
        

    def objective(self, trial: object, data) -> float:
        lowess_fraction = trial.suggest_int('lowess_fraction', 20, 60, step=5)
        velocity_up = trial.suggest_float('velocity_up', 0.05, 1.0, step=0.05)
        velocity_down = trial.suggest_float('velocity_down', -1.0, -0.05, step=0.05)
        acceleration_up = trial.suggest_float('acceleration_up', 0.05, 1.0, step=0.05)
        acceleration_down = trial.suggest_float('acceleration_down', -1.0, -0.05, step=0.05)
        take_profit = trial.suggest_float('take_profit', 0.05, 0.3, step=0.05)
        stop_loss = trial.suggest_float('stop_loss', 0.05, 0.3, step=0.05)
        
        portfolio = StrategyDerivatives(data, 
                                        lowess_fraction, 
                                        velocity_up,
                                        velocity_down, 
                                        acceleration_up, 
                                        acceleration_down, 
                                        take_profit, 
                                        stop_loss).apply_strategy()

        return np.mean(portfolio.total_return())
        
    def run(self):
        func = lambda trial: self.objective(trial=trial, data=self._data)
        study = optuna.create_study(direction=config.OPTIMIZATION['DIRECTION'])
        early_stop = EarlyStoppingCallback(config.OPTIMIZATION['EARLY_STOPPING_ITERATIONS'], direction=config.OPTIMIZATION['DIRECTION'])
        study.optimize(func, callbacks=[early_stop], timeout=config.OPTIMIZATION['OPTIMIZATION_TIME'], n_jobs=config.OPTIMIZATION['N_JOBS'])
        return study.best_params, study.best_value

if __name__ == '__main__':
    data = ExtractData().extract_data(['TSLA', 'SPY'])
    opt = Optimizer(data)
    best_params, best_value = opt.run()
