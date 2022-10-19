import os
import sys
sys.path.append(f'{os.getcwd()}/src')
import utils
import config
import vectorbt as vbt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from extract import ExtractData


class StrategyDerivatives:

    def __init__(self, data, lowess_fraction=None, acceleration_up=None, velocity_down=None, velocity_up=None, acceleration_down=None, take_profit=None, stop_loss=None):
        self._method = 'derivative'
        self._data = data
        self._multiasset = utils.check_multiasset(data)
        
        self._lowess_fraction = lowess_fraction
        self._acceleration_up = acceleration_up
        self._velocity_down = velocity_down
        self._velocity_up = velocity_up
        self._acceleration_down = acceleration_down

        self._take_profit = take_profit
        self._stop_loss = stop_loss

    
    def _normalize_derivative(self, derivative):
        normalized_derivative =  ((np.divide(
                                      (
                                          derivative - np.min(derivative, axis=0, keepdims=True)),
                                          np.max(derivative, axis=0, keepdims=True) - np.min(derivative, axis=0, keepdims=True)
                                      )
                                  ) * 2) - 1

        return normalized_derivative

    def _calculate_derivative_indicator(self):
        lowess = sm.nonparametric.lowess

        if self._multiasset:
            denoised_list = list()

            for ticker in self._data.columns:
                y = self._data[ticker].to_numpy().flatten()
                x = np.array(range(len(y)))
                denoised_y = lowess(y, x, 1/self._lowess_fraction, is_sorted=True)[:,1] # the smaller the third parameter, the smoothest the fit
                denoised_list.append(denoised_y)
            denoised_ys = np.stack(denoised_list, axis=1)
            
            first_derivative = np.gradient(denoised_ys, axis=0)
            second_derivative = np.gradient(first_derivative, axis=0)
            first_derivative = self._normalize_derivative(first_derivative)
            second_derivative = self._normalize_derivative(second_derivative)

        else:
            #do data denoising
            y = self._data.to_numpy().flatten()
            x = np.array(range(len(y)))
            denoised_ys = lowess(y, x, 1/self._lowess_fraction, is_sorted=True)[:,1] # the smaller the third parameter, the smoothest the fit
            
            first_derivative = np.gradient(denoised_ys)
            second_derivative = np.gradient(first_derivative)
            first_derivative = (((first_derivative - np.min(first_derivative)) / (np.max(first_derivative) - np.min(first_derivative))) * 2) - 1
            second_derivative = (((second_derivative - np.min(second_derivative)) / (np.max(second_derivative) - np.min(second_derivative))) * 2) - 1

        
        return first_derivative, second_derivative

    def _get_derivative_signals(self, first_derivative, second_derivative):

        buy_condition = ((
            (first_derivative > self._velocity_down) & (first_derivative < self._velocity_up) & (second_derivative >= self._acceleration_up)
        ))
        sell_condition = ((
            (first_derivative > self._velocity_down) & (first_derivative < self._velocity_up) & (second_derivative <= self._acceleration_down)
        ))    

        signals = np.where(buy_condition, 1, 0)
        signals = np.where(sell_condition, -1, signals)

        return signals


    def _calculate_indicators(self, data, lowess_fraction, velocity_up, velocity_down, acceleration_up, acceleration_down):
        first_derivative, second_derivative = self._calculate_derivative_indicator()
        signals = self._get_derivative_signals(first_derivative, second_derivative)
        return signals

    def apply_strategy(self):
        indicator = vbt.IndicatorFactory(
            class_name='first_and_second_order_derivatives',
            short_name='derivatives',
            input_names=['close'],
            param_names=['lowess_fraction', 'velocity_up', 'velocity_down', 'acceleration_up', 'acceleration_down'],
            output_names=['signals']
        ).from_apply_func(
            self._calculate_indicators,
            keep_pd=True
        )

        res = indicator.run(self._data, 
                            self._lowess_fraction, 
                            self._acceleration_up, 
                            self._velocity_down, 
                            self._velocity_up, 
                            self._acceleration_down)
        
        entries = res.signals == 1.0
        exits = res.signals == -1.0
        pf = vbt.Portfolio.from_signals(
            self._data,
            entries,
            exits,
            fees=config.OPTIMIZATION['FEE_RATE'],
            tp_stop=self._take_profit,
            sl_stop=self._stop_loss
        )


        return pf

    def run(self):
        portfolio = self.apply_strategy()
        return portfolio

if __name__ == '__main__':
    data = ExtractData().extract_data(['SPY', 'TSLA'])
    
    sc = StrategyDerivatives(
        data = data,
        lowess_fraction=50, 
        acceleration_up=0.1, 
        velocity_down=-1.0, 
        velocity_up=0.1, 
        acceleration_down=-1.0,
        take_profit=0.05, 
        stop_loss=0.05
    )
    pf = sc.run()