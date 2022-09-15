import csv
import os
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from typing import List

class Strategy:
    def __init__(self, fee: float = 0.001):
        self.results_path = '../../outputs/strategy_optimization_folds.csv'
        self.field_names = [
            'TICKER', 'FOLD', 'INTERVAL', 'TAKE_PROFIT', 'STOP_LOSS', 'MONTHS_VALIDATION', 'VELOCITY_UP',
            'VELOCITY_DOWN', 'ACCELERATION_UP', 'ACCELERATION_DOWN', 'ACCUMULATED_INVESTMENT', 'NUM_TRANSACTIONS',
            'ROI'
        ]
        self.fee = fee

    def _create_folds(self, df: pd.DataFrame, interval: str, months: int) -> List[List[int]]:
        time_column = df.columns[0]
        folds = list()
        size = None
        if interval == '5M':
            df_interval = df[df[time_column].dt.minute % 5 == 0].copy()
            size = (30*24*12*months) # 3 months
        if interval == '15M':
            df_interval = df[df[time_column].dt.minute % 15 == 0].copy()
            size = (30*24*4*months) # 3 months
        elif interval == '30M':
            df_interval = df[df[time_column].dt.minute % 30 == 0].copy()
            size = (30*24*2*months) # 3 months
        elif interval == '1H':
            df_interval = df[df[time_column].dt.minute == 0].copy()
            size = (30*24*months) # 3 months
        elif interval == '2H':
            df_interval = df[(df[time_column].dt.hour % 2 == 0) & (df[time_column].dt.minute == 0)].copy()
            size = (30*12*months) # 3 months
        elif interval == '4H':
            df_interval = df[(df[time_column].dt.hour % 4 == 0) & (df[time_column].dt.minute == 0)].copy()
            size = (30*6*months) # 3 months
        elif interval == '6H':
            df_interval = df[(df[time_column].dt.hour % 6 == 0) & (df[time_column].dt.minute == 0)].copy()
            size = (30*4*months) # 3 months
        elif interval == '12H':
            df_interval = df[(df[time_column].dt.hour % 12 == 0) & (df[time_column].dt.minute == 0)].copy()
            size = (30*2*months) # 3 months

        n_splits = math.floor(df_interval.shape[0] / size)
        for split in range(1, n_splits + 1):
            if split == 1:
                folds.append(df_interval.iloc[:size,:].index)
            elif split > 1:
                folds.append(df_interval.iloc[(split-1)*size:size*split,:].index)
        
        train_folds = folds[:-2]
        test_folds = folds[-2:]

        return train_folds, test_folds

    def _calculate_derivatives(self, x: np.array, y: np.array):
        # loess denoising---------
        lowess = sm.nonparametric.lowess
        result = lowess(y, x, 1/20, is_sorted=True) # the smaller the third parameter, the smoothest the fit
        #-------------------------
        
        # derivation calculation----------
        dx = np.gradient(x)
        dy = np.gradient(result[:, 1])
        dy_second = np.gradient(dy)
        first_derivative = dy / dx
        second_derivative = dy_second / dx
        y = y / np.max(y)
        first_derivative = first_derivative / np.max(first_derivative)
        second_derivative = second_derivative / np.max(second_derivative)
        #----------------------------------
        
        #df generation----------
        data_matrix = np.stack((y,first_derivative, second_derivative), axis = 1)
        df = pd.DataFrame(data_matrix, columns = ['y', 'first_derivative', 'second_derivative'])
        #-----------------------
        
        # std condition inserted to the df-----------
        std_second = np.std(second_derivative)
        mean_second = np.mean(second_derivative)
        lower_bound = mean_second - 7 * std_second
        upper_bound = mean_second + 7 * std_second
        condition_second_derivative_is_outlier = (
                                                  (df['second_derivative'] <= lower_bound) |
                                                  (df['second_derivative'] >= upper_bound)
                                                  )
        df.loc[condition_second_derivative_is_outlier, 'outlier'] = True
        condition_is_outlier = (df['outlier'] == True)
        #--------------------------------------------
        
        # interpolation---------------------------
        df.loc[condition_is_outlier, 'first_derivative'] = np.nan
        df.loc[condition_is_outlier, 'second_derivative'] = np.nan
        df = df.interpolate(method = 'quadratic')
        df.drop('outlier', axis=1, inplace=True)

        return df
    
    def _read_data(self, ticker):
        df = pd.read_csv(f'../../data/validation/{ticker}_1_MIN_INTERVAL.csv', parse_dates=['Time'])
        df.columns = ['TIME', 'CLOSE']

        return df

    def _append_result_csv(self, path: str, field_names: list, ticker: str, fold: List[List[int]], interval: str, take_profit: float, 
                           stop_loss: float, months_validation: int, velocity_up: float, velocity_down: float, acceleration_up: float,
                           acceleration_down: float, accumulated_investment: float, num_transactions: int, roi: float) -> None:
        dict_row = {
            'TICKER': ticker,
            'FOLD': fold,
            'INTERVAL': interval,
            'TAKE_PROFIT': take_profit,
            'STOP_LOSS': stop_loss,
            'MONTHS_VALIDATION': months_validation,
            'VELOCITY_UP': velocity_up,
            'VELOCITY_DOWN': velocity_down,
            'ACCELERATION_UP': acceleration_up,
            'ACCELERATION_DOWN': acceleration_down,
            'ACCUMULATED_INVESTMENT': accumulated_investment,
            'NUM_TRANSACTIONS': num_transactions,
            'ROI': roi
        }
        if not os.path.exists(path):
            df_results = pd.DataFrame(columns=field_names)
            df_results.to_csv(path, index=False)
        else:
            with open(path, mode='a') as csv_file:
                dict_object = csv.DictWriter(csv_file, fieldnames=field_names)
                dict_object.writerow(dict_row)


    def apply_strategy(self, ticker: str, interval: str, take_profit: float, stop_loss: float, months_validation: int,
                      velocity_up: float, velocity_down: float, acceleration_up: float, acceleration_down: float):

        df = self._read_data(ticker)
        train_folds, _ = self._create_folds(df, interval, months_validation)
        rois = list()
        for fold, fold_idx in enumerate(train_folds):
            current_state = None
            accumulated_investment = 1
            accumulated_pct_change = 0
            num_transactions = 0
            investment_progress = list()

            df_fold = df.iloc[fold_idx,:]
            x = np.arange(0, df_fold.shape[0])
            y = df_fold['CLOSE'].to_numpy()
            df_fold = self._calculate_derivatives(x, y)
            df_fold['pct_change'] = df_fold['y'].pct_change()
            for i, row in df_fold.iterrows():
                investment_progress.append(accumulated_investment)
                buy_condition = (
                    (
                        (row['first_derivative'] > velocity_down and row['first_derivative'] < velocity_up) and
                        (row['second_derivative'] >= acceleration_up)
                    )
                )
                sell_condition = (
                    (
                        (row['first_derivative'] > velocity_down and row['first_derivative'] < velocity_up) and
                        (row['second_derivative'] <= acceleration_down)
                    )
                )
                pct_change = row['pct_change']
                if current_state == 1:
                    accumulated_investment *= (1 + pct_change)
                    accumulated_pct_change += pct_change
                    
                if buy_condition:
                    if current_state == -1 or current_state == None:
                        current_state = 1
                        accumulated_investment *= (1 - self.fee)
                        num_transactions += 1
                elif sell_condition or accumulated_pct_change >= take_profit or accumulated_pct_change <= stop_loss:
                    if current_state == 1:
                        current_state = -1
                        accumulated_investment *= (1 - self.fee)
                        accumulated_pct_change = 0
                        num_transactions += 1
            
            roi = (accumulated_investment - 1) * 100
            rois.append(roi)

            self._append_result_csv(
                self.results_path, self.field_names, ticker, fold, interval, take_profit, stop_loss, months_validation, velocity_up,
                velocity_down, acceleration_up, acceleration_down, accumulated_investment, num_transactions, roi
            )
            
        return np.mean(rois)

    def _plot_strategy_performance(self, df: pd.DataFrame, take_profit: float, stop_loss: float,
                    velocity_up: float, velocity_down: float, acceleration_up: float, acceleration_down: float, folds: List[List[int]], type_folds: str) -> None:

        for fold, fold_idx in enumerate(folds):
            current_state = None
            accumulated_investment = 1
            accumulated_pct_change = 0
            num_transactions = 0
            investment_progress = list()

            df_fold = df.iloc[fold_idx,:]
            x = np.arange(0, df_fold.shape[0])
            y = df_fold['CLOSE'].to_numpy()
            df_fold = self._calculate_derivatives(x, y)
            df_fold['pct_change'] = df_fold['y'].pct_change()
            for i, row in df_fold.iterrows():
                investment_progress.append(accumulated_investment)
                buy_condition = (
                    (
                        (row['first_derivative'] > velocity_down and row['first_derivative'] < velocity_up) and
                        (row['second_derivative'] >= acceleration_up)
                    )
                )
                sell_condition = (
                    (
                        (row['first_derivative'] > velocity_down and row['first_derivative'] < velocity_up) and
                        (row['second_derivative'] <= acceleration_down)
                    )
                )
                pct_change = row['pct_change']
                if current_state == 1:
                    accumulated_investment *= (1 + pct_change)
                    accumulated_pct_change += pct_change
                    
                if buy_condition:
                    if current_state == -1 or current_state == None:
                        current_state = 1
                        accumulated_investment *= (1 - self.fee)
                        num_transactions += 1
                elif sell_condition or accumulated_pct_change >= take_profit or accumulated_pct_change <= stop_loss:
                    if current_state == 1:
                        current_state = -1
                        accumulated_investment *= (1 - self.fee)
                        accumulated_pct_change = 0
                        num_transactions += 1
            
            roi = (accumulated_investment - 1) * 100

            df_fold['time'] = df_fold.index
            df_fold['accum'] = investment_progress
            ax = df_fold.plot(x='time', y='y', legend=False, figsize=(20,10))
            ax2 = ax.twinx()
            df_fold.plot(x='time', y='accum', ax=ax2, legend=False, color='red', figsize=(20,10))
            ax.figure.legend(['TRUTH', 'ACCUMULATED INVESTMENT'], fontsize=18)
            if type_folds == 'train':
                plt.title(f'TRAIN FOLD={fold} | ROI={roi}')
            else:
                plt.title(f'TEST FOLD={fold} | ROI={roi}')
            plt.show()

    
    def plot_train_test_folds_performance(self, ticker: str, interval: str, take_profit: float, stop_loss: float, months_validation: int,
                    velocity_up: float, velocity_down: float, acceleration_up: float, acceleration_down: float) -> None:

        df = self._read_data(ticker)
        train_folds, test_folds = self._create_folds(df, interval, months_validation)


        self._plot_strategy_performance(df, take_profit, stop_loss, velocity_up,
                    velocity_down, acceleration_up, acceleration_down, train_folds, 'train')
        
        self._plot_strategy_performance(df, take_profit, stop_loss, velocity_up,
                    velocity_down, acceleration_up, acceleration_down, test_folds, 'test')