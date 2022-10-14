from typing import Dict
import os
import json
import pandas as pd
from strategy_derivatives import StrategyDerivatives
from extract import ExtractData
from utils import train_test_split, get_best_trial_parameters, prepare_asset_dataframe_format
import config

class PlotController:

    def __init__(self, data, portfolio=None, use_folds=False, best_params=None, train_folds=None, test_folds=None,  plot_type: str='full') -> None: 
        self._data = data
        # self.ticker_price = prepare_asset_dataframe_format(tickers=config.OPTIMIZATION['TICKERS'])
        self.type = plot_type
        self._train_folds = train_folds
        self._test_folds = test_folds
        self._use_folds = use_folds
        self._pf = portfolio
        self._best_params = best_params
        self._validate_fold_usage()


    def _validate_fold_usage(self):
        if self._use_folds and (self._train_folds is None or self._test_folds is None):
            raise Exception('Not possible to use folds without receiveing them.')

    def _plot_portfolio(self):
        rets = self._pf.total_return()


    # def _plot_portfolio(self, pf):
    #     returns = pf.total_return()
    #     print(f'TICKER {ticker}')
    #     print('ALL RETURNS')
    #     print(returns)
    #     print(portfolio.stats())
    #     portfolio.plot(subplots=[
    #         'cum_returns',
    #         'trades',
    #         'drawdowns'
    #     ], title=f'{ticker}').show()

    def apply_strategy_and_plot(self, data, ticker):
        portfolios = StrategyDerivatives(data, **self._best_params).run()
        portfolio = portfolios

        returns = portfolio.total_return()
        print(f'TICKER {ticker}')
        print('ALL RETURNS')
        print(returns)
        print(portfolio.stats())
        portfolio.plot(subplots=[
            'cum_returns',
            'trades',
            'drawdowns'
        ], title=f'{ticker}').show()


    def _get_relevant_data(self) -> pd.DataFrame:
        # TODO: Add here interval handling depending on type
        if self.type == 'full':
            # TODO: Implement handling both of them
            return self.data
        price_train, price_test = train_test_split(data=self._data, test_months=config.OPTIMIZATION['TEST_MONTHS'])
        if self.type == 'train':
            del price_test
            return price_train
        elif self.type == 'test':
            del price_train
            return price_test

        else:
            raise NotImplemented("type value has to be some of these: ['train', 'test', full].")

    def run(self):
        # 1. Coger resultados
        print(self._data.columns)
        for ticker in self._data.columns:
            data = self._data[ticker]
            self.apply_strategy_and_plot(data, ticker)
        # 3. ¿Hay más de un fold?

if __name__ == '__main__':
    with open(f'{os.getcwd()}/best_params_per_fold.json') as json_file:
        best_params = json.load(json_file)["1"]["best_params"]
    data =  ExtractData().extract_data(['SPY', 'TSLA'])
    pc = PlotController(data, best_params=best_params)
    pc.run()