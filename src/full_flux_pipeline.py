import json
import numpy as np
import logging
logger = logging.getLogger()
logger.setLevel(level=logging.INFO)

from extract import ExtractData
from create_folds import FoldController
from optimizer_controller import Optimizer
from strategy_derivatives import StrategyDerivatives
from plot_controller import PlotController

class FluxPipeline:

    def __init__(self, ticker_list, interval, fold_type):
        # TODO: Add dictionary implementation to be able to extract different periods for different tickers --> {'BTCUSDT': '5m', 'SPY': '1d', ...}
        self._tickers = ticker_list
        self._interval = interval
        self._fold_type = fold_type
        
    def _optimize_folds(self, data, train_index, test_index):

        best_chunk_params = dict()
        for chunk_index in range(0, len(train_index)):
            print(min(train_index[chunk_index]), max(train_index[chunk_index]))
            train_df = data.loc[train_index[chunk_index]]
            test_df = data.loc[test_index[chunk_index]]

            best_params, best_value = Optimizer(train_df).run()
            portfolio = StrategyDerivatives(test_df, **best_params).run()
            test_value = np.mean(portfolio.total_return())
            print(f'BEST CHUNK {chunk_index} PARAMS: {best_params}')
            print(f'BEST CHUNK {chunk_index} RETURNS: {best_value}')
            print(f'BEST CHUNK {chunk_index} TEST-RET: {test_value}')

            best_chunk_params[str(chunk_index)] = {}
            best_chunk_params[str(chunk_index)]['train_dates'] = f'{min(train_index[chunk_index])} <--> {max(train_index[chunk_index])}'
            best_chunk_params[str(chunk_index)]['test_dates'] = f'{min(test_index[chunk_index])} <--> {max(test_index[chunk_index])}'
            best_chunk_params[str(chunk_index)]['best_params'] = best_params
            best_chunk_params[str(chunk_index)]['best_value'] = best_value
            best_chunk_params[str(chunk_index)]['test_value'] = test_value

        return best_chunk_params

    def _test_best_params(self, data, best_params):
        # TODO: Change it to specify the strategy as a class attribute
  
        final_params = None
        best_returns = -9999
        for key in best_params.keys():
            best_chunk_params = best_params[key]['best_params']
            portfolio = StrategyDerivatives(data, **best_chunk_params).run()
            test_value = np.mean(portfolio.total_return())
            if test_value > best_returns:
                final_params = best_chunk_params
                best_returns = test_value
                logging.info(f'[TEST] Strategy updated. \n\nBest params: {final_params}\nBest returns: {best_returns}')

        return final_params, best_returns, portfolio

    def run(self):
        logging.info(f'[EXTRACT DATA] Extraction of {self._tickers} tickers  running...')
        extract_task = ExtractData()
        tickers_data = extract_task.extract_data(self._tickers, overwrite=True)

        logging.info(f'[CREATE FOLDS] Creating folds...')
        create_fold_tasks = FoldController(data=tickers_data, fold_type=self._fold_type)
        _, train_index, _, test_index = create_fold_tasks.run()
        
        logging.info(f'[OPTIMIZATION] Getting best parameters per fold...')
        best_chunk_params = self._optimize_folds(tickers_data, train_index, test_index)

        logging.info(f'[TEST] Getting final strategy...')
        final_params, final_return, portfolio = self._test_best_params(tickers_data, best_chunk_params)

        
        best_chunk_params['final_best_params'] = final_params
        best_chunk_params['final_return'] = final_return
        with open(f"{self._interval}_optimization.json", "w") as outfile:
            json.dump(best_chunk_params, outfile, indent=4, sort_keys=False)


        logging.info(f'[STRATEGY] \n\n Best parameters: {final_params} \n Return: {final_return}')
        
        logging.info(f'[PLOT] Plotting strategy')
        pc = PlotController(tickers_data, best_params=final_params)
        pc.run()

if __name__ == '__main__':
    pipeline = FluxPipeline(ticker_list=['BTCUSDT', 'ETHUSDT'], interval='30m', fold_type='walk-forward')
    pipeline.run()
