import config
import logging
import vectorbt as vbt
logger = logging.getLogger()
logger.setLevel(level=logging.INFO)
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

class FoldController:

    def __init__(self, data: pd.DataFrame, fold_type='normal', test_days=config.OPTIMIZATION['FOLDS_MONTHS'], interval='1Day'):
        self._data = data
        self._fold_type = self._validate_fold_type(fold_type)
        self._test_fold_days = test_days
        self._interval = interval

    def _validate_fold_type(self, fold_type: str) -> str:
        valid_values = ['walk-forward', 'walk-forward-accumulated']
        if fold_type not in valid_values:
            raise NotImplementedError(f'{fold_type} method is not implemented for fold creation... Try any value of the next: {valid_values}')
        return fold_type

    def _fold_size_to_interval_adapter(self, num_records):
        #  In config, folds are specified in days so they have to be transformed into record count depending on the interval
        #  The method returns the number of records that will be in a fold. This includes train and test records.

        multiplier_dict = {
            '5min': 24*12,
            '15min': 24*4,
            '30min': 24*2,
            '1h': 24,
            '2h': 12,
            '4h': 6,
            '6h': 4,
            '12h': 2,
            '24h': 1,
            '1Day': 1
        }

        return int(num_records * multiplier_dict[self._interval]) 

    def _get_creation_fold_parameters(self):
        window_len = self._fold_size_to_interval_adapter(config.FOLDS['FOLD_SIZE_IN_DAYS'])
        fold_test_len = self._fold_size_to_interval_adapter(config.FOLDS['FOLD_TEST_SIZE_IN_DAYS'])

        mod_data = len(self._data) % window_len
        if mod_data != 0:
            self._data = self._data.iloc[mod_data:]

        num_folds = int(len(self._data) / (window_len - fold_test_len))

        return num_folds, window_len, fold_test_len

    def _create_walk_forward_optimization_folds(self, num_folds, full_fold_size, test_fold_size):
        return self._data.vbt.rolling_split(n=num_folds, window_len=full_fold_size, set_lens=(test_fold_size,), left_to_right=False)

    def _create_walk_forward_optimization_accumulated_folds(self, num_folds):
        splitter = TimeSeriesSplit(n_splits=num_folds)
        return self.data.vbt.split(splitter)


    def run(self):
        num_folds, full_fold_size, test_fold_size = self._get_creation_fold_parameters()

        if self._fold_type == 'walk-forward':
            (is_prices, is_dates), (oos_prices, oos_dates)  = self._create_walk_forward_optimization_folds(num_folds, full_fold_size, test_fold_size)
            
        elif self._fold_type == 'walk-forward-accumulated':
            (is_prices, is_dates), (oos_prices, oos_dates) = self._create_walk_forward_optimization_accumulated_folds()

        return is_prices, is_dates, oos_prices, oos_dates

if __name__ == '__main__':
    fc = FoldController()
    fc.run()
