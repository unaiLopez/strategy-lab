import os
import sys
sys.path.append(f'{os.getcwd()}/src')
import numpy as np
import config
import vectorbt as vbt
from random import randint, seed
from extract import ExtractData

class RandomStrategy:
    
    def __init__(self, data):
        self._data = data
        self._low_threshold = 15
        self._high_threshold = 100 - self._low_threshold
        seed(42)

    def _create_signals(self, data):
        signals = np.array([])
        for _ in range(len(data)):
            x = randint(0, 100)
            if x <= self._low_threshold:
                signals = np.append(signals, -1)
                
            elif x >= self._high_threshold:
                signals = np.append(signals, 1)
                
            else:
                signals = np.append(signals, 0)
     
        buy = (signals==1)
        sell = (signals==-1)
        signals = np.where(buy, 1, 0)
        signals = np.where(sell, -1, signals)
        return signals
        

    def apply_strategy(self, ticker_data):
        indicator = vbt.IndicatorFactory(
            class_name='random_strategy',
            short_name='random',
            input_names=['close'],
            output_names=['signals']
        ).from_apply_func(
            self._create_signals,
            keep_pd=True
        )

        res = indicator.run(ticker_data)
        

        entries = res.signals == 1
        exits = res.signals == -1
        pf = vbt.Portfolio.from_signals(
            ticker_data,
            entries,
            exits,
            fees=config.OPTIMIZATION['FEE_RATE']
        )

        return pf

    
    def run(self):
        fig_dict = {}
        for ticker in self._data.columns:
            data = self._data[ticker]
            pf = self.apply_strategy(data)
            fig = pf.plot(subplots=[
                            'cum_returns',
                            'trades',
                            'drawdowns'
                        ], title=f'{ticker}')

            fig_dict[ticker] = fig
        return fig_dict

        
if __name__ == '__main__':
    ed = ExtractData()
    data = ExtractData().extract_data(['SPY', 'TSLA'])
    fig_dict = RandomStrategy(data).run()