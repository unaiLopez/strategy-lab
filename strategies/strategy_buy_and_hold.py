import os
import sys
sys.path.append(f'{os.getcwd()}/src')
import numpy as np
import config
import vectorbt as vbt
from extract import ExtractData

class BuyHoldStrategy:
    
    def __init__(self, data):
        self._data = data

    def _create_signals(self, data):
        signals = np.zeros(len(data))
        signals = np.put(signals, [0, len(data)], [1, -1])
        
        return signals
        

    def apply_strategy(self, ticker_data):
        indicator = vbt.IndicatorFactory(
            class_name='buy_hold_strategy',
            short_name='buy_hold',
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
    fig_dict = BuyHoldStrategy(data).run()