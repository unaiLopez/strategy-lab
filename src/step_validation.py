import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import train_test_split, get_best_trial_parameters, prepare_asset_dataframe_format
from strategy import custom_indicator
import config

if __name__ == '__main__':
    interval, lowess_fraction, velocity_up, velocity_down, acceleration_up, acceleration_down, take_profit, stop_loss = get_best_trial_parameters()

    ticker_price = prepare_asset_dataframe_format(tickers=config.OPTIMIZATION['TICKERS'])
    ticker_price = ticker_price.resample(interval).last()
    ticker_price.dropna(axis=0, inplace=True)
    price_train, price_test = train_test_split(data=ticker_price, test_months=config.OPTIMIZATION['TEST_MONTHS'])

    for ticker in config.OPTIMIZATION['TICKERS']:
        ticker_price_train = price_train[ticker].to_frame()
        ticker_price_test = price_test[ticker].to_frame()
    
        current_state = -1
        buy_price = None
        accumulated_investment = 1
        accumulated_pct_change = 0
        num_transactions = 0
        fee_rate = config.OPTIMIZATION['FEE_RATE']
        investment_progress = list()
        for i in tqdm(range(len(ticker_price_test))):
            price = ticker_price_test.iloc[i].values[0]
            investment_progress.append(accumulated_investment)
            df_price = pd.DataFrame(data=[price], columns=list(ticker_price_train.columns))
            ticker_price_train = pd.concat([ticker_price_train, df_price], axis=0)
            pct_changes = ticker_price_train.pct_change().to_numpy()
            signals = custom_indicator(ticker_price_train, lowess_fraction, velocity_up, velocity_down, acceleration_up, acceleration_down)

            pct_change = pct_changes[-1,0]
            signal = signals[-1]
            if current_state == 1:
                accumulated_investment *= (1 + pct_change)
                accumulated_pct_change += pct_change
            if signal == 1.0 and current_state == -1:
                current_state = 1
                accumulated_investment *= (1 - fee_rate)
                num_transactions += 1
            elif (signal == -1.0 and current_state == 1) or (accumulated_pct_change >= take_profit) or (accumulated_pct_change <= -stop_loss):
                current_state = -1
                accumulated_investment *= (1 - fee_rate)
                accumulated_pct_change = 0
                num_transactions += 1
            
        returns = (accumulated_investment - 1)

        print(f'TICKER {ticker}')
        print(f"RETURNS IN {config.OPTIMIZATION['TEST_MONTHS']}: {returns}")
        print(f'NUM TRANSACTIONS ARE {num_transactions}')
        df_simulation = pd.DataFrame(data=ticker_price_test.values, index=ticker_price_test.index, columns=['y'])
        df_simulation['time'] = df_simulation.index
        df_simulation['accum'] = investment_progress
        ax = df_simulation.plot(x='time', y='y', legend=False, figsize=(20,10))
        ax2 = ax.twinx()
        df_simulation.plot(x='time', y='accum', ax=ax2, legend=False, color='red', figsize=(20,10))
        ax.figure.legend(['TRUTH', 'ACCUMULATED INVESTMENT'], fontsize=18)
        plt.title(f'{ticker}', fontsize=24)
    plt.show()

        
        