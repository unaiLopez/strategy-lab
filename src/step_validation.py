import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import train_test_split, get_best_trial_parameters
from strategy import custom_indicator
import config

if __name__ == '__main__':
    interval, lowess_fraction, velocity_up, velocity_down, acceleration_up, acceleration_down, rsi_window, lower_rsi, upper_rsi = get_best_trial_parameters()

    tickers = ['SOLUSDT', 'BTCUSDT', 'ETHUSDT']
    tickers = ['BTCUSDT']
    ticker_price = pd.read_csv(config.OPTIMIZATION['PATH_DATA'])
    timestamp = pd.DatetimeIndex(ticker_price.timestamp.values)
    ticker_price.index = timestamp
    ticker_price = ticker_price[tickers]
    ticker_price[ticker_price.index >= '2020-11-20']
    ticker_price = ticker_price.resample(interval).last()
    ticker_price.dropna(axis=0, inplace=True)
    ticker_price_train, ticker_price_test = train_test_split(data=ticker_price, test_months=config.OPTIMIZATION['TEST_MONTHS'])
    
    current_state = -1
    buy_price = None
    accumulated_investment = 1
    accumulated_pct_change = 0
    num_transactions = 0
    fee_rate = config.OPTIMIZATION['FEE_RATE']
    investment_progress = list()
    for i in tqdm(range(len(ticker_price_test))):
        price = ticker_price_test.iloc[i]
        investment_progress.append(accumulated_investment)

        ticker_price_train = pd.concat([ticker_price_train, pd.Series(price)], axis=0)
        pct_changes = ticker_price_train.pct_change().to_numpy()
        signals = custom_indicator(ticker_price_train, lowess_fraction, velocity_up, velocity_down, acceleration_up, acceleration_down, rsi_window, lower_rsi, upper_rsi)

        pct_change = pct_changes[-1]
        signal = signals[-1]

        if current_state == 1:
            accumulated_investment *= (1 + pct_change)
            accumulated_pct_change += pct_change
        
        if signal == 1.0 and current_state == -1:
            current_state = 1
            accumulated_investment *= (1 - fee_rate)
            num_transactions += 1
        elif signal == -1.0 and current_state == 1:
            current_state = -1
            accumulated_investment *= (1 - fee_rate)
            accumulated_pct_change = 0
            num_transactions += 1
        
    returns = (accumulated_investment - 1)

    print(f"RETURNS IN {config.OPTIMIZATION['TEST_MONTHS']}: {returns}")
    print(f'NUM TRANSACTIONS ARE {num_transactions}')

    df_simulation = pd.DataFrame(data=ticker_price_test.values, index=ticker_price_test.index, columns=['y'])
    df_simulation['time'] = df_simulation.index
    df_simulation['accum'] = investment_progress
    ax = df_simulation.plot(x='time', y='y', legend=False, figsize=(20,10))
    ax2 = ax.twinx()
    df_simulation.plot(x='time', y='accum', ax=ax2, legend=False, color='red', figsize=(20,10))
    ax.figure.legend(['TRUTH', 'ACCUMULATED INVESTMENT'], fontsize=18)
    plt.show()

        
        