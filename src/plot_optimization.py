import pandas as pd

from strategy import apply_strategy

if __name__ == '__main__':
    df_optimization_trials = pd.read_csv('../data/results/optimization_trials.csv')
    ticker_price = pd.read_csv('../data/validation/BTCEUR_1_MIN_INTERVAL.csv')
    index = pd.DatetimeIndex(ticker_price.Time.values)
    ticker_price = pd.Series(data=ticker_price.BTCEUR.values, index=index)

    best_trial = df_optimization_trials.iloc[0,:]
    interval = best_trial.params_interval
    take_profit = best_trial.params_take_profit
    stop_loss = best_trial.params_stop_loss
    lowess_fraction = best_trial.params_lowess_fraction
    velocity_up = best_trial.params_velocity_up
    velocity_down = best_trial.params_velocity_down
    acceleration_up = best_trial.params_acceleration_up
    acceleration_down = best_trial.params_acceleration_down

    close_interval = ticker_price.resample(interval).last()
    close_interval.dropna(axis=0, inplace=True)

    portfolio = apply_strategy(close_interval, lowess_fraction, velocity_up, velocity_down, acceleration_up,
                   acceleration_down, take_profit, stop_loss, fee_rate=0.001)
    
    returns = portfolio.total_return()
    print('ALL RETURNS')
    print(returns)
    print(portfolio.stats())
    portfolio.plot(subplots=[
        'cum_returns',
        'trades',
        'drawdowns'
    ]).show()
        
