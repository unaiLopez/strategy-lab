import vectorbt as vbt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime

tickers = ['BTC-USD']
end = datetime.datetime.now()
start = end - datetime.timedelta(days=730)
interval = '1h'

ticker_price = vbt.YFData.download(
    tickers,
    missing_index='drop',
    start=start,
    end=end,
    interval=interval
).get('Close')

def custom_indicator(close, interval, velocity_up, velocity_down, acceleration_up, acceleration_down):
    close_interval = close.resample(interval).last()
    # loess denoising---------
    lowess = sm.nonparametric.lowess
    y = close_interval.to_numpy().flatten()
    x = np.array(range(len(y)))
    denoised_ys = lowess(y, x, 1/20, is_sorted=True)[:,1] # the smaller the third parameter, the smoothest the fit
    """
    if len(tickers) == 1:
        y = close_interval.to_numpy().flatten()
        x = np.array(range(len(y)))
        denoised_ys = lowess(y, x, 1/20, is_sorted=True)[:,1] # the smaller the third parameter, the smoothest the fit
    elif len(tickers) > 1:
        denoised_list = list()
        for ticker in tickers:
            y = close_interval[ticker].to_numpy().flatten()
            x = np.array(range(len(y)))
            denoised_y = lowess(y, x, 1/20, is_sorted=True)[:,1] # the smaller the third parameter, the smoothest the fit
            denoised_list.append(denoised_y)
        denoised_ys = np.stack(denoised_list, axis=1)
        print(denoised_ys)
    """
    #-------------------------
    
    # derivation calculation----------
    first_derivative = np.gradient(denoised_ys)
    second_derivative = np.gradient(first_derivative)
    
    first_derivative = first_derivative / np.max(first_derivative)
    second_derivative = second_derivative / np.max(second_derivative)
    #----------------------------------
    
    #df generation and aligning for resample----------
    df_first_derivative = pd.DataFrame(first_derivative, columns=close.columns, index=close_interval.index)
    df_second_derivative = pd.DataFrame(second_derivative, columns=close.columns, index=close_interval.index)
    
    df_first_derivative, _ = df_first_derivative.align(
        close,
        broadcast_axis=0,
        method='ffill',
        join='right'
    )
    df_second_derivative, _ = df_second_derivative.align(
        close,
        broadcast_axis=0,
        method='ffill',
        join='right'
    )
    #-----------------------

    #-----signal conditions----
    first_derivative = df_first_derivative.to_numpy()
    second_derivative = df_second_derivative.to_numpy()
    buy_condition = (
                    (
                        ((first_derivative > velocity_down) & (first_derivative < velocity_up)) &
                        (second_derivative >= acceleration_up)
                    )
                )
    sell_condition = (
        (
            ((first_derivative > velocity_down) & (first_derivative < velocity_up)) &
            (second_derivative <= acceleration_down)
        )
    )

    signals = np.where(buy_condition, 1, 0)
    signals = np.where(sell_condition, -1, signals)
    #-------------------------

    return signals

interval = '12h'
velocity_up = 0.839
velocity_down = -0.013
acceleration_up = 0.028
acceleration_down = -0.65
stop_loss = 0.0057
take_profit = 0.09
fee_rate = 0.001 #0.1%

ind = vbt.IndicatorFactory(
    class_name='first_and_second_order_derivatives',
    short_name='derivatives',
    input_names=['close'],
    param_names=['interval', 'velocity_up', 'velocity_down', 'acceleration_up', 'acceleration_down'],
    output_names=['signals']
).from_apply_func(
    custom_indicator,
    interval=interval,
    velocity_up=velocity_up,
    velocity_down=velocity_down,
    acceleration_up=acceleration_up,
    acceleration_down=acceleration_down,
    keep_pd=True
)

res = ind.run(
    ticker_price
)

entries = res.signals == 1.0
exits = res.signals == -1.0

pf = vbt.Portfolio.from_signals(
    ticker_price,
    entries,
    exits,
    sl_stop=stop_loss,
    tp_stop=take_profit,
    fees=fee_rate
)

print(pf.stats())
print(pf.total_return())

pf.plot(subplots=[
    'cum_returns',
    'trades',
    'drawdowns'
]).show()


