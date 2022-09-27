from strategy import apply_strategy
from utils import train_test_split, get_best_trial_parameters, prepare_asset_dataframe_format
import config

if __name__ == '__main__':
    interval, lowess_fraction, velocity_up, velocity_down, acceleration_up, acceleration_down, take_profit, stop_loss = get_best_trial_parameters()
    
    ticker_price = prepare_asset_dataframe_format(tickers=config.OPTIMIZATION['TICKERS'])
    close_interval = ticker_price.resample(interval).last()
    close_interval.dropna(axis=0, inplace=True)
    _, price_test = train_test_split(data=close_interval, test_months=config.OPTIMIZATION['TEST_MONTHS'])

    for ticker in config.OPTIMIZATION['TICKERS']:
        ticker_price_test= price_test[ticker]
        portfolios = apply_strategy(ticker_price_test, interval, lowess_fraction, velocity_up, velocity_down, acceleration_up, acceleration_down, take_profit, stop_loss, use_folds=False)
        portfolio = portfolios[0]

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
        
