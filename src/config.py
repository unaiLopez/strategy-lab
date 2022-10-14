import os

#######################################################################################################################################################
#              FOLDS PARAMETERS
#######################################################################################################################################################

FOLDS = {
    'NUM_FOLDS': 10,
    'FOLD_TYPE': 'walk-forward',
    'FOLD_SIZE_IN_DAYS': 120,
    'FOLD_TEST_SIZE_IN_DAYS': 30,
}

#######################################################################################################################################################
#              OPTIMIZATION PARAMETERS
#######################################################################################################################################################

OPTIMIZATION = {
    'FEE_RATE': 0.001, #broker fee rate
    'TEST_MONTHS': 12, #months of testing
    'FOLDS_MONTHS': 6, #month size for optimization folds
    'OPTIMIZATION_TIME': 60, #time to optimize
    'EARLY_STOPPING_ITERATIONS': 200, #number of optimization iterations before early stopping
    'USE_FOLDS_IN_OPTIMIZATION': False, #optimize using cross validation
    'N_JOBS': -1, #number of cores to execute optimization
    'DIRECTION': 'maximize', #optimization direction
    'PATH_OPTIMIZATION_RESULTS': f'{os.getcwd()}/data/results/optimization_trials.csv', #path to save optimization results
    'PATH_DATA': f'{os.getcwd()}/data/all_tickers.csv',  #path to optimization data
    'START_DATE': '2017-01-01', #optimization data start date
    'TICKERS': ['ETHUSDT', 'SOLUSDT', 'BTCUSDT'] #tickers to optimize strategy
}


#######################################################################################################################################################
#              EXTRACT DATA PARAMETERS
#######################################################################################################################################################

CREDENTIALS = {
    'ALPACA': {
        'API_KEY': 'PKMRG1YGM6JYL4Y5P2E8',
        'SECRET_KEY': '54dmcvvAZepOqs3HZVXAuxwzLsZ5TkEFiLvYViLD '
    }
}

ASSET_TYPE = ['crypto', 'stocks']

TICKERS = {
    'STOCKS': ["TSLA", "AAPL", "SPOT", "NIO", "BABA", "SPY", "GOOGL", "NVIDIA"],
    'CRYPTO': ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]
}

ACCEPTED_INTERVALS = {
    '1m': '1Min',
    '2m': '2Min',
    '5m': '5Min',
    '15m': '10Min',
    '30m': '30Min',
    '90m': '90Min',
    '1h': '1Hour',
    '2h': '2Hour',
    '3h': '3Hour',
    '6h': '6Hour',
    '12h': '12Hour',
    '1d': '1Day'
}
    
MODE = ['save', 'retrieve']