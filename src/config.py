#######################################################################################################################################################
#              OPTIMIZATION PARAMETERS
#######################################################################################################################################################

OPTIMIZATION = {
    'FEE_RATE': 0.001, #broker fee rate
    'TEST_MONTHS': 12, #months of testing
    'FOLDS_MONTHS': 6, #month size for optimization folds
    'OPTIMIZATION_TIME': 120, #time to optimize
    'EARLY_STOPPING_ITERATIONS': 150, #number of optimization iterations before early stopping
    'USE_FOLDS_IN_OPTIMIZATION': False, #optimize using cross validation
    'N_JOBS': -1, #number of cores to execute optimization
    'DIRECTION': 'maximize', #optimization direction
    'PATH_OPTIMIZATION_RESULTS': '../data/results/optimization_trials.csv', #path to save optimization results
    'PATH_DATA': '../data/all_tickers.csv',  #path to optimization data
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

TICKERS = {
    'STOCKS': ["TSLA", "AAPL", "SPOT", "NIO", "BABA", "SPY", "GOOGL"],
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