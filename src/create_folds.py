import config

def create_folds(resampled_df, interval):
    if interval == '5m':
        size = (30*24*12*config.FOLDS_MONTHS)
    if interval == '15m':
        size = (30*24*4*config.FOLDS_MONTHS)
    elif interval == '30m':
        size = (30*24*2*config.FOLDS_MONTHS)
    elif interval == '1h':
        size = (30*24*config.FOLDS_MONTHS)
    elif interval == '2h':
        size = (30*12*config.FOLDS_MONTHS)
    elif interval == '4h':
        size = (30*6*config.FOLDS_MONTHS)
    elif interval == '6h':
        size = (30*4*config.FOLDS_MONTHS)
    elif interval == '12h':
        size = (30*2*config.FOLDS_MONTHS)
    elif interval == '24h':
        size = (30*config.FOLDS_MONTHS)

    n_splits = (resampled_df.shape[0]) // size
    folds = list()
    for i in range(n_splits):
        fold = range((size*i), size*(i+1))
        folds.append(fold)
    
    return folds