import pandas as pd

def step_by_step_strategy(df: pd.DataFrame, interval: str, lowess_fraction: int, velocity_up: float,
                          velocity_down: float, acceleration_up: float, acceleration_down: float):

    train_folds, test_folds = create_folds_and_interval_df(df=df, interval=interval, months=months_validation)
    rois = list()
    for fold, fold_idx in enumerate(train_folds):
        df_fold = df.iloc[fold_idx,:]
        x = np.arange(0, df_fold.shape[0])
        y = df_fold['CLOSE'].to_numpy()
        df_fold = derivatives(x, y)
        current_state = None
        buy_price = None
        accumulated_investment = 1
        accumulated_pct_change = 0
        num_transactions = 0
        investment_progress = list()
        df_fold['pct_change'] = df_fold['y'].pct_change()
        for i, row in df_fold.iterrows():
            investment_progress.append(accumulated_investment)
            buy_condition = (
                (
                    (row['first_derivative'] > velocity_down and row['first_derivative'] < velocity_up) and
                    (row['second_derivative'] >= acceleration_up)
                )
            )
            sell_condition = (
                (
                    (row['first_derivative'] > velocity_down and row['first_derivative'] < velocity_up) and
                    (row['second_derivative'] <= acceleration_down)
                )
            )
            current_price = row['y']
            pct_change = row['pct_change']
            if current_state == 1:
                accumulated_investment *= (1 + pct_change)
                accumulated_pct_change += pct_change
                
            if buy_condition:
                if current_state == -1 or current_state == None:
                    current_state = 1
                    accumulated_investment *= (1 - binance_fee)
                    num_transactions += 1
            elif sell_condition or accumulated_pct_change >= take_profit or accumulated_pct_change <= stop_loss:
                if current_state == 1:
                    current_state = -1
                    accumulated_investment *= (1 - binance_fee)
                    accumulated_pct_change = 0
                    num_transactions += 1

if __name__ == '__main__':
    ticker_price = pd.read_csv('../data/validation/BTCEUR_1_MIN_INTERVAL.csv')
    prices = ticker_price.BTCEUR.values
    
    for price in prices:
        