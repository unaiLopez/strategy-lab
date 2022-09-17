import datetime
import pandas as pd

from typing import Tuple

def train_test_split(data: pd.Series, test_months: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    last_date = data.index.max()
    cut_date = last_date - datetime.timedelta(days=test_months*30)
    data_train = data[data.index <= cut_date]
    data_test = data[data.index > cut_date]

    return data_train, data_test