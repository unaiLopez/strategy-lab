import pandas as pd

from datetime import datetime
from binance.client import Client

client = Client()            
df = pd.DataFrame(client.get_historical_klines('BTCUSDT', '5m', '2017-01-01', datetime.now().strftime("%Y-%m-%d")))
df = df[[0, 4]]
df.columns = ['timestamp', 'close']
df = df.set_index('timestamp')
df.index = pd.to_datetime(df.index, unit='ms')

df.to_csv('../data/validation/BTCUSDT_5_MIN_INTERVAL.csv')