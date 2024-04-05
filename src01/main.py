import pandas as pd 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

from rl_sim import Simulator

filepath = '../data/AAPL_2024_03_06_1D_stock_raw_v5_best.csv'
df = pd.read_csv(filepath)


# Date preparation
df['dateTime'] = pd.to_datetime(df['dateTime'])
df = df[['dateTime','Open','High','Low','Close','Volume']]
df['Timestamp'] = df['dateTime']
df.drop(columns=['dateTime'],inplace=True)
df = df.set_index(['Timestamp'])
print(df.head())

splitingDate = '2023-01-01'
df_train = df[df.index<=splitingDate]
df_test = df[df.index>splitingDate]
print(df_train.shape)
print(df_test.shape)

## Training
sim_train = Simulator(df_train)
sim_train.run()
print('FINISH TRAINING')

## Testing
