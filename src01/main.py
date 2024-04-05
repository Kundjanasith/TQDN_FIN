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
sim_train.run(rewards_log_path='rewards_log/log.txt')
print('FINISH TRAINING')

## Testing
# sim_test = Simulator(df_test)
# sim_test.predict(policy_path='policy_models/models7.h5',target_path='target_models/models7.h5',log_path='results/7.txt')
# print('FINISH TESTING')

## Visualize
# import matplotlib.pyplot as plt 
# df_vis = pd.read_csv('results/7.txt_data')
# df_vis['Timestamp'] = pd.to_datetime(df_vis['Timestamp'])
# plt.subplot(211)
# plt.plot(df_vis['Timestamp'],df_vis['Close'],label='Price')
# plt.plot(df_vis.loc[df_vis['Action'] == 1.0]['Timestamp'], df_vis['Close'][df_vis['Action'] == 1.0], '^', markersize=5, color='green',label='Long')   
# plt.plot(df_vis.loc[df_vis['Action'] == -1.0]['Timestamp'], df_vis['Close'][df_vis['Action'] == -1.0], 'v', markersize=5, color='red',label='Short')
# plt.legend()
# plt.ylabel('Price')
# plt.subplot(212)
# plt.plot(df_vis['Timestamp'],df_vis['Money'],label='Capital')
# plt.plot(df_vis.loc[df_vis['Action'] == 1.0]['Timestamp'], df_vis['Money'][df_vis['Action'] == 1.0], '^', markersize=5, color='green',label='Long')   
# plt.plot(df_vis.loc[df_vis['Action'] == -1.0]['Timestamp'], df_vis['Money'][df_vis['Action'] == -1.0], 'v', markersize=5, color='red',label='Short')
# plt.legend()
# plt.ylabel('Capital')
# plt.show()


