import importlib
from temSim import TradingEnv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
stateLength = 30
observationSpace = 1 + (stateLength-1)*4
actionSpace = 2
strategy = 'TDQN'
strategyModule = importlib.import_module(str(strategy))
className = getattr(strategyModule, strategy)
tradingStrategy = className(observationSpace, actionSpace)
tradingStrategy.loadModel('models/49')
import pandas as pd 
filepath = '../data/AAPL_2024_03_06_1D_stock_raw_v5_best.csv'
data = pd.read_csv(filepath)
data['dateTime'] = pd.to_datetime(data['dateTime'])
startingDate = data['dateTime'][0]
splitingDate = '2023-01-01'
endingDate = data['dateTime'][len(data)-1]
stock = 'APPl'
money = 100000
percentageCosts = [0, 0.1, 0.2]
transactionCosts = percentageCosts[1]/100

trainingEnv = TradingEnv(data, stock, startingDate, splitingDate, money, stateLength, transactionCosts)
testingEnv = TradingEnv(data, stock, splitingDate, endingDate, money, stateLength, transactionCosts)

from tradingPerformance import PerformanceEstimator
testingEnv = tradingStrategy.testing2(testingEnv)
p = PerformanceEstimator(testingEnv.data)
p.displayPerformance('TDQN')



"""
# trainingEnv = tradingStrategy.testing(trainingEnv, trainingEnv)
testingEnv = tradingStrategy.testing2(testingEnv)

# ratio = trainingEnv.data['Money'][len(trainingEnv.data)-1]/testingEnv.data['Money'][0]
# testingEnv.data['Money'] = ratio * testingEnv.data['Money']
# dataframes = [testingEnv.data]

# data = pd.concat(dataframes)
data = testingEnv.data
import matplotlib.pyplot as plt
# Set the Matplotlib figure and subplots
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time')

# Plot the first graph -> Evolution of the stock market price
# trainingEnv.data['Close'].plot(ax=ax1, color='blue', lw=2), label='_nolegend_'
testingEnv.data['Close'].plot(ax=ax1, color='blue', lw=2) 
ax1.plot(data.loc[data['Action'] == 1.0].index, data['Close'][data['Action'] == 1.0], '^', markersize=5, color='green')   
ax1.plot(data.loc[data['Action'] == -1.0].index, data['Close'][data['Action'] == -1.0], 'v', markersize=5, color='red')
        
# Plot the second graph -> Evolution of the trading capital
# trainingEnv.data['Money'].plot(ax=ax2, color='blue', lw=2), label='_nolegend_'
testingEnv.data['Money'].plot(ax=ax2, color='blue', lw=2) 
ax2.plot(data.loc[data['Action'] == 1.0].index, data['Money'][data['Action'] == 1.0], '^', markersize=5, color='green')   
ax2.plot(data.loc[data['Action'] == -1.0].index, data['Money'][data['Action'] == -1.0], 'v', markersize=5, color='red')

# Plot the vertical line seperating the training and testing datasets
ax1.axvline(pd.Timestamp(splitingDate), color='black', linewidth=2.0)
ax2.axvline(pd.Timestamp(splitingDate), color='black', linewidth=2.0)
        
ax1.legend(["Price", "Long",  "Short", "Train/Test separation"])
ax2.legend(["Capital", "Long", "Short", "Train/Test separation"])
plt.show()
"""
