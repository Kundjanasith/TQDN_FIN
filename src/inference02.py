import importlib
from temSim import TradingEnv
import warnings
from tradingPerformance import PerformanceEstimator
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt 
import pandas as pd 


# def computerPerf(testingEnv):


def analyze(ep_test=None):
    stateLength = 30
    observationSpace = 1 + (stateLength-1)*4
    actionSpace = 2
    strategy = 'TDQN'
    strategyModule = importlib.import_module(str(strategy))
    className = getattr(strategyModule, strategy)
    tradingStrategy = className(observationSpace, actionSpace)
    filepath = '../data/AAPL_2024_03_06_1D_stock_raw_v5_best.csv'
    data = pd.read_csv(filepath)
    data['dateTime'] = pd.to_datetime(data['dateTime'])
    startingDate = data['dateTime'][0]
    splitingDate = '2023-01-01'
    endingDate = data['dateTime'][len(data)-1]
    stock = 'APPl'
    money = 100000
    # percentageCosts = [0, 0.1, 0.2]
    # transactionCosts = percentageCosts[1]/100
    transactionCosts = 0
    PnL_arr = []
    annualizedReturn_arr = []
    annualizedVolatily_arr = []
    sharpeRatio_arr = []
    sortinoRatio_arr = []
    maxDD_arr = []
    maxDDD_arr = []
    profitability_arr = []
    averageProfitLossRatio_arr = []
    skewness_arr = []
    for i in range(ep_test):
        tradingStrategy.loadModel('models3/%d'%i)
        testingEnv = TradingEnv(data, stock, splitingDate, endingDate, money, stateLength, transactionCosts)
        testingEnv = tradingStrategy.testing2(testingEnv)
        p = PerformanceEstimator(testingEnv.data)
        PnL, annualizedReturn, annualizedVolatily, sharpeRatio, sortinoRatio, maxDD, maxDDD, profitability, averageProfitLossRatio, skewness = p.computePerformance2()
        PnL_arr.append(PnL)
        annualizedReturn_arr.append(annualizedReturn)
        annualizedVolatily_arr.append(annualizedVolatily)
        sharpeRatio_arr.append(sharpeRatio)
        sortinoRatio_arr.append(sortinoRatio)
        maxDD_arr.append(maxDD)
        maxDDD_arr.append(maxDDD)
        profitability_arr.append(profitability)
        averageProfitLossRatio_arr.append(averageProfitLossRatio)
        skewness_arr.append(skewness)
    plt.clf()
    plt.title("Profit & Loss (P&L)")
    plt.plot(PnL_arr)
    plt.savefig('fig_inference02/PnL.png',bbox_inches='tight')

    plt.clf()
    plt.title("Annualized Return")
    plt.plot(annualizedReturn_arr)
    plt.savefig('fig_inference02/annualizedReturn.png',bbox_inches='tight')

    plt.clf()
    plt.title("Sharpe Ratio")
    plt.plot(sharpeRatio_arr)
    plt.savefig('fig_inference02/sharpeRatio.png',bbox_inches='tight')

    plt.clf()
    plt.title("Sortino Ratio")
    plt.plot(sortinoRatio_arr)
    plt.savefig('fig_inference02/sortinoRatio.png',bbox_inches='tight')

    plt.clf()
    plt.title("Maximum Drawdown")
    plt.plot(maxDD_arr)
    plt.savefig('fig_inference02/maxDD.png',bbox_inches='tight')

    plt.clf()
    plt.title("Maximum Drawdown Duration")
    plt.plot(maxDDD_arr)
    plt.savefig('fig_inference02/maxDDD.png',bbox_inches='tight')

    plt.clf()
    plt.title("Profitability")
    plt.plot(profitability_arr)
    plt.savefig('fig_inference02/profitability.png',bbox_inches='tight')

    plt.clf()
    plt.title("Ratio Average Profit/Loss")
    plt.plot(averageProfitLossRatio_arr)
    plt.savefig('fig_inference02/averageProfitLossRatio.png',bbox_inches='tight')

    plt.clf()
    plt.title("Skewness")
    plt.plot(skewness_arr)
    plt.savefig('fig_inference02/skewness.png',bbox_inches='tight')

import matplotlib.pyplot as plt

def analyze2():
    stateLength = 30
    observationSpace = 1 + (stateLength-1)*4
    actionSpace = 2
    strategy = 'TDQN'
    strategyModule = importlib.import_module(str(strategy))
    className = getattr(strategyModule, strategy)
    tradingStrategy = className(observationSpace, actionSpace)
    filepath = '../data/AAPL_2024_03_06_1D_stock_raw_v5_best.csv'
    data = pd.read_csv(filepath)
    data['dateTime'] = pd.to_datetime(data['dateTime'])
    startingDate = data['dateTime'][0]
    splitingDate = '2023-01-01'
    endingDate = data['dateTime'][len(data)-1]
    stock = 'APPl'
    money = 100000
    testingEnv = TradingEnv(data, stock, splitingDate, endingDate, money, stateLength, transactionCosts)
    for i in range(50):
        tradingStrategy.loadModel('models3/%d'%i)
        testingEnv = tradingStrategy.testing2(testingEnv)
        plt.clf()
        data = testingEnv.data
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
        ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time')

        # Plot the first graph -> Evolution of the stock market price
        # trainingEnv.data['Close'].plot(ax=ax1, color='blue', lw=2), label='_nolegend_'
        testingEnv.data['Close'].plot(ax=ax1, color='blue', lw=2) 
        ax1.plot(data.loc[data['Action'] == 1.0].index, data['Close'][data['Action'] == 1.0], '^', markersize=5, color='green')   
        ax1.plot(data.loc[data['Action'] == -1.0].index, data['Close'][data['Action'] == -1.0], 'v', markersize=5, color='red')
                
        testingEnv.data['Money'].plot(ax=ax2, color='blue', lw=2) 
        ax2.plot(data.loc[data['Action'] == 1.0].index, data['Money'][data['Action'] == 1.0], '^', markersize=5, color='green')   
        ax2.plot(data.loc[data['Action'] == -1.0].index, data['Money'][data['Action'] == -1.0], 'v', markersize=5, color='red')

        # ax1.axvline(pd.Timestamp(splitingDate), color='black', linewidth=2.0)
        # ax2.axvline(pd.Timestamp(splitingDate), color='black', linewidth=2.0)
                
        ax1.legend(["Price", "Long",  "Short", "Train/Test separation"])
        ax2.legend(["Capital", "Long", "Short", "Train/Test separation"])
        plt.show()

# analyze(50)
analyze2()