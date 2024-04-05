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

analyze(50)