
import importlib 
from rl_env import TradingEnv
from rl_perf import PerformanceEstimator


class Simulator:
    def __init__(self, data):
        self.data = data
        self.money = 100000
        self.numberOfEpisodes = 10
        self.stateLength = 30
        self.observationSpace = 1 + (self.stateLength-1)*4
        self.actionSpace = 2
        self.percentageCosts = [0, 0.1, 0.2]
        self.transactionCosts = self.percentageCosts[1]/100

    def run(self, rewards_log_path):
        strategy = 'temDQN'
        trainingParameters = [self.numberOfEpisodes] 
        trainingEnv = TradingEnv(self.data, self.money, self.stateLength, self.transactionCosts)
        strategyModule = importlib.import_module(str(strategy))
        className = getattr(strategyModule, strategy)
        tradingStrategy = className(self.observationSpace, self.actionSpace)
        trainingEnv = tradingStrategy.training(trainingEnv, trainingParameters=trainingParameters, rewards_log_path=rewards_log_path)
    
    def predict(self, policy_path, target_path, log_path):
        strategy = 'temDQN'
        testingEnv = TradingEnv(self.data, self.money, self.stateLength, self.transactionCosts)
        strategyModule = importlib.import_module(str(strategy))
        className = getattr(strategyModule, strategy)
        tradingStrategy = className(self.observationSpace, self.actionSpace)
        testingEnv = tradingStrategy.predict(testingEnv, policy_path=policy_path, target_path=target_path)
        testingEnv.data.to_csv(log_path+'_data')
        p = PerformanceEstimator(testingEnv.data)
        PnL, annualizedReturn, annualizedVolatily, sharpeRatio, sortinoRatio, maxDD, maxDDD, profitability, averageProfitLossRatio, skewness = p.computePerformance()
        file_o = open(log_path,'w')
        file_o.write('PnL,annualizedReturn,annualizedVolatily,sharpeRatio,sortinoRatio,maxDD,maxDDD,profitability,averageProfitLossRatio,skewness\n')
        file_o.write('%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n'%(PnL, annualizedReturn, annualizedVolatily, sharpeRatio, sortinoRatio, maxDD, maxDDD, profitability, averageProfitLossRatio, skewness))
        file_o.close()