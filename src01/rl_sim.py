
import importlib 
from rl_env import TradingEnv


class Simulator:
    def __init__(self, data):
        self.data = data
        self.money = 100000
        self.numberOfEpisodes = 50
        self.stateLength = 30
        self.observationSpace = 1 + (self.stateLength-1)*4
        self.actionSpace = 2
        self.percentageCosts = [0, 0.1, 0.2]
        self.transactionCosts = self.percentageCosts[1]/100

    def run(self):
        strategy = 'temDQN'
        trainingParameters = [self.numberOfEpisodes] 
        trainingEnv = TradingEnv(self.data, self.money, self.stateLength, self.transactionCosts)
        print(trainingEnv)
        strategyModule = importlib.import_module(str(strategy))
        className = getattr(strategyModule, strategy)
        tradingStrategy = className(self.observationSpace, self.actionSpace)
        trainingEnv = tradingStrategy.training(trainingEnv, trainingParameters=trainingParameters)
        # fileName = 'tem'
        # tradingStrategy.saveModel(fileName)
        # tradingStrategy.loadModel(fileName)
        