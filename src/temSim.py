import pandas as pd 
import gym
import numpy as np
import importlib, math
import matplotlib.pyplot as plt 


class TradingEnv(gym.Env):
    """
    GOAL: Implement a custom trading environment compatible with OpenAI Gym.
    
    VARIABLES:  - data: Dataframe monitoring the trading activity.
                - state: RL state to be returned to the RL agent.
                - reward: RL reward to be returned to the RL agent.
                - done: RL episode termination signal.
                - t: Current trading time step.
                - marketSymbol: Stock market symbol.
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - stateLength: Number of trading time steps included in the state.
                - numberOfShares: Number of shares currently owned by the agent.
                - transactionCosts: Transaction costs associated with the trading
                                    activity (e.g. 0.01 is 1% of loss).
                                
    METHODS:    - __init__: Object constructor initializing the trading environment.
                - reset: Perform a soft reset of the trading environment.
                - step: Transition to the next trading time step.
                - render: Illustrate graphically the trading environment.
    """

    def __init__(self, data, marketSymbol, startingDate, endingDate, money, stateLength=30,
                 transactionCosts=0, startingPoint=0):
        self.data = data.copy()
        # Interpolate in case of missing data
        self.data.replace(0.0, np.nan, inplace=True)
        # self.data.interpolate(method='linear', limit=5, limit_area='inside', inplace=True)
        # self.data.fillna(method='ffill', inplace=True)
        # self.data.fillna(method='bfill', inplace=True)
        # self.data.fillna(0, inplace=True)
        
        # Set the trading activity dataframe
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = float(money)
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        # Set the RL variables common to every OpenAI gym environments
        self.state = [self.data['Close'][0:stateLength].tolist(),
                      self.data['Low'][0:stateLength].tolist(),
                      self.data['High'][0:stateLength].tolist(),
                      self.data['Volume'][0:stateLength].tolist(),
                      [0]]
        self.reward = 0.
        self.done = 0

        # Set additional variables related to the trading activity
        self.marketSymbol = marketSymbol
        self.startingDate = startingDate
        self.endingDate = endingDate
        self.stateLength = stateLength
        self.t = stateLength
        self.numberOfShares = 0
        self.transactionCosts = transactionCosts
        self.epsilon = 0.1

        # If required, set a custom starting point for the trading activity
        if startingPoint:
            self.setStartingPoint(startingPoint)


    def reset(self):
        """
        GOAL: Perform a soft reset of the trading environment. 
        
        INPUTS: /    
        
        OUTPUTS: - state: RL state returned to the trading strategy.
        """

        # Reset the trading activity dataframe
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = self.data['Cash'][0]
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        # Reset the RL variables common to every OpenAI gym environments
        self.state = [self.data['Close'][0:self.stateLength].tolist(),
                      self.data['Low'][0:self.stateLength].tolist(),
                      self.data['High'][0:self.stateLength].tolist(),
                      self.data['Volume'][0:self.stateLength].tolist(),
                      [0]]
        self.reward = 0.
        self.done = 0

        # Reset additional variables related to the trading activity
        self.t = self.stateLength
        self.numberOfShares = 0

        return self.state

    
    def computeLowerBound(self, cash, numberOfShares, price):
        """
        GOAL: Compute the lower bound of the complete RL action space, 
              i.e. the minimum number of share to trade.
        
        INPUTS: - cash: Value of the cash owned by the agent.
                - numberOfShares: Number of shares owned by the agent.
                - price: Last price observed.
        
        OUTPUTS: - lowerBound: Lower bound of the RL action space.
        """

        # Computation of the RL action lower bound
        deltaValues = - cash - numberOfShares * price * (1 + self.epsilon) * (1 + self.transactionCosts)
        if deltaValues < 0:
            lowerBound = deltaValues / (price * (2 * self.transactionCosts + (self.epsilon * (1 + self.transactionCosts))))
        else:
            lowerBound = deltaValues / (price * self.epsilon * (1 + self.transactionCosts))
        return lowerBound
    

    def step(self, action):
        """
        GOAL: Transition to the next trading time step based on the
              trading position decision made (either long or short).
        
        INPUTS: - action: Trading decision (1 = long, 0 = short).    
        
        OUTPUTS: - state: RL state to be returned to the RL agent.
                 - reward: RL reward to be returned to the RL agent.
                 - done: RL episode termination signal (boolean).
                 - info: Additional information returned to the RL agent.
        """

        # Stting of some local variables
        t = self.t
        numberOfShares = self.numberOfShares
        customReward = False

        # CASE 1: LONG POSITION
        if(action == 1):
            self.data['Position'][t] = 1
            # Case a: Long -> Long
            if(self.data['Position'][t - 1] == 1):
                self.data['Cash'][t] = self.data['Cash'][t - 1]
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
            # Case b: No position -> Long
            elif(self.data['Position'][t - 1] == 0):
                self.numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = 1
            # Case c: Short -> Long
            else:
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                self.numberOfShares = math.floor(self.data['Cash'][t]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t] - self.numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = 1

        # CASE 2: SHORT POSITION
        elif(action == 0):
            self.data['Position'][t] = -1
            # Case a: Short -> Short
            if(self.data['Position'][t - 1] == -1):
                lowerBound = self.computeLowerBound(self.data['Cash'][t - 1], -numberOfShares, self.data['Close'][t-1])
                if lowerBound <= 0:
                    self.data['Cash'][t] = self.data['Cash'][t - 1]
                    self.data['Holdings'][t] =  - self.numberOfShares * self.data['Close'][t]
                else:
                    numberOfSharesToBuy = min(math.floor(lowerBound), self.numberOfShares)
                    self.numberOfShares -= numberOfSharesToBuy
                    self.data['Cash'][t] = self.data['Cash'][t - 1] - numberOfSharesToBuy * self.data['Close'][t] * (1 + self.transactionCosts)
                    self.data['Holdings'][t] =  - self.numberOfShares * self.data['Close'][t]
                    customReward = True
            # Case b: No position -> Short
            elif(self.data['Position'][t - 1] == 0):
                self.numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                self.data['Holdings'][t] = - self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = -1
            # Case c: Long -> Short
            else:
                self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                self.numberOfShares = math.floor(self.data['Cash'][t]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t] + self.numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                self.data['Holdings'][t] = - self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = -1

        # CASE 3: PROHIBITED ACTION
        else:
            raise SystemExit("Prohibited action! Action should be either 1 (long) or 0 (short).")

        # Update the total amount of money owned by the agent, as well as the return generated
        self.data['Money'][t] = self.data['Holdings'][t] + self.data['Cash'][t]
        self.data['Returns'][t] = (self.data['Money'][t] - self.data['Money'][t-1])/self.data['Money'][t-1]

        # Set the RL reward returned to the trading agent
        if not customReward:
            self.reward = self.data['Returns'][t]
        else:
            self.reward = (self.data['Close'][t-1] - self.data['Close'][t])/self.data['Close'][t-1]

        # Transition to the next trading time step
        self.t = self.t + 1
        self.state = [self.data['Close'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Low'][self.t - self.stateLength : self.t].tolist(),
                      self.data['High'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Volume'][self.t - self.stateLength : self.t].tolist(),
                      [self.data['Position'][self.t - 1]]]
        if(self.t == self.data.shape[0]):
            self.done = 1  

        # Same reasoning with the other action (exploration trick)
        otherAction = int(not bool(action))
        customReward = False
        if(otherAction == 1):
            otherPosition = 1
            if(self.data['Position'][t - 1] == 1):
                otherCash = self.data['Cash'][t - 1]
                otherHoldings = numberOfShares * self.data['Close'][t]
            elif(self.data['Position'][t - 1] == 0):
                numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = self.data['Cash'][t - 1] - numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                otherHoldings = numberOfShares * self.data['Close'][t]
            else:
                otherCash = self.data['Cash'][t - 1] - numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                numberOfShares = math.floor(otherCash/(self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = otherCash - numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                otherHoldings = numberOfShares * self.data['Close'][t]
        else:
            otherPosition = -1
            if(self.data['Position'][t - 1] == -1):
                lowerBound = self.computeLowerBound(self.data['Cash'][t - 1], -numberOfShares, self.data['Close'][t-1])
                if lowerBound <= 0:
                    otherCash = self.data['Cash'][t - 1]
                    otherHoldings =  - numberOfShares * self.data['Close'][t]
                else:
                    numberOfSharesToBuy = min(math.floor(lowerBound), numberOfShares)
                    numberOfShares -= numberOfSharesToBuy
                    otherCash = self.data['Cash'][t - 1] - numberOfSharesToBuy * self.data['Close'][t] * (1 + self.transactionCosts)
                    otherHoldings =  - numberOfShares * self.data['Close'][t]
                    customReward = True
            elif(self.data['Position'][t - 1] == 0):
                numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = self.data['Cash'][t - 1] + numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                otherHoldings = - numberOfShares * self.data['Close'][t]
            else:
                otherCash = self.data['Cash'][t - 1] + numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                numberOfShares = math.floor(otherCash/(self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = otherCash + numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                otherHoldings = - self.numberOfShares * self.data['Close'][t]
        otherMoney = otherHoldings + otherCash
        if not customReward:
            otherReward = (otherMoney - self.data['Money'][t-1])/self.data['Money'][t-1]
        else:
            otherReward = (self.data['Close'][t-1] - self.data['Close'][t])/self.data['Close'][t-1]
        otherState = [self.data['Close'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Low'][self.t - self.stateLength : self.t].tolist(),
                      self.data['High'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Volume'][self.t - self.stateLength : self.t].tolist(),
                      [otherPosition]]
        self.info = {'State' : otherState, 'Reward' : otherReward, 'Done' : self.done}

        # Return the trading environment feedback to the RL trading agent
        return self.state, self.reward, self.done, self.info


    def render(self):
        """
        GOAL: Illustrate graphically the trading activity, by plotting
              both the evolution of the stock market price and the 
              evolution of the trading capital. All the trading decisions
              (long and short positions) are displayed as well.
        
        INPUTS: /   
        
        OUTPUTS: /
        """

        # Set the Matplotlib figure and subplots
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
        ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)

        # Plot the first graph -> Evolution of the stock market price
        self.data['Close'].plot(ax=ax1, color='blue', lw=2)
        ax1.plot(self.data.loc[self.data['Action'] == 1.0].index, 
                 self.data['Close'][self.data['Action'] == 1.0],
                 '^', markersize=5, color='green')   
        ax1.plot(self.data.loc[self.data['Action'] == -1.0].index, 
                 self.data['Close'][self.data['Action'] == -1.0],
                 'v', markersize=5, color='red')
        
        # Plot the second graph -> Evolution of the trading capital
        self.data['Money'].plot(ax=ax2, color='blue', lw=2)
        ax2.plot(self.data.loc[self.data['Action'] == 1.0].index, 
                 self.data['Money'][self.data['Action'] == 1.0],
                 '^', markersize=5, color='green')   
        ax2.plot(self.data.loc[self.data['Action'] == -1.0].index, 
                 self.data['Money'][self.data['Action'] == -1.0],
                 'v', markersize=5, color='red')
        
        # Generation of the two legends and plotting
        ax1.legend(["Price", "Long",  "Short"])
        ax2.legend(["Capital", "Long", "Short"])
        plt.savefig(''.join(['Figures/', str(self.marketSymbol), '_Rendering', '.png']))
        #plt.show()


    def setStartingPoint(self, startingPoint):
        """
        GOAL: Setting an arbitrary starting point regarding the trading activity.
              This technique is used for better generalization of the RL agent.
        
        INPUTS: - startingPoint: Optional starting point (iteration) of the trading activity.
        
        OUTPUTS: /
        """

        # Setting a custom starting point
        self.t = np.clip(startingPoint, self.stateLength, len(self.data.index))

        # Set the RL variables common to every OpenAI gym environments
        self.state = [self.data['Close'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Low'][self.t - self.stateLength : self.t].tolist(),
                      self.data['High'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Volume'][self.t - self.stateLength : self.t].tolist(),
                      [self.data['Position'][self.t - 1]]]
        if(self.t == self.data.shape[0]):
            self.done = 1

class Simulator:
    def __init__(self, filepath=None):
        self.data = pd.read_csv(filepath)
        # print(self.data)
        self.data['dateTime'] = pd.to_datetime(self.data['dateTime'])
        # print((self.data['dateTime'][0] - self.data['dateTime'][9]).days)
        self.startingDate = self.data['dateTime'][0]
        self.splitingDate = '2023-01-01'
        self.endingDate = self.data['dateTime'][len(self.data)-1]
        self.money = 100000
        # self.numberOfEpisodes = 50
        self.numberOfEpisodes = 100
        self.stateLength = 30
        self.observationSpace = 1 + (self.stateLength-1)*4
        self.actionSpace = 2
        self.percentageCosts = [0, 0.1, 0.2]
        self.transactionCosts = self.percentageCosts[0]/100

    def run(self):
        strategy = 'TDQN'
        trainingParameters = [self.numberOfEpisodes] 
        stock = 'APPL'
        trainingEnv = TradingEnv(self.data, stock, self.startingDate, self.splitingDate, self.money, self.stateLength, self.transactionCosts)
        print(trainingEnv)
        strategyModule = importlib.import_module(str(strategy))
        className = getattr(strategyModule, strategy)
        tradingStrategy = className(self.observationSpace, self.actionSpace)
        trainingEnv = tradingStrategy.training(trainingEnv, trainingParameters=trainingParameters,verbose=True, plotTraining=True)
        fileName = 'tem'
        tradingStrategy.saveModel(fileName)
        tradingStrategy.loadModel(fileName)
        testingEnv = TradingEnv(self.data, stock, self.splitingDate, self.endingDate, self.money, self.stateLength, self.transactionCosts)
        testingEnv = tradingStrategy.testing(trainingEnv, testingEnv, rendering=False, showPerformance=True)
        print('training')
        print(trainingEnv.data.columns,len(trainingEnv.data))
        print('testing')
        print(testingEnv.data.columns,len(testingEnv.data))
        # self.plotEntireTrading(trainingEnv, testingEnv)
        # print(trainingEnv.data['Action'])
        print(np.unique(trainingEnv.data['Action']))
        # print(testingEnv.data['Action'])
        print(np.unique(testingEnv.data['Action']))
        plt.subplot(121)
        plt.plot(trainingEnv.data['dateTime'],trainingEnv.data['Action'])
        plt.subplot(122)
        plt.plot(testingEnv.data['dateTime'],testingEnv.data['Action'])
        trainingEnv.data.to_csv('trainingEnv.csv')
        testingEnv.data.to_csv('testingEnv.csv')
        plt.savefig('x.png')

    # def plotEntireTrading(self, trainingEnv, testingEnv):
    #     # print(type(trainingEnv.data))
    #     # print(type(testingEnv.data))
    #     ratio = trainingEnv.data['Money'][len(trainingEnv.data)-1]/testingEnv.data['Money'][0]
    #     testingEnv.data['Money'] = ratio * testingEnv.data['Money']

    #     # Concatenation of the training and testing trading dataframes
    #     dataframes = [trainingEnv.data, testingEnv.data]
    #     data = pd.concat(dataframes)

    #     # Set the Matplotlib figure and subplots
    #     fig = plt.figure(figsize=(10, 8))
    #     ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
    #     ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)

    #     # Plot the first graph -> Evolution of the stock market price
    #     trainingEnv.data['Close'].plot(ax=ax1, color='blue', lw=2)
    #     testingEnv.data['Close'].plot(ax=ax1, color='blue', lw=2, label='_nolegend_') 
    #     ax1.plot(data.loc[data['Action'] == 1.0].index, 
    #              data['Close'][data['Action'] == 1.0],
    #              '^', markersize=5, color='green')   
    #     ax1.plot(data.loc[data['Action'] == -1.0].index, 
    #              data['Close'][data['Action'] == -1.0],
    #              'v', markersize=5, color='red')
        
    #     # Plot the second graph -> Evolution of the trading capital
    #     trainingEnv.data['Money'].plot(ax=ax2, color='blue', lw=2)
    #     testingEnv.data['Money'].plot(ax=ax2, color='blue', lw=2, label='_nolegend_') 
    #     ax2.plot(data.loc[data['Action'] == 1.0].index, 
    #              data['Money'][data['Action'] == 1.0],
    #              '^', markersize=5, color='green')   
    #     ax2.plot(data.loc[data['Action'] == -1.0].index, 
    #              data['Money'][data['Action'] == -1.0],
    #              'v', markersize=5, color='red')

    #     # Plot the vertical line seperating the training and testing datasets
    #     ax1.axvline(pd.Timestamp(self.splitingDate), color='black', linewidth=2.0)
    #     ax2.axvline(pd.Timestamp(self.splitingDate), color='black', linewidth=2.0)
        
    #     # Generation of the two legends and plotting
    #     ax1.legend(["Price", "Long",  "Short", "Train/Test separation"])
    #     ax2.legend(["Capital", "Long", "Short", "Train/Test separation"])
    #     plt.savefig(''.join(['Figures/', str(trainingEnv.marketSymbol), '_TrainingTestingRendering', '.png'])) 
