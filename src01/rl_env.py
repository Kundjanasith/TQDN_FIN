import gym
import numpy as np 
import math

class TradingEnv(gym.Env):
    def __init__(self, data, money, stateLength=30, transactionCosts=0):
        self.data = data.copy()
        self.data.replace(0.0, np.nan, inplace=True)
        self.data.interpolate(method='linear', limit=5, limit_area='inside', inplace=True)
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)
        self.data.fillna(0, inplace=True)
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = float(money)
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.
        self.state = [self.data['Close'][0:stateLength].tolist(),
                      self.data['Low'][0:stateLength].tolist(),
                      self.data['High'][0:stateLength].tolist(),
                      self.data['Volume'][0:stateLength].tolist(),
                      [0]]
        self.reward = 0.
        self.done = 0
        self.stateLength = stateLength
        self.t = stateLength
        self.numberOfShares = 0
        self.transactionCosts = transactionCosts
        self.epsilon = 0.1


    def reset(self):
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = self.data['Cash'][0]
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.
        self.state = [self.data['Close'][0:self.stateLength].tolist(),
                      self.data['Low'][0:self.stateLength].tolist(),
                      self.data['High'][0:self.stateLength].tolist(),
                      self.data['Volume'][0:self.stateLength].tolist(),
                      [0]]
        self.reward = 0.
        self.done = 0
        self.t = self.stateLength
        self.numberOfShares = 0

        return self.state

    
    def computeLowerBound(self, cash, numberOfShares, price):
        deltaValues = - cash - numberOfShares * price * (1 + self.epsilon) * (1 + self.transactionCosts)
        if deltaValues < 0:
            lowerBound = deltaValues / (price * (2 * self.transactionCosts + (self.epsilon * (1 + self.transactionCosts))))
        else:
            lowerBound = deltaValues / (price * self.epsilon * (1 + self.transactionCosts))
        return lowerBound
    

    def step(self, action):
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
        return self.state, self.reward, self.done, self.info

    def setStartingPoint(self, startingPoint):
        self.t = np.clip(startingPoint, self.stateLength, len(self.data.index))
        self.state = [self.data['Close'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Low'][self.t - self.stateLength : self.t].tolist(),
                      self.data['High'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Volume'][self.t - self.stateLength : self.t].tolist(),
                      [self.data['Position'][self.t - 1]]]
        if(self.t == self.data.shape[0]):
            self.done = 1