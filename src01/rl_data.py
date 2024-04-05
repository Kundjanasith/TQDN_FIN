import copy
import numpy as np

shiftRange = [0]
stretchRange = [1]
filterRange = [5]
noiseRange = [0]

class DataAugmentation:
    
    def shiftTimeSeries(self, tradingEnv, shiftMagnitude=0):
        newTradingEnv = copy.deepcopy(tradingEnv)
        if shiftMagnitude < 0:
            minValue = np.min(tradingEnv.data['Volume'])
            shiftMagnitude = max(-minValue, shiftMagnitude)
        newTradingEnv.data['Volume'] += shiftMagnitude
        return newTradingEnv


    def streching(self, tradingEnv, factor=1):
        newTradingEnv = copy.deepcopy(tradingEnv)
        returns = newTradingEnv.data['Close'].pct_change() * factor
        for i in range(1, len(newTradingEnv.data.index)):
            newTradingEnv.data['Close'][i] = newTradingEnv.data['Close'][i-1] * (1 + returns[i])
            newTradingEnv.data['Low'][i] = newTradingEnv.data['Close'][i] * tradingEnv.data['Low'][i]/tradingEnv.data['Close'][i]
            newTradingEnv.data['High'][i] = newTradingEnv.data['Close'][i] * tradingEnv.data['High'][i]/tradingEnv.data['Close'][i]
            newTradingEnv.data['Open'][i] = newTradingEnv.data['Close'][i-1]
        return newTradingEnv


    def noiseAddition(self, tradingEnv, stdev=1):
        newTradingEnv = copy.deepcopy(tradingEnv)
        for i in range(1, len(newTradingEnv.data.index)):
            price = newTradingEnv.data['Close'][i]
            volume = newTradingEnv.data['Volume'][i]
            priceNoise = np.random.normal(0, stdev*(price/100))
            volumeNoise = np.random.normal(0, stdev*(volume/100))
            newTradingEnv.data['Close'][i] *= (1 + priceNoise/100)
            newTradingEnv.data['Low'][i] *= (1 + priceNoise/100)
            newTradingEnv.data['High'][i] *= (1 + priceNoise/100)
            newTradingEnv.data['Volume'][i] *= (1 + volumeNoise/100)
            newTradingEnv.data['Open'][i] = newTradingEnv.data['Close'][i-1]
        return newTradingEnv


    def lowPassFilter(self, tradingEnv, order=5):
        newTradingEnv = copy.deepcopy(tradingEnv)
        newTradingEnv.data['Close'] = newTradingEnv.data['Close'].rolling(window=order).mean()
        newTradingEnv.data['Low'] = newTradingEnv.data['Low'].rolling(window=order).mean()
        newTradingEnv.data['High'] = newTradingEnv.data['High'].rolling(window=order).mean()
        newTradingEnv.data['Volume'] = newTradingEnv.data['Volume'].rolling(window=order).mean()
        for i in range(order):
            newTradingEnv.data['Close'][i] = tradingEnv.data['Close'][i]
            newTradingEnv.data['Low'][i] = tradingEnv.data['Low'][i]
            newTradingEnv.data['High'][i] = tradingEnv.data['High'][i]
            newTradingEnv.data['Volume'][i] = tradingEnv.data['Volume'][i]
        newTradingEnv.data['Open'] = newTradingEnv.data['Close'].shift(1)
        newTradingEnv.data['Open'][0] = tradingEnv.data['Open'][0]
        return newTradingEnv

    def generate(self, tradingEnv):
        tradingEnvList = []
        for shift in shiftRange:
            tradingEnvShifted = self.shiftTimeSeries(tradingEnv, shift)
            for stretch in stretchRange:
                tradingEnvStretched = self.streching(tradingEnvShifted, stretch)
                for order in filterRange:
                    tradingEnvFiltered = self.lowPassFilter(tradingEnvStretched, order)
                    for noise in noiseRange:
                        tradingEnvList.append(self.noiseAddition(tradingEnvFiltered, noise))
        return tradingEnvList
    