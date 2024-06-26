from collections import deque
from tqdm import tqdm
from rl_data import DataAugmentation
from rl_perf import PerformanceEstimator
import random
import sys, os
capacity = 100000
filterOrder = 5
class ReplayMemory:
    def __init__(self, capacity=capacity):
        self.memory = deque(maxlen=capacity)
    def push(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))
    def sample(self, batchSize):
        state, action, reward, nextState, done = zip(*random.sample(self.memory, batchSize))
        return state, action, reward, nextState, done
    def __len__(self):
        return len(self.memory)
    def reset(self):
        self.memory = deque(maxlen=capacity)

import tensorflow as tf
numberOfNeurons = 512
dropout = 0.2
class DQN(tf.keras.Model):
    def __init__(self, numberOfInputs, numberOfOutputs, numberOfNeurons=64, dropout=0.5):
        super(DQN, self).__init__()
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input(shape=(numberOfInputs,)))
        self.model.add(tf.keras.layers.Dense(numberOfNeurons, activation='linear'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())
        self.model.add(tf.keras.layers.Dropout(dropout))
        self.model.add(tf.keras.layers.Dense(numberOfNeurons, activation='linear'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())
        self.model.add(tf.keras.layers.Dropout(dropout))
        self.model.add(tf.keras.layers.Dense(numberOfNeurons, activation='linear'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())
        self.model.add(tf.keras.layers.Dropout(dropout))
        self.model.add(tf.keras.layers.Dense(numberOfNeurons, activation='linear'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())
        self.model.add(tf.keras.layers.Dropout(dropout))
        self.model.add(tf.keras.layers.Dense(numberOfOutputs, activation='linear'))
        
    def get_model(self):
        return self.model
    
import numpy as np 
import math
gamma = 0.4
learningRate = 0.0001
targetNetworkUpdate = 1000
learningUpdatePeriod = 1
epsilonStart = 1.0
epsilonEnd = 0.01
epsilonDecay = 10000
GPUNumber = 1
batchSize = 32
L2Factor = 0.000001
rewardClipping = 1
class temDQN:
    def __init__(self, observationSpace, actionSpace, numberOfNeurons=numberOfNeurons, dropout=dropout, 
                 gamma=gamma, learningRate=learningRate, targetNetworkUpdate=targetNetworkUpdate,
                 epsilonStart=epsilonStart, epsilonEnd=epsilonEnd, epsilonDecay=epsilonDecay,
                 capacity=capacity, batchSize=batchSize):
        random.seed(0)
        self.device = '/gpu:'+str(GPUNumber) if tf.config.list_physical_devices('GPU') else '/cpu:0'
        self.gamma = gamma
        self.learningRate = learningRate
        self.targetNetworkUpdate = targetNetworkUpdate
        self.capacity = capacity
        self.batchSize = batchSize
        self.replayMemory = ReplayMemory(capacity)
        self.observationSpace = observationSpace
        self.actionSpace = actionSpace
        self.policyNetwork = DQN(observationSpace, actionSpace, numberOfNeurons, dropout).get_model()
        self.targetNetwork = DQN(observationSpace, actionSpace, numberOfNeurons, dropout).get_model()
        self.targetNetwork.set_weights(self.policyNetwork.get_weights())
        self.optimizer = tf.keras.optimizers.legacy.Adam(learningRate, decay=L2Factor)
        self.epsilonValue = lambda iteration: epsilonEnd + (epsilonStart - epsilonEnd) * math.exp(-1 * iteration / epsilonDecay)
        self.iterations = 0
    
    def getNormalizationCoefficients(self, tradingEnv):
        tradingData = tradingEnv.data
        closePrices = tradingData['Close'].tolist()
        lowPrices = tradingData['Low'].tolist()
        highPrices = tradingData['High'].tolist()
        volumes = tradingData['Volume'].tolist()
        coefficients = []
        margin = 1
        # 1. Close price => returns (absolute) => maximum value (absolute)
        returns = [abs((closePrices[i]-closePrices[i-1])/closePrices[i-1]) for i in range(1, len(closePrices))]
        coeffs = (0, np.max(returns)*margin)
        coefficients.append(coeffs)
        # 2. Low/High prices => Delta prices => maximum value
        deltaPrice = [abs(highPrices[i]-lowPrices[i]) for i in range(len(lowPrices))]
        coeffs = (0, np.max(deltaPrice)*margin)
        coefficients.append(coeffs)
        # 3. Close/Low/High prices => Close price position => no normalization required
        coeffs = (0, 1)
        coefficients.append(coeffs)
        # 4. Volumes => minimum and maximum values
        coeffs = (np.min(volumes)/margin, np.max(volumes)*margin)
        coefficients.append(coeffs)
        
        return coefficients
    
    def processState(self, state, coefficients):
        # Normalization of the RL state
        closePrices = [state[0][i] for i in range(len(state[0]))]
        lowPrices = [state[1][i] for i in range(len(state[1]))]
        highPrices = [state[2][i] for i in range(len(state[2]))]
        volumes = [state[3][i] for i in range(len(state[3]))]
        # 1. Close price => returns => MinMax normalization
        returns = [(closePrices[i]-closePrices[i-1])/closePrices[i-1] for i in range(1, len(closePrices))]
        if coefficients[0][0] != coefficients[0][1]:
            state[0] = [((x - coefficients[0][0])/(coefficients[0][1] - coefficients[0][0])) for x in returns]
        else:
            state[0] = [0 for x in returns]
        # 2. Low/High prices => Delta prices => MinMax normalization
        deltaPrice = [abs(highPrices[i]-lowPrices[i]) for i in range(1, len(lowPrices))]
        if coefficients[1][0] != coefficients[1][1]:
            state[1] = [((x - coefficients[1][0])/(coefficients[1][1] - coefficients[1][0])) for x in deltaPrice]
        else:
            state[1] = [0 for x in deltaPrice]
        # 3. Close/Low/High prices => Close price position => No normalization required
        closePricePosition = []
        for i in range(1, len(closePrices)):
            deltaPrice = abs(highPrices[i]-lowPrices[i])
            if deltaPrice != 0:
                item = abs(closePrices[i]-lowPrices[i])/deltaPrice
            else:
                item = 0.5
            closePricePosition.append(item)
        if coefficients[2][0] != coefficients[2][1]:
            state[2] = [((x - coefficients[2][0])/(coefficients[2][1] - coefficients[2][0])) for x in closePricePosition]
        else:
            state[2] = [0.5 for x in closePricePosition]
        # 4. Volumes => MinMax normalization
        volumes = [volumes[i] for i in range(1, len(volumes))]
        if coefficients[3][0] != coefficients[3][1]:
            state[3] = [((x - coefficients[3][0])/(coefficients[3][1] - coefficients[3][0])) for x in volumes]
        else:
            state[3] = [0 for x in volumes]
        
        # Process the state structure to obtain the appropriate format
        state = [item for sublist in state for item in sublist]

        return state

    
    def processReward(self, reward):
        return np.clip(reward, -rewardClipping, rewardClipping)
 

    def updateTargetNetwork(self):
        # Check if an update is required (update frequency)
        if(self.iterations % targetNetworkUpdate == 0):
            # Transfer the DNN parameters (policy network -> target network)
            self.targetNetwork.set_weights(self.policyNetwork.get_weights())


    def chooseAction(self, state):
        # Choose the best action based on the RL policy
        tensorState = tf.convert_to_tensor(state, dtype=tf.float32)
        tensorState = tf.expand_dims(tensorState, 0)
        QValues = self.policyNetwork(tensorState)
        action = tf.argmax(QValues, axis=1).numpy()[0]
        Q = tf.reduce_max(QValues).numpy()
        QValues = QValues.numpy()
        return action, Q, QValues

    
    def chooseActionEpsilonGreedy(self, state, previousAction):
        alpha = 0.1
        # EXPLOITATION -> RL policy
        if random.random() > self.epsilonValue(self.iterations):
            # Sticky action (RL generalization mechanism)
            if random.random() > alpha:
                action, Q, QValues = self.chooseAction(state)
            else:
                action = previousAction
                Q = 0
                QValues = [0, 0]
        # EXPLORATION -> Random
        else:
            action = np.random.randint(self.actionSpace)
            Q = 0
            QValues = [0, 0]
        # Increment the iterations counter (for Epsilon Greedy)
        self.iterations += 1
        return action, Q, QValues
    

    def learning(self, batchSize=batchSize):
        gradientClipping = 1
        # Check that the replay memory is filled enough
        if len(self.replayMemory) >= batchSize:
            # Sample a batch of experiences from the replay memory
            state, action, reward, nextState, done = self.replayMemory.sample(batchSize)
            # Compute the current Q values returned by the policy network
            with tf.GradientTape() as tape:
                current_policy_output = self.policyNetwork(np.array(state))
                currentQValues = tf.squeeze(tf.gather(current_policy_output, action, axis=1))
                # Compute the next Q values returned by the target network
                next_policy_output = self.policyNetwork(np.array(nextState))
                nextActions = tf.argmax(next_policy_output, axis=1)
                # next_action_indices = tf.expand_dims(nextActions, axis=1)
                nextQValues = tf.squeeze(tf.gather(self.targetNetwork(np.array(nextState)), nextActions, axis=1))
                done = np.mean(done)
                expectedQValues = reward + gamma * nextQValues * (1 - done)
                # Compute the Huber loss
                loss = tf.keras.losses.Huber()(expectedQValues, currentQValues)
            # Computation of the gradients
            gradients = tape.gradient(loss, self.policyNetwork.trainable_variables)
            # Gradient Clipping
            gradients, _ = tf.clip_by_global_norm(gradients, gradientClipping)
            # Perform the Deep Neural Network optimization
            self.optimizer.apply_gradients(zip(gradients, self.policyNetwork.trainable_variables))
            # If required, update the target deep neural network (update frequency)
            self.updateTargetNetwork()

    def training(self, trainingEnv, trainingParameters=[], rewards_log_path=None):
        # Apply data augmentation techniques to improve the training set
        dataAugmentation = DataAugmentation()
        trainingEnvList = dataAugmentation.generate(trainingEnv)
        totalReward = 0
        os.system('touch %s'%rewards_log_path)
        
        try:
            # If required, print the training progression
            print("Training progression (hardware selected => " + str(self.device) + "):")
            # Training phase for the number of episodes specified as parameter
            for episode in tqdm(range(trainingParameters[0])):
                # For each episode, train on the entire set of training environments
                for i in range(len(trainingEnvList)):
                    # Set the initial RL variables
                    coefficients = self.getNormalizationCoefficients(trainingEnvList[i])
                    trainingEnvList[i].reset()
                    startingPoint = random.randrange(len(trainingEnvList[i].data.index))
                    trainingEnvList[i].setStartingPoint(startingPoint)
                    state = self.processState(trainingEnvList[i].state, coefficients)
                    previousAction = 0
                    done = 0
                    stepsCounter = 0
                    # Interact with the training environment until termination
                    while done == 0:
                        # print(len(state))
                        # Choose an action according to the RL policy and the current RL state
                        action, _, _ = self.chooseActionEpsilonGreedy(state, previousAction)
                        
                        # Interact with the environment with the chosen action
                        nextState, reward, done, info = trainingEnvList[i].step(action)
                        # Process the RL variables retrieved and insert this new experience into the Experience Replay memory
                        reward = self.processReward(reward)
                        nextState = self.processState(nextState, coefficients)
                        self.replayMemory.push(state, action, reward, nextState, done)
                        # Trick for better exploration
                        otherAction = int(not bool(action))
                        otherReward = self.processReward(info['Reward'])
                        otherNextState = self.processState(info['State'], coefficients)
                        otherDone = info['Done']
                        self.replayMemory.push(state, otherAction, otherReward, otherNextState, otherDone)
                        # Execute the DQN learning procedure
                        stepsCounter += 1
                        if stepsCounter == learningUpdatePeriod:
                            self.learning()
                            stepsCounter = 0
                        # Update the RL state
                        state = nextState
                        previousAction = action
                        totalReward += reward
                file_o = open(rewards_log_path,'a')
                file_o.write('%d,%f\n'%(episode,totalReward))
                file_o.close()

                self.policyNetwork.save('policy_models/models%d.h5'%episode) 
                self.targetNetwork.save('target_models/models%d.h5'%episode) 
        
        except KeyboardInterrupt:
            print()
            print("WARNING: Training prematurely interrupted...")
            print()
            # self.policyNetwork.eval()

        # Assess the algorithm performance on the training trading environment
        trainingEnv = self.testing(trainingEnv)

        # If required, print the strategy performance in a table
        # if showPerformance:
        #     analyser = PerformanceEstimator(trainingEnv.data)
        #     analyser.displayPerformance('TDQN')
        
        # Closing of the tensorboard writer
        # self.writer.close()
        
        return trainingEnv
    
    def testing(self, testingEnv):
        # Apply data augmentation techniques to process the testing set

        dataAugmentation = DataAugmentation()
        testingEnvSmoothed = dataAugmentation.lowPassFilter(testingEnv, filterOrder)
        # trainingEnv = dataAugmentation.lowPassFilter(trainingEnv, filterOrder)

        # Initialization of some RL variables
        coefficients = self.getNormalizationCoefficients(testingEnv)
        state = self.processState(testingEnvSmoothed.reset(), coefficients)
        testingEnv.reset()
        QValues0 = []
        QValues1 = []
        done = 0

        # Interact with the environment until the episode termination
        while done == 0:

            # Choose an action according to the RL policy and the current RL state
            action, _, QValues = self.chooseAction(state)
                
            # Interact with the environment with the chosen action
            nextState, _, done, _ = testingEnvSmoothed.step(action)
            testingEnv.step(action)
                
            # Update the new state
            state = self.processState(nextState, coefficients)

            # Storing of the Q values
            QValues = QValues[0]
            QValues0.append(QValues[0])
            QValues1.append(QValues[1])
        
        return testingEnv
    
    def predict(self, testingEnv, policy_path, target_path):
        self.policyNetwork.load_weights(policy_path)
        self.targetNetwork.load_weights(target_path)
        # Apply data augmentation techniques to process the testing set
        dataAugmentation = DataAugmentation()
        testingEnvSmoothed = dataAugmentation.lowPassFilter(testingEnv, filterOrder)
        # trainingEnv = dataAugmentation.lowPassFilter(trainingEnv, filterOrder)

        # Initialization of some RL variables
        coefficients = self.getNormalizationCoefficients(testingEnv)
        state = self.processState(testingEnvSmoothed.reset(), coefficients)
        testingEnv.reset()
        QValues0 = []
        QValues1 = []
        done = 0

        # Interact with the environment until the episode termination
        while done == 0:

            # Choose an action according to the RL policy and the current RL state
            action, _, QValues = self.chooseAction(state)
                
            # Interact with the environment with the chosen action
            nextState, _, done, _ = testingEnvSmoothed.step(action)
            testingEnv.step(action)
                
            # Update the new state
            state = self.processState(nextState, coefficients)

            # Storing of the Q values
            QValues = QValues[0]
            QValues0.append(QValues[0])
            QValues1.append(QValues[1])
        
        return testingEnv