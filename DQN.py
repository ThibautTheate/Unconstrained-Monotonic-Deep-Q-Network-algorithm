# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import os
import math
import random
import copy
import datetime
import json

import numpy as np
import pandas as pd

from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# pylint: disable=E1101
# pylint: disable=E1102

from torch.utils.tensorboard import SummaryWriter

from replayMemory import ReplayMemory

from Models.FeedforwardDNN import FeedForwardDNN
from Models.DNN_Atari import DNN_Atari



###############################################################################
#################################### Class DQN ################################
###############################################################################

class DQN:
    """
    GOAL: Implementing the DQN Deep Reinforcement Learning algorithm.
    
    VARIABLES: - device: Hardware specification (CPU or GPU).
               - gamma: Discount factor of the RL algorithm.
               - learningRate: Learning rate of the DL optimizer (ADAM).
               - epsilon: Epsilon value for the DL optimizer (ADAM).
               - targetNetworkUpdate: Update frequency of the target network.
               - learningUpdatePeriod: Frequency of the learning procedure.
               - batchSize: Size of the batch to sample from the replay memory.
               - capacity: Capacity of the replay memory.
               - replayMemory: Experience Replay memory.
               - rewardClipping: Clipping of the RL rewards.
               - gradientClipping: Clipping of the training loss.
               - optimizer: DL optimizer (ADAM).
               - epsilonStart: Initial value of epsilon (Epsilon-Greedy).
               - epsilonEnd: Final value of epsilon (Epsilon-Greedy).
               - epsilonDecay: Exponential decay of epsilon (Epsilon-Greedy).
               - epsilonTest: Test value of epsilon (Epsilon-Greedy).
               - epsilonValue: Current value of epsilon (Epsilon-Greedy).
               - policyNetwork: Deep Neural Network representing the info used by the RL policy.
               - targetNetwork: Deep Neural Network representing the target network.
               - iterations: Counter of the number of iterations.
                                
    METHODS: - __init__: Initialization of the DQN algorithm.
             - readParameters: Read the JSON file to load the parameters.
             - initReporting: Initialize the reporting tools.
             - processState: Process the RL state returned by the environment.
             - processReward: Process the RL reward returned by the environment.
             - updateTargetNetwork: Update the target network (parameters transfer).
             - chooseAction: Choose a valid action based on the current state
                             observed, according to the RL policy learned.
             - chooseActionEpsilonGreedy: Choose a valid action based on the
                                          current state observed, according to
                                          the RL policy learned, following the 
                                          Epsilon Greedy exploration mechanism.
             - fillReplayMemory: Fill the replay memory with random experiences before
                                 the training procedure begins.
             - learning: Execute the DQN learning procedure.
             - training: Train the DQN agent by interacting with the environment.
             - testing: Test the DQN agent learned policy on the RL environment.
             - plotExpectedPerformance: Plot the expected performance of the DQN algorithm.
             - saveModel: Save the RL policy learned.
             - loadModel: Load a RL policy.
             - plotEpsilonAnnealing: Plot the annealing behaviour of the Epsilon
                                     (Epsilon-Greedy exploration technique).
    """

    def __init__(self, observationSpace, actionSpace, environment,
                 parametersFileName='', reporting=True):
        """
        GOAL: Initializing the RL agent based on the DQN Deep Reinforcement Learning
              algorithm, by setting up the algorithm parameters as well as 
              the Deep Neural Networks.
        
        INPUTS: - observationSpace: RL observation space.
                - actionSpace: RL action space.
                - environment: Name of the RL environment.
                - parametersFileName: Name of the JSON parameters file.
                - reporting: Enable the reporting of the results.
        
        OUTPUTS: /
        """

        # Initialize the random function with a fixed random seed
        random.seed(0)

        # Setting of the parameters
        if parametersFileName == '':
            parametersFileName = ''.join(['Parameters/parameters_DQN_', str(environment), '.json'])
        parameters = self.readParameters(parametersFileName)

        # Set the device for DNN computations (CPU or GPU)
        self.device = torch.device('cuda:'+str(parameters['GPUNumber']) if torch.cuda.is_available() else 'cpu')

        # Set the general parameters of the DQN algorithm
        self.gamma = parameters['gamma']
        self.learningRate = parameters['learningRate']
        self.epsilon = parameters['epsilon']
        self.targetNetworkUpdate = parameters['targetNetworkUpdate']
        self.learningUpdatePeriod = parameters['learningUpdatePeriod']
        self.rewardClipping = parameters['rewardClipping']
        self.gradientClipping = parameters['gradientClipping']

        # Set the Experience Replay mechanism
        self.batchSize = parameters['batchSize']
        self.capacity = parameters['capacity']
        self.replayMemory = ReplayMemory(self.capacity)

        # Set both the observation and action spaces
        self.observationSpace = observationSpace
        self.actionSpace = actionSpace

        # Set the two Deep Neural Networks of the DQN algorithm (policy and target)
        self.atari = parameters['atari']
        if self.atari:
            self.policyNetwork = DNN_Atari(observationSpace, actionSpace).to(self.device)
            self.targetNetwork = DNN_Atari(observationSpace, actionSpace).to(self.device)
        else:
            self.policyNetwork = FeedForwardDNN(observationSpace, actionSpace, parameters['structureDNN']).to(self.device)
            self.targetNetwork = FeedForwardDNN(observationSpace, actionSpace, parameters['structureDNN']).to(self.device)
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())

        # Set the Deep Learning optimizer
        self.optimizer = optim.Adam(self.policyNetwork.parameters(), lr=self.learningRate, eps=self.epsilon)

        # Set the Epsilon-Greedy exploration technique
        self.epsilonStart = parameters['epsilonStart']
        self.epsilonEnd = parameters['epsilonEnd']
        self.epsilonDecay = parameters['epsilonDecay']
        self.epsilonTest = parameters['epsilonTest']
        self.epsilonValue = lambda iteration: self.epsilonEnd + (self.epsilonStart - self.epsilonEnd) * math.exp(-1 * iteration / self.epsilonDecay)
        
        # Initialization of the counter for the number of steps
        self.steps = 0

        # Initialization of the experiment folder and tensorboard writer
        if reporting:
            self.initReporting(parameters, 'DQN')


    def readParameters(self, fileName):
        """
        GOAL: Read the appropriate JSON file to load the parameters.
        
        INPUTS: - fileName: Name of the JSON file to read.
        
        OUTPUTS: - parametersDict: Dictionary containing the parameters.
        """

        # Reading of the parameters file, and conversion to Python disctionary
        with open(fileName) as parametersFile:
            parametersDict = json.load(parametersFile)
        return parametersDict

    
    def initReporting(self, parameters, algorithm='DQN'):
        """
        GOAL: Initialize both the experiment folder and the tensorboard
              writer for reporting (and storing) the results.
        
        INPUTS: - parameters: Parameters to ne stored in the experiment folder.
                - algorithm: Name of the RL algorithm.
        
        OUTPUTS: /
        """

        while True:
            try:
                time = datetime.datetime.now().strftime("%d_%m_%Y-%H:%M:%S")
                self.experimentFolder = ''.join(['Experiments/', algorithm, '_', time, '/'])
                os.mkdir(self.experimentFolder)
                with open(''.join([self.experimentFolder , 'Parameters.json']), "w") as f:  
                    json.dump(parameters, f, indent=4)
                self.writer = SummaryWriter(''.join(['Tensorboard/', algorithm, '_', time]))
                break
            except:
                pass
    
    
    def processState(self, state):
        """
        GOAL: Potentially process the RL state returned by the environment.
        
        INPUTS: - state: RL state returned by the environment.
        
        OUTPUTS: - state: RL state processed.
        """

        return state

    
    def processReward(self, reward):
        """
        GOAL: Potentially process the RL reward returned by the environment.
        
        INPUTS: - reward: RL reward returned by the environment.
        
        OUTPUTS: - reward: RL reward processed.
        """

        return np.clip(reward, -self.rewardClipping, self.rewardClipping)
 

    def updateTargetNetwork(self):
        """
        GOAL: Taking into account the update frequency (parameter), update the
              target Deep Neural Network by copying the policy Deep Neural Network
              parameters (weights, bias, etc.).
        
        INPUTS: /
        
        OUTPUTS: /
        """

        # Check if an update is required (update frequency)
        if(self.steps % self.targetNetworkUpdate == 0):
            # Transfer the DNN parameters (policy network -> target network)
            self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())
        

    def chooseAction(self, state, plot=False):
        """
        GOAL: Choose a valid RL action from the action space according to the
              RL policy as well as the current RL state observed.
        
        INPUTS: - state: RL state returned by the environment.
                - plot: Enable the plotting of information about the decision.
        
        OUTPUTS: - action: RL action chosen from the action space.
        """

        # Choose the best action based on the RL policy
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            QValues = self.policyNetwork(state).squeeze(0)
            _, action = QValues.max(0)
            return action.item()

    
    def chooseActionEpsilonGreedy(self, state, epsilon):
        """
        GOAL: Choose a valid RL action from the action space according to the
              RL policy as well as the current RL state observed, following the 
              Epsilon Greedy exploration mechanism.
        
        INPUTS: - state: RL state returned by the environment.
                - epsilon: Epsilon value from Epsilon Greedy technique.
        
        OUTPUTS: - action: RL action chosen from the action space.
        """

        # EXPLOITATION -> RL policy
        if(random.random() > epsilon):
            action = self.chooseAction(state)
        # EXPLORATION -> Random
        else:
            action = random.randrange(self.actionSpace)

        return action


    def fillReplayMemory(self, trainingEnv):
        """
        GOAL: Fill the experiences replay memory with random experiences before the
              the training procedure begins.
        
        INPUTS: - trainingEnv: Training RL environment.
                
        OUTPUTS: /
        """

        # Fill the replay memory with random RL experiences
        while self.replayMemory.__len__() < self.capacity:

            # Set the initial RL variables
            state = self.processState(trainingEnv.reset())
            done = 0

            # Interact with the training environment until termination
            while done == 0:

                # Choose an action according to the RL policy and the current RL state
                action = random.randrange(self.actionSpace)
                
                # Interact with the environment with the chosen action
                nextState, reward, done, info = trainingEnv.step(action)
                
                # Process the RL variables retrieved and insert this new experience into the Experience Replay memory
                reward = self.processReward(reward)
                nextState = self.processState(nextState)
                self.replayMemory.push(state, action, reward, nextState, done)

                # Update the RL state
                state = nextState


    def learning(self):
        """
        GOAL: Sample a batch of past experiences and learn from it
              by updating the Reinforcement Learning policy.
        
        INPUTS: /
        
        OUTPUTS: - loss: Loss of the learning procedure.
        """
        
        # Check that the replay memory is filled enough
        if (len(self.replayMemory) >= self.batchSize):

            # Sample a batch of experiences from the replay memory
            batch = self.dataLoaderIter.next()
            state = batch[0].float().to(self.device)
            action = batch[1].long().to(self.device)
            reward = batch[2].float().to(self.device)
            nextState = batch[3].float().to(self.device)
            done = batch[4].float().to(self.device)

            # Compute the current Q values returned by the policy network
            currentQValues = self.policyNetwork(state).gather(1, action.unsqueeze(1)).squeeze(1)

            # Compute the next Q values returned by the target network
            with torch.no_grad():
                nextActions = torch.max(self.policyNetwork(nextState), 1)[1]
                nextQValues = self.targetNetwork(nextState).gather(1, nextActions.unsqueeze(1)).squeeze(1)
                #nextQValues = self.policyNetwork(nextState).gather(1, nextActions.unsqueeze(1)).squeeze(1) # Double DQN improvement
                expectedQValues = reward + self.gamma * nextQValues * (1 - done)

            # Compute the loss (typically Huber or MSE loss)
            loss = F.smooth_l1_loss(currentQValues, expectedQValues)
            #loss = F.mse_loss(currentQValues, expectedQValues)

            # Computation of the gradients
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), self.gradientClipping)

            # Perform the Deep Neural Network optimization
            self.optimizer.step()

            return loss.item()

        
    def training(self, trainingEnv, numberOfEpisodes, verbose=True, rendering=False, plotTraining=True):
        """
        GOAL: Train the RL agent by interacting with the RL environment.
        
        INPUTS: - trainingEnv: Training RL environment.
                - numberOfEpisodes: Number of episodes for the training phase.
                - verbose: Enable the printing of a training feedback.
                - rendering: Enable the environment rendering.
                - plotTraining: Enable the plotting of the training results.
        
        OUTPUTS: - trainingEnv: Training RL environment.
        """

        # Initialization of several variables for storing the training and testing results
        performanceTraining = []
        performanceTesting = []

        try:
            # If required, print the training progression
            if verbose:
                print("Training progression (hardware selected => " + str(self.device) + "):")

            # Fill the replay memory with a number of random experiences
            self.fillReplayMemory(trainingEnv)
            self.steps = 0

            # Training phase for the number of episodes specified as parameter
            for episode in range(numberOfEpisodes):
                
                # Set the initial RL variables
                state = self.processState(trainingEnv.reset())
                done = 0

                # Set the performance tracking veriables
                totalReward = 0

                # Interact with the training environment until termination
                while done == 0:

                    # Choose an action according to the RL policy and the current RL state
                    action = self.chooseActionEpsilonGreedy(state, self.epsilonValue(self.steps))
                    
                    # Interact with the environment with the chosen action
                    nextState, reward, done, info = trainingEnv.step(action)
                    
                    # Process the RL variables retrieved and insert this new experience into the Experience Replay memory
                    reward = self.processReward(reward)
                    nextState = self.processState(nextState)
                    self.replayMemory.push(state, action, reward, nextState, done)

                    # Execute the learning procedure of the RL algorithm
                    if self.steps % self.learningUpdatePeriod == 0:
                        self.dataLoader = DataLoader(dataset=self.replayMemory, batch_size=self.batchSize, shuffle=True)
                        self.dataLoaderIter = iter(self.dataLoader)
                        self.learning()

                    # If required, update the target deep neural network (update frequency)
                    self.updateTargetNetwork()

                    # Update the RL state
                    state = nextState

                    # Continuous tracking of the training performance
                    totalReward += reward

                    # Incrementation of the number of iterations (steps)
                    self.steps += 1
                    
                # Store and report the performance of the RL policy (training)
                performanceTraining.append([episode, self.steps, totalReward])
                self.writer.add_scalar("Training score (1)", totalReward, episode)
                self.writer.add_scalar("Training score (2)", totalReward, self.steps)

                # Store and report the performance of the RL policy (testing)
                if episode % 4 == 0:
                    _, testingScore = self.testing(trainingEnv, False, False)
                    performanceTesting.append([episode, self.steps, testingScore])
                    self.writer.add_scalar("Testing score (1)", testingScore, episode)
                    self.writer.add_scalar("Testing score (2)", testingScore, self.steps)

                # Store the training and testing results in a csv file
                if episode % 100 == 0:
                    dataframeTraining = pd.DataFrame(performanceTraining, columns=['Episode', 'Steps', 'Score'])
                    dataframeTraining.to_csv(''.join([self.experimentFolder, 'TrainingResults.csv']))
                    dataframeTesting = pd.DataFrame(performanceTesting, columns=['Episode', 'Steps', 'Score'])
                    dataframeTesting.to_csv(''.join([self.experimentFolder, 'TestingResults.csv']))

                # If required, print a training feedback
                if verbose:
                    print("".join(["Episode ", str(episode+1), "/", str(numberOfEpisodes), ": training score = ", str(totalReward)]), end='\r', flush=True)
        
        except KeyboardInterrupt:
            print()
            print("WARNING: Training prematurely interrupted...")
            print()

        # Assess the algorithm performance on the training environment
        trainingEnv, testingScore = self.testing(trainingEnv, verbose, rendering)

        # Store the training results into a csv file
        dataframeTraining = pd.DataFrame(performanceTraining, columns=['Episode', 'Steps', 'Score'])
        dataframeTraining.to_csv(''.join([self.experimentFolder, 'TrainingResults.csv']))

        # Store the testing results into a csv file
        dataframeTesting = pd.DataFrame(performanceTesting, columns=['Episode', 'Steps', 'Score'])
        dataframeTesting.to_csv(''.join([self.experimentFolder, 'TestingResults.csv']))

        # If required, plot the training results
        if plotTraining:
            plt.figure()
            dataframeTraining.plot(x='Episode', y='Score')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.savefig(''.join([self.experimentFolder, 'TrainingScore1.png']))
            plt.show()
            plt.figure()
            dataframeTraining.plot(x='Steps', y='Score')
            plt.xlabel('Steps')
            plt.ylabel('Score')
            plt.savefig(''.join([self.experimentFolder, 'TrainingScore2.png']))
            plt.show()
            plt.figure()
            dataframeTesting.plot(x='Episode', y='Score')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.savefig(''.join([self.experimentFolder, 'TestingScore1.png']))
            plt.show()
            plt.figure()
            dataframeTesting.plot(x='Steps', y='Score')
            plt.xlabel('Steps')
            plt.ylabel('Score')
            plt.savefig(''.join([self.experimentFolder, 'TestingScore2.png']))
            plt.show()

        # Closing of the tensorboard writer
        self.writer.close()
        
        return trainingEnv


    def testing(self, testingEnv, verbose=True, rendering=True):
        """
        GOAL: Test the RL agent trained on the RL environment provided.
        
        INPUTS: - testingEnv: Testing RL environment.
                - verbose: Enable the printing of the testing performance.
                - rendering: Enable the rendering of the RL environment.
        
        OUTPUTS: - testingEnv: Testing RL environment.
                 - testingScore : Score associated with the testing phase.
        """

        # Initialization of some RL variables
        state = self.processState(testingEnv.reset())
        done = 0

        # Initialization of some variables tracking the RL agent performance
        testingScore = 0

        # Interact with the environment until the episode termination
        while done == 0:

            # Choose an action according to the RL policy and the current RL state
            action = self.chooseActionEpsilonGreedy(state, self.epsilonTest)

            # If required, show the environment rendering
            if rendering:
                testingEnv.render()
                self.chooseAction(state, True)
                
            # Interact with the environment with the chosen action
            nextState, reward, done, _ = testingEnv.step(action)
                
            # Process the RL variables retrieved
            state = self.processState(nextState)
            reward = self.processReward(reward)

            # Continuous tracking of the training performance
            testingScore += reward

        # If required, print the testing performance
        if verbose:
            print("".join(["Test environment: score = ", str(testingScore)]))

        return testingEnv, testingScore

    
    def plotExpectedPerformance(self, trainingEnv, numberOfEpisodes, iterations=10):
        """
        GOAL: Plot the expected performance of the DRL algorithm.
        
        INPUTS: - trainingEnv: Training RL environment.
                - numberOfEpisodes: Number of episodes for the training phase.
                - iterations: Number of training/testing iterations to compute
                              the expected performance.
        
        OUTPUTS: - trainingEnv: Training RL environment.
        """

        # Save the initial Deep Neural Network weights
        initialWeights =  copy.deepcopy(self.policyNetwork.state_dict())

        # Initialization of several variables for monitoring the performance of the RL agent
        performanceTraining = np.zeros((numberOfEpisodes, iterations))
        performanceTesting = np.zeros((numberOfEpisodes, iterations))

        # Print the hardware selected for the training of the Deep Neural Network (either CPU or GPU)
        print("Hardware selected for training: " + str(self.device))

        try:

            # Apply the training/testing procedure for the number of iterations specified
            for iteration in range(iterations):

                # Print the progression
                print(''.join(["Expected performance evaluation progression: ", str(iteration+1), "/", str(iterations)]))

                # Fill the replay memory with a number of random experiences
                self.fillReplayMemory(trainingEnv)
                self.steps = 0

                # Training phase for the number of episodes specified as parameter
                for episode in tqdm(range(numberOfEpisodes)):
                    
                    # Set the initial RL variables
                    state = self.processState(trainingEnv.reset())
                    done = 0

                    # Set the performance tracking variables
                    totalReward = 0

                    # Interact with the training environment until termination
                    while done == 0:

                        # Choose an action according to the RL policy and the current RL state
                        action = self.chooseActionEpsilonGreedy(state, self.epsilonValue(self.steps))
                        
                        # Interact with the environment with the chosen action
                        nextState, reward, done, info = trainingEnv.step(action)
                        
                        # Process the RL variables retrieved and insert this new experience into the Experience Replay memory
                        reward = self.processReward(reward)
                        nextState = self.processState(nextState)
                        self.replayMemory.push(state, action, reward, nextState, done)

                       # Execute the learning procedure
                        if self.steps % self.learningUpdatePeriod == 0:
                            self.dataLoader = DataLoader(dataset=self.replayMemory, batch_size=self.batchSize, shuffle=True)
                            self.dataLoaderIter = iter(self.dataLoader)
                            self.learning()

                        # If required, update the target deep neural network (update frequency)
                        self.updateTargetNetwork()

                        # Update the RL state
                        state = nextState

                        # Continuous tracking of the training performance
                        totalReward += reward

                        # Incrementation of the number of iterations (steps)
                        self.steps += 1

                    # Store and report the performance of the RL policy (training)
                    performanceTraining[episode][iteration] = totalReward
                    self.writer.add_scalar("Training score (1)", totalReward, episode)
                    self.writer.add_scalar("Training score (2)", totalReward, self.steps)

                    # Store and report the performance of the RL policy (testing)
                    _, testingScore = self.testing(trainingEnv, False, False)
                    performanceTesting[episode][iteration] = testingScore
                    self.writer.add_scalar("Testing score (1)", testingScore, episode)
                    self.writer.add_scalar("Testing score (2)", testingScore, self.steps)

                # Restore the initial state of the RL agent
                if iteration < (iterations-1):
                    trainingEnv.reset()
                    self.policyNetwork.load_state_dict(initialWeights)
                    self.targetNetwork.load_state_dict(initialWeights)
                    self.optimizer = optim.Adam(self.policyNetwork.parameters(), lr=self.learningRate, eps=self.epsilon)
                    self.replayMemory.reset()
                    self.steps = 0
            
            iteration += 1
        
        except KeyboardInterrupt:
            print()
            print("WARNING: Expected performance evaluation prematurely interrupted...")
            print()

        # Smooth the training and testing performances for better readibility (moving average)
        performanceTraining = np.transpose(performanceTraining)
        performanceTesting = np.transpose(performanceTesting)
        for i in range(iterations):
            movingAverage = pd.DataFrame(performanceTraining[i], columns=['Score']).rolling(100, min_periods=1).mean()
            performanceTraining[i] = movingAverage['Score'].tolist()
            movingAverage = pd.DataFrame(performanceTesting[i], columns=['Score']).rolling(100, min_periods=1).mean()
            performanceTesting[i] = movingAverage['Score'].tolist()
        performanceTraining = np.transpose(performanceTraining)
        performanceTesting = np.transpose(performanceTesting)

        # Compute the expectation and standard deviation of the training and testing performances
        expectedPerformance = []
        stdPerformance = []
        for episode in range(numberOfEpisodes):
            expectedPerformance.append(np.mean(performanceTraining[episode][:iteration]))
            stdPerformance.append(np.std(performanceTraining[episode][:iteration]))
        expectedPerformanceTraining = np.array(expectedPerformance)
        stdPerformanceTraining = np.array(stdPerformance)
        expectedPerformance = []
        stdPerformance = []
        for episode in range(numberOfEpisodes):
            expectedPerformance.append(np.mean(performanceTesting[episode][:iteration]))
            stdPerformance.append(np.std(performanceTesting[episode][:iteration]))
        expectedPerformanceTesting = np.array(expectedPerformance)
        stdPerformanceTesting = np.array(stdPerformance)

        # Store the training and testing results into a csv file
        dataTraining = {'Expectation': expectedPerformanceTraining,
                        'StandardDeviation': stdPerformanceTraining}
        dataframeTraining = pd.DataFrame(dataTraining)
        dataframeTraining.to_csv(''.join([self.experimentFolder, 'TrainingResults.csv']))

        dataTesting = {'Expectation': expectedPerformanceTesting,
                        'StandardDeviation': stdPerformanceTesting}
        dataframeTesting = pd.DataFrame(dataTesting)
        dataframeTesting.to_csv(''.join([self.experimentFolder, 'TestingResults.csv']))

        # Plot the expected performance (training and testing)
        fig = plt.figure()
        ax = fig.add_subplot(111, ylabel='Training score', xlabel='Episode')
        ax.plot(expectedPerformanceTraining)
        ax.fill_between(range(len(expectedPerformanceTraining)), expectedPerformanceTraining-stdPerformanceTraining, expectedPerformanceTraining+stdPerformanceTraining, alpha=0.25)
        plt.savefig(''.join([self.experimentFolder, 'TrainingScores.png']))
        fig = plt.figure()
        ax = fig.add_subplot(111, ylabel='Testing score', xlabel='Episode')
        ax.plot(expectedPerformanceTesting)
        ax.fill_between(range(len(expectedPerformanceTesting)), expectedPerformanceTesting-stdPerformanceTesting, expectedPerformanceTesting+stdPerformanceTesting, alpha=0.25)
        plt.savefig(''.join([self.experimentFolder, 'TestingScores.png']))

        # Closing of the tensorboard writer
        self.writer.close()
        
        return trainingEnv

        
    def saveModel(self, fileName):
        """
        GOAL: Save the RL policy, by saving the policy Deep Neural Network.
        
        INPUTS: - fileName: Name of the file.
        
        OUTPUTS: /
        """

        torch.save(self.policyNetwork.state_dict(), fileName)


    def loadModel(self, fileName):
        """
        GOAL: Load the RL policy, by loading the policy Deep Neural Network.
        
        INPUTS: - fileName: Name of the file.
        
        OUTPUTS: /
        """

        self.policyNetwork.load_state_dict(torch.load(fileName, map_location=self.device))
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())


    def plotEpsilonAnnealing(self):
        """
        GOAL: Plot the annealing behaviour of the Epsilon variable
              (Epsilon-Greedy exploration technique).
        
        INPUTS: /
        
        OUTPUTS: /
        """

        plt.figure()
        plt.plot([self.epsilonValue(i) for i in range(1000000)])
        plt.xlabel("Steps")
        plt.ylabel("Epsilon")
        plt.show()
        