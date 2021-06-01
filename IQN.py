# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import math

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# pylint: disable=E1101
# pylint: disable=E1102

from replayMemory import ReplayMemory

from Models.IQN_Model import IQN_Model
from Models.IQN_Model_Atari import IQN_Model_Atari

from DQN import DQN



###############################################################################
################################## Class IQN ##################################
###############################################################################

class IQN(DQN):
    """
    GOAL: Implementing the IQN Deep Reinforcement Learning algorithm.
    
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
                                
    METHODS: - __init__: Initialization of the RL algorithm.
             - chooseAction: Choose a valid action based on the current state
                             observed, according to the RL policy learned.
             - learning: Execute the RL algorithm learning procedure.
    """

    def __init__(self, observationSpace, actionSpace, environment,
                 parametersFileName='', reporting=True):
        """
        GOAL: Initializing the RL agent based on the IQN Deep Reinforcement Learning
              algorithm, by setting up the algorithm parameters as well as 
              the Deep Neural Networks.
        
        INPUTS: - observationSpace: RL observation space.
                - actionSpace: RL action space.
                - environment: Name of the RL environment.
                - parametersFileName: Name of the JSON parameters file.
                - reporting: Enable the reporting of the results.
        
        OUTPUTS: /
        """

        # Initialization of the DQN parent class
        DQN.__init__(self, observationSpace, actionSpace, environment, parametersFileName, False)

        # Setting of the parameters
        if parametersFileName == '':
            parametersFileName = ''.join(['Parameters/parameters_IQN_', str(environment), '.json'])
        parameters = self.readParameters(parametersFileName)

        # Set the device for DNN computations (CPU or GPU)
        self.device = torch.device('cuda:'+str(parameters['GPUNumber']) if torch.cuda.is_available() else 'cpu')

        # Set the general parameters of the RL algorithm
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

        # Set the distribution support
        self.N = parameters['N']
        self.K = parameters['K']
        self.NCos = parameters['NCos']
        self.kappa = 1.0

        # Set the two Deep Neural Networks of the RL algorithm (policy and target)
        self.atari = parameters['atari']
        if self.atari:
            self.policyNetwork = IQN_Model_Atari(observationSpace, actionSpace, self.NCos, self.device).to(self.device)
            self.targetNetwork = IQN_Model_Atari(observationSpace, actionSpace, self.NCos, self.device).to(self.device)
        else:
            self.policyNetwork = IQN_Model(observationSpace, actionSpace, parameters['structureDNN'], parameters['stateEmbedding'], self.NCos, self.device).to(self.device)
            self.targetNetwork = IQN_Model(observationSpace, actionSpace, parameters['structureDNN'], parameters['stateEmbedding'], self.NCos, self.device).to(self.device)
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())

        # Set the Deep Learning optimizer
        self.optimizer = optim.Adam(self.policyNetwork.parameters(), lr=self.learningRate, eps=self.epsilon)

        # Set the Epsilon-Greedy exploration technique
        self.epsilonStart = parameters['epsilonStart']
        self.epsilonEnd = parameters['epsilonEnd']
        self.epsilonDecay = parameters['epsilonDecay']
        self.epsilonTest = parameters['epsilonTest']
        self.epsilonValue = lambda iteration: self.epsilonEnd + (self.epsilonStart - self.epsilonEnd) * math.exp(-1 * iteration / self.epsilonDecay)

        # Initialization of the experiment folder and tensorboard writer
        self.initReporting(parameters, 'IQN')


    def chooseAction(self, state, plot=False):
        """
        GOAL: Choose a valid RL action from the action space according to the
              RL policy as well as the current RL state observed.
        
        INPUTS: - state: RL state returned by the environment.
                - plot: Enable the plotting of the random returns distributions.
        
        OUTPUTS: - action: RL action chosen from the action space.
        """

        # Choose the best action based on the RL policy
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            quantiles, _ = self.policyNetwork(state, self.K)
            Qvalues = quantiles.mean(2)
            _, action = Qvalues.max(1)
            
            return action.item()


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

            # Computation of the current return distribution
            quantiles, taus = self.policyNetwork(state, self.N)
            action = action.view(self.batchSize, 1, 1).expand(self.batchSize, 1, self.N)
            quantiles = quantiles.gather(1, action).squeeze(1)

            # Computation of the new distribution to be learnt by the policy DNN
            with torch.no_grad(): 
                nextQuantiles, _ = self.targetNetwork(nextState, self.N)
                nextAction = nextQuantiles.mean(2).max(1)[1].view(self.batchSize, 1, 1).expand(self.batchSize, 1, self.N)
                nextQuantiles = nextQuantiles.gather(1, nextAction).squeeze(1)
                targetQuantiles = reward.unsqueeze(1) + self.gamma * nextQuantiles * (1 - done.unsqueeze(1))

            # Computation of the loss
            difference = targetQuantiles.unsqueeze(1) - quantiles.unsqueeze(2)
            error = difference.abs()
            loss = torch.where(error <= self.kappa, 0.5 * error.pow(2), self.kappa * (error - (0.5 * self.kappa)))
            loss = (taus - (difference < 0).float()).abs() * loss/self.kappa
            loss = loss.mean(1).sum(1).mean()

            # Computation of the gradients
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), self.gradientClipping)

            # Perform the Deep Neural Network optimization
            self.optimizer.step()

            return loss.item()
