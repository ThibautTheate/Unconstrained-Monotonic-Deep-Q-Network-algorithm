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

from Models.FQF_Model import FQF_Model
from Models.FQF_Model_Atari import FQF_Model_Atari
from Models.FQF_Model_Bis import FQF_Model_Bis

from DQN import DQN



###############################################################################
################################## Class FQF ##################################
###############################################################################

class FQF(DQN):
    """
    GOAL: Implementing the FQF Deep Reinforcement Learning algorithm.
    
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
        GOAL: Initializing the RL agent based on the FQF Deep Reinforcement Learning
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
            parametersFileName = ''.join(['Parameters/parameters_FQF_', str(environment), '.json'])
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
            self.policyNetwork = FQF_Model_Atari(observationSpace, actionSpace, self.NCos, self.device).to(self.device)
            self.targetNetwork = FQF_Model_Atari(observationSpace, actionSpace, self.NCos, self.device).to(self.device)
            stateEmbedding = self.policyNetwork.getEmbeddingSize()
        else:
            self.policyNetwork = FQF_Model(observationSpace, actionSpace, parameters['structureDNN'], parameters['stateEmbedding'], self.NCos, self.device).to(self.device)
            self.targetNetwork = FQF_Model(observationSpace, actionSpace, parameters['structureDNN'], parameters['stateEmbedding'], self.NCos, self.device).to(self.device)
            stateEmbedding = parameters['stateEmbedding']
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())

        # Set the Deep Learning optimizer
        self.optimizer = optim.Adam(self.policyNetwork.parameters(), lr=self.learningRate, eps=self.epsilon)

        # Set the Fraction Proposal Network of the FQF algorithm + associated parameters
        self.fractionProposalNetwork = FQF_Model_Bis(stateEmbedding, self.N, self.device).to(self.device)
        self.optimizerFPN = optim.RMSprop(self.fractionProposalNetwork.parameters(), lr=0.000000001, alpha=0.95, eps=0.00001)
        self.entropyCoefficient = 0.001

        # Set the Epsilon-Greedy exploration technique
        self.epsilonStart = parameters['epsilonStart']
        self.epsilonEnd = parameters['epsilonEnd']
        self.epsilonDecay = parameters['epsilonDecay']
        self.epsilonTest = parameters['epsilonTest']
        self.epsilonValue = lambda iteration: self.epsilonEnd + (self.epsilonStart - self.epsilonEnd) * math.exp(-1 * iteration / self.epsilonDecay)

        # Initialization of the experiment folder and tensorboard writer
        self.initReporting(parameters, 'FQF')


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
            stateEmbedding = self.policyNetwork.embedding(state)
            _, tausBis, _ = self.fractionProposalNetwork(stateEmbedding)
            quantiles = self.policyNetwork(state, tausBis, stateEmbedding)
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
            stateEmbedding = self.policyNetwork.embedding(state)
            taus, tausBis, entropy = self.fractionProposalNetwork(stateEmbedding)
            quantiles = self.policyNetwork(state, tausBis, stateEmbedding)
            actionBis = action.view(self.batchSize, 1, 1).expand(self.batchSize, 1, self.N)
            quantiles = quantiles.gather(1, actionBis).squeeze(1)

            # Computation of the Fractional loss for the FPN
            with torch.no_grad():
                quantilesBis = self.policyNetwork(state, taus[:, 1:-1], stateEmbedding)
                actionBis = action.view(self.batchSize, 1, 1).expand(self.batchSize, 1, self.N-1)
                quantilesBis = quantilesBis.gather(1, actionBis).squeeze(1)
            gradients1 = quantilesBis - quantiles[:, :-1]
            gradients2 = quantilesBis - quantiles[:, 1:]
            flag1 = quantilesBis > torch.cat([quantiles[:, :1], quantilesBis[:, :-1]], dim=1)
            flag2 = quantilesBis < torch.cat([quantilesBis[:, 1:], quantiles[:, -1:]], dim=1)
            gradients = (torch.where(flag1, gradients1, - gradients1) + torch.where(flag2, gradients2, -gradients2)).view(self.batchSize, self.N-1)
            fractionalLoss = (gradients * taus[:, 1:-1]).sum(dim=1).mean()
            fractionalLoss += self.entropyCoefficient * entropy.mean()

            # Computation of the new distribution to be learnt by the policy DNN
            with torch.no_grad(): 
                nextStateEmbedding = self.targetNetwork.embedding(nextState)
                nextQuantiles = self.targetNetwork(nextState, tausBis, nextStateEmbedding)
                nextAction = nextQuantiles.mean(2).max(1)[1].view(self.batchSize, 1, 1).expand(self.batchSize, 1, self.N)
                nextQuantiles = nextQuantiles.gather(1, nextAction).squeeze(1)
                targetQuantiles = reward.unsqueeze(1) + self.gamma * nextQuantiles * (1 - done.unsqueeze(1))

            # Computation of the quantile huber loss
            difference = targetQuantiles.unsqueeze(1) - quantiles.unsqueeze(2)
            error = difference.abs()
            loss = torch.where(error <= self.kappa, 0.5 * error.pow(2), self.kappa * (error - (0.5 * self.kappa)))
            loss = (tausBis.unsqueeze(2) - (difference < 0).float()).abs() * loss/self.kappa
            loss = loss.mean(1).sum(1).mean()

            # Update of the Fraction Proposal Network parameters
            self.optimizerFPN.zero_grad()
            fractionalLoss.backward(retain_graph=True)
            self.optimizerFPN.step()

            # Computation of the gradients
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), self.gradientClipping)

            # Perform the Deep Neural Network optimization
            self.optimizer.step()

            return loss.item()
