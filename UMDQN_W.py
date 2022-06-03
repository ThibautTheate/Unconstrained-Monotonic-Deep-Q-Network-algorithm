# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import math

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import torch
import torch.optim as optim

from replayMemory import ReplayMemory

from Models.UMDQN_W_Model import UMDQN_W_Model
from Models.UMDQN_W_Model_Atari import UMDQN_W_Model_Atari

from DQN import DQN



###############################################################################
############################### Class UMDQN_W #################################
###############################################################################

class UMDQN_W(DQN):
    """
    GOAL: Implementing the UMDQN_W Deep Reinforcement Learning algorithm.
    
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
        GOAL: Initializing the RL agent based on the UMDQN_W Deep Reinforcement Learning
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
            parametersFileName = ''.join(['Parameters/parameters_UMDQN_W_', str(environment), '.json'])
        parameters = self.readParameters(parametersFileName)

        # Set the device for DNN computations (CPU or GPU)
        self.device = torch.device('cuda:'+str(parameters['GPUNumber']) if torch.cuda.is_available() else 'cpu')

        # Set the general parameters of the RL algorithm
        self.gamma = parameters['gamma']
        self.learningRate = parameters['learningRate']
        self.epsilon = parameters['epsilon']
        self.targetUpdatePeriod = parameters['targetUpdatePeriod']
        self.learningUpdatePeriod = parameters['learningUpdatePeriod']
        self.rewardClipping = parameters['rewardClipping']
        self.gradientClipping = parameters['gradientClipping']

        # Set the Experience Replay mechanism
        self.batchSize = parameters['batchSize']
        self.capacity = parameters['capacity']
        self.replayMemory = ReplayMemory(self.capacity)

        # Set the distribution support (quantile fractions)
        self.numberOfSamples = parameters['numberOfSamples']
        self.support = np.linspace(0.0, 1.0, self.numberOfSamples)
        self.supportTorch = torch.linspace(0.0, 1.0, self.numberOfSamples, device=self.device)
        self.supportRepeatedBatchSize = self.supportTorch.repeat(self.batchSize, 1).view(-1, 1)
        self.kappa = 1.0
        
        # Set the two Deep Neural Networks of the RL algorithm (policy and target)
        self.atari = parameters['atari']
        self.minatar = parameters['minatar']
        if self.atari or self.minatar:
            self.policyNetwork = UMDQN_W_Model_Atari(observationSpace, actionSpace, parameters['structureUMNN'], parameters['stateEmbedding'], parameters['numberOfSteps'], self.device, minAtar=self.minatar).to(self.device)
            self.targetNetwork = UMDQN_W_Model_Atari(observationSpace, actionSpace, parameters['structureUMNN'], parameters['stateEmbedding'], parameters['numberOfSteps'], self.device, minAtar=self.minatar).to(self.device)
        else:
            self.policyNetwork = UMDQN_W_Model(observationSpace, actionSpace, parameters['structureDNN'], parameters['structureUMNN'], parameters['stateEmbedding'], parameters['numberOfSteps'], self.device).to(self.device)
            self.targetNetwork = UMDQN_W_Model(observationSpace, actionSpace, parameters['structureDNN'], parameters['structureUMNN'], parameters['stateEmbedding'], parameters['numberOfSteps'], self.device).to(self.device)
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
        self.initReporting(parameters, 'UMDQN_W')


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
            quantiles = self.policyNetwork(state, self.supportTorch.unsqueeze(1))
            QValues = quantiles.mean(1)
            _, action = QValues.max(0)

            # If required, plot the return distribution associated with each action
            if plot:
                colors = ['blue', 'red', 'orange', 'green', 'purple', 'brown']
                plt.figure()
                ax = plt.subplot(1, 1, 1)
                taus = torch.linspace(0.0, 1.0, self.numberOfSamples*10, device=self.device).unsqueeze(1)
                quantiles = self.policyNetwork(state, taus)
                QValues = quantiles.mean(1)
                taus = taus.cpu().numpy()
                quantiles = quantiles.squeeze(0).cpu().numpy()
                QValues = QValues.squeeze(0).cpu().numpy()
                for a in range(self.actionSpace):
                    ax.plot(taus, quantiles[a], linestyle='-', label=''.join(['Action ', str(a), ' random return Z']), color=colors[a])
                    ax.axhline(y=QValues[a], linewidth=2, linestyle='--', label=''.join(['Action ', str(a), ' expected return Q']), color=colors[a])
                ax.set_xlabel('Quantile fraction')
                ax.set_ylabel('Quantile Function (QF)')
                ax.legend()
                plt.show()
                """
                # Saving of the data into external files
                taus = np.linspace(0, 1, self.numberOfSamples*10)
                dataQF = {
                'Action0_x': taus,
                'Action0_y': quantiles[0],
                'Action1_x': taus,
                'Action1_y': quantiles[1],
                'Action2_x': taus,
                'Action2_y': quantiles[2],
                'Action3_x': taus,
                'Action3_y': quantiles[3],
                }
                dataframeQF = pd.DataFrame(dataQF)
                dataframeQF.to_csv('Figures/Distributions/UMDQN_W.csv')
                quit()
                """
            
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
            quantiles = self.policyNetwork(state, self.supportRepeatedBatchSize)
            selection = torch.tensor([self.actionSpace*i + action[i] for i in range(self.batchSize)], dtype=torch.long, device=self.device)
            quantiles = torch.index_select(quantiles, 0, selection)

            # Computation of the new distribution to be learnt by the policy DNN
            with torch.no_grad():
                nextQuantiles = self.targetNetwork(nextState, self.supportRepeatedBatchSize)
                nextAction = nextQuantiles.view(self.batchSize, self.actionSpace, self.numberOfSamples).mean(2).max(1)[1]
                selection = torch.tensor([self.actionSpace*i + nextAction[i] for i in range(self.batchSize)], dtype=torch.long, device=self.device)
                nextQuantiles = torch.index_select(nextQuantiles, 0, selection)
                targetQuantiles = reward.unsqueeze(1) + self.gamma * nextQuantiles * (1 - done.unsqueeze(1))

            #"""
            # Improve stability with the lower and upper bounds of the random return
            minZ = -1
            maxZ = 10
            quantiles = quantiles.clamp(min=minZ, max=maxZ)
            targetQuantiles = targetQuantiles.clamp(min=minZ, max=maxZ)
            #"""
            
            # Computation of the loss
            difference = targetQuantiles.unsqueeze(1) - quantiles.unsqueeze(2)
            error = difference.abs()
            loss = torch.where(error <= self.kappa, 0.5 * error.pow(2), self.kappa * (error - (0.5 * self.kappa)))
            loss = (self.supportRepeatedBatchSize.view(self.batchSize, self.numberOfSamples, 1) - (difference < 0).float()).abs() * loss/self.kappa
            loss = loss.mean(1).sum(1).mean()

            # Computation of the gradients
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), self.gradientClipping)

            # Perform the Deep Neural Network optimization
            self.optimizer.step()

            return loss.item()
