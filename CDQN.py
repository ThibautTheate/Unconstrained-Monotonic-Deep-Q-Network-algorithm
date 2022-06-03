# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import math

import numpy as np

from matplotlib import pyplot as plt

import torch
import torch.optim as optim

from replayMemory import ReplayMemory

from Models.CDQN_Model import CDQN_Model
from Models.CDQN_Model_Atari import CDQN_Model_Atari

from DQN import DQN



###############################################################################
############################### Class CDQN ####################################
###############################################################################

class CDQN(DQN):
    """
    GOAL: Implementing the Categorical DQN (C51) Deep Reinforcement Learning algorithm.
    
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
        GOAL: Initializing the RL agent based on the CDQN Deep Reinforcement Learning
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
            parametersFileName = ''.join(['Parameters/parameters_CDQN_', str(environment), '.json'])
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

        # Set the distribution support
        self.numberOfAtoms = parameters['numberOfAtoms']
        self.minReturn = parameters['minReturn']
        self.maxReturn = parameters['maxReturn']
        self.support = np.linspace(self.minReturn, self.maxReturn, self.numberOfAtoms)
        self.supportTorch = torch.linspace(self.minReturn, self.maxReturn, self.numberOfAtoms).to(self.device)

        # Set the two Deep Neural Networks of the RL algorithm (policy and target)
        self.atari = parameters['atari']
        self.minatar = parameters['minatar']
        if self.atari or self.minatar:
            self.policyNetwork = CDQN_Model_Atari(observationSpace, actionSpace*self.numberOfAtoms, self.numberOfAtoms, minAtar=self.minatar).to(self.device)
            self.targetNetwork = CDQN_Model_Atari(observationSpace, actionSpace*self.numberOfAtoms, self.numberOfAtoms, minAtar=self.minatar).to(self.device)
        else:
            self.policyNetwork = CDQN_Model(observationSpace, actionSpace*self.numberOfAtoms, parameters['structureDNN'], self.numberOfAtoms).to(self.device)
            self.targetNetwork = CDQN_Model(observationSpace, actionSpace*self.numberOfAtoms, parameters['structureDNN'], self.numberOfAtoms).to(self.device)
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
        self.initReporting(parameters, 'CDQN')


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
            distribution = self.policyNetwork(state).squeeze(0)
            distributionReturn = distribution * self.supportTorch
            QValues = distributionReturn.sum(1)
            _, action = QValues.max(0)

            # If required, plot the return distribution associated with each action
            if plot:
                colors = ['blue', 'red', 'orange', 'green', 'purple', 'brown']
                fig = plt.figure()
                ax = fig.add_subplot()
                QValues = QValues.cpu().numpy()
                for a in range(self.actionSpace):
                    dist = distribution[a].cpu().numpy()
                    ax.bar(self.support, dist, label=''.join(['Action ', str(a), ' random return Z']), width=(self.maxReturn-self.minReturn)/self.numberOfAtoms, edgecolor='black', alpha=0.5, color=colors[a])
                    ax.axvline(x=QValues[a], linewidth=2, linestyle='--', label=''.join(['Action ', str(a), ' expected return Q']), color=colors[a])
                ax.set_xlabel('Random return')
                ax.set_ylabel('Probability Density Function (PDF)')
                ax.legend()
                plt.show()
            
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
            distribution = self.policyNetwork(state)
            action = action.unsqueeze(1).unsqueeze(1).expand(self.batchSize, 1, self.numberOfAtoms)
            distribution = distribution.gather(1, action).squeeze(1)

            # Computation of the new distribution to be learnt by the policy DNN
            with torch.no_grad():
                nextDistribution = self.targetNetwork(nextState)
                nextAction = (nextDistribution * self.supportTorch).sum(2).max(1)[1].unsqueeze(1).unsqueeze(1).expand(self.batchSize, 1, self.numberOfAtoms)
                nextDistribution = nextDistribution.gather(1, nextAction).squeeze(1)
                deltaZ = float(self.maxReturn - self.minReturn) / (self.numberOfAtoms - 1)
                tz = reward.view(-1, 1) + (1 - done.view(-1, 1)) * self.gamma * self.supportTorch
                tz = tz.clamp(min=self.minReturn, max=self.maxReturn)
                b  = ((tz - self.minReturn) / deltaZ)
                l  = b.floor().long()
                u  = b.ceil().long()
                offset = torch.linspace(0, (self.batchSize - 1) * self.numberOfAtoms, self.batchSize).long().unsqueeze(1).expand(self.batchSize, self.numberOfAtoms).to(self.device)
                projectedDistribution = torch.zeros(nextDistribution.size()).to(self.device)
                projectedDistribution.view(-1).index_add_(0, (l + offset).view(-1), (nextDistribution * (u.float() - b)).view(-1))
                projectedDistribution.view(-1).index_add_(0, (u + offset).view(-1), (nextDistribution * (b - l.float())).view(-1))

            # Computation of the loss
            loss = -(projectedDistribution * distribution.log()).sum(1).mean()

            # Computation of the gradients
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), self.gradientClipping)

            # Perform the Deep Neural Network optimization
            self.optimizer.step()

            return loss.item()
