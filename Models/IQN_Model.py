# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import numpy as np
import torch
import torch.nn as nn
# pylint: disable=E1101
# pylint: disable=E1102

from Models.FeedforwardDNN import FeedForwardDNN



###############################################################################
############################## Class IQN_Model ################################
###############################################################################

class IQN_Model(nn.Module):
    """
    GOAL: Implementing the DL model for the IQN distributional RL algorithm.
    
    VARIABLES:  - network: Deep Neural Network.
                                
    METHODS:    - __init__: Initialization of the Deep Neural Network.
                - forward: Forward pass of the Deep Neural Network.
    """

    def __init__(self, numberOfInputs, numberOfOutputs, structure, stateEmbedding, NCos=64, device='cpu'):
        """
        GOAL: Defining and initializing the Deep Neural Network.
        
        INPUTS: - numberOfInputs: Number of inputs of the Deep Neural Network.
                - numberOfOutputs: Number of outputs of the Deep Neural Network.
                - structure: Structure of the state embedding Deep Neural Network (hidden layers).
                - stateEmbedding: Number of values to represent the state.
                - Ncos: Number of elements in cosine function.
        
        OUTPUTS: /
        """

        # Call the constructor of the parent class (Pytorch torch.nn.Module)
        super(IQN_Model, self).__init__()

        # Initialization of useful variables
        self.device = device
        self.NCos = NCos
        self.piMultiples = torch.tensor([np.pi*i for i in range(self.NCos)], dtype=torch.float).view(1, 1, self.NCos).to(self.device)
    
        # Initialization of the Deep Neural Network
        self.stateEmbedding = FeedForwardDNN(numberOfInputs, stateEmbedding, structure)
        self.cosEmbedding = nn.Sequential(nn.Linear(NCos, stateEmbedding), nn.ReLU())
        self.feedForwardDNN = FeedForwardDNN(stateEmbedding, numberOfOutputs, [256])

    
    def forward(self, x, N, randomSampling=True):
        """
        GOAL: Implementing the forward pass of the Deep Neural Network.
        
        INPUTS: - x: Input of the Deep Neural Network.
                - N: Number of quantiles to generate.
                - randomSampling: Boolean specifying whether the quantiles are
                                  sampled randomly or not (default: True).
        
        OUTPUTS: - y: Output of the Deep Neural Network.
        """

        # State embedding part of the Deep Neural Network
        batchSize = x.size(0)
        x = self.stateEmbedding(x).unsqueeze(1)

        # Generate a number of quantiles (randomly or not)
        if randomSampling:
            taus = torch.rand(batchSize, N).to(self.device).unsqueeze(2)
        else:
            taus = torch.linspace(0.0, 1.0, N).to(self.device)
            taus = taus.repeat(batchSize, 1).unsqueeze(2)
            
        # Quantile embedding part of the Deep Neural Network
        cos = torch.cos(taus*self.piMultiples).view(batchSize*N, self.NCos)
        cos = self.cosEmbedding(cos).view(batchSize, N, -1)

        # Multiplication of both state and cos embeddings outputs (combination)
        x = (x * cos).view(batchSize, N, -1)

        # Distribution part of the Deep Neural Network
        x = self.feedForwardDNN(x)
        return x.transpose(1, 2), taus
