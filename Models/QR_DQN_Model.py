# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import torch.nn as nn
# pylint: disable=E1101
# pylint: disable=E1102

from Models.FeedforwardDNN import FeedForwardDNN



###############################################################################
############################# Class QR_DQN_Model ##############################
###############################################################################

class QR_DQN_Model(nn.Module):
    """
    GOAL: Implementing the DL model for the QR-DQN distributional RL algorithm.
    
    VARIABLES:  - network: Deep Neural Network.
                                
    METHODS:    - __init__: Initialization of the Deep Neural Network.
                - forward: Forward pass of the Deep Neural Network.
    """

    def __init__(self, numberOfInputs, numberOfOutputs, structure, numberOfQuantiles=200):
        """
        GOAL: Defining and initializing the Deep Neural Network.
        
        INPUTS: - numberOfInputs: Number of inputs of the Deep Neural Network.
                - numberOfOutputs: Number of outputs of the Deep Neural Network.
                - structure: Structure of the Deep Neural Network (hidden layers).
                - numberOfQuantiles: Number of quantiles for approximating the distribution.
        
        OUTPUTS: /
        """

        # Call the constructor of the parent class (Pytorch torch.nn.Module)
        super(QR_DQN_Model, self).__init__()

        # Initialization of useful variables
        self.numberOfQuantiles = numberOfQuantiles
        self.numberOfActions = int(numberOfOutputs/numberOfQuantiles)
    
        # Initialization of the Deep Neural Network.
        self.network = FeedForwardDNN(numberOfInputs, numberOfOutputs, structure)

    
    def forward(self, x):
        """
        GOAL: Implementing the forward pass of the Deep Neural Network.
        
        INPUTS: - x: Input of the Deep Neural Network.
        
        OUTPUTS: - y: Output of the Deep Neural Network.
        """

        x = self.network(x)
        return x.view(x.size(0), self.numberOfActions, self.numberOfQuantiles)
