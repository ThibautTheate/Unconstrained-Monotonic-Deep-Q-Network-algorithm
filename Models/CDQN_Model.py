# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import torch.nn as nn
import torch.nn.functional as F
# pylint: disable=E1101
# pylint: disable=E1102

from Models.FeedforwardDNN import FeedForwardDNN



###############################################################################
################################ Class CDQN_Model #############################
###############################################################################

class CDQN_Model(nn.Module):
    """
    GOAL: Implementing the DL model for the CDQN distributional RL algorithm.
    
    VARIABLES:  - network: Deep Neural Network.
                                
    METHODS:    - __init__: Initialization of the Deep Neural Network.
                - forward: Forward pass of the Deep Neural Network.
    """

    def __init__(self, numberOfInputs, numberOfOutputs, structure, numberOfAtoms=51):
        """
        GOAL: Defining and initializing the Deep Neural Network.
        
        INPUTS: - numberOfInputs: Number of inputs of the Deep Neural Network.
                - numberOfOutputs: Number of outputs of the Deep Neural Network.
                - structure: Structure of the Deep Neural Network (hidden layers).
                - numberOfAtoms: Number of atoms for the support (see C51 algorithm).
        
        OUTPUTS: /
        """

        # Call the constructor of the parent class (Pytorch torch.nn.Module)
        super(CDQN_Model, self).__init__()

        # Initialization of useful variables
        self.numberOfAtoms = numberOfAtoms
        self.numberOfActions = int(numberOfOutputs/numberOfAtoms)
    
        # Initialization of the Deep Neural Network
        self.network = FeedForwardDNN(numberOfInputs, numberOfOutputs, structure)

    
    def forward(self, x):
        """
        GOAL: Implementing the forward pass of the Deep Neural Network.
        
        INPUTS: - x: Input of the Deep Neural Network.
        
        OUTPUTS: - y: Output of the Deep Neural Network.
        """

        x = self.network(x)
        y = F.softmax(x.view(-1, self.numberOfActions, self.numberOfAtoms), dim=-1)
        return y.clamp(min=0.000001)
