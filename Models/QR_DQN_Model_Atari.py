# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import torch.nn as nn
# pylint: disable=E1101
# pylint: disable=E1102

from Models.DNN_Atari import DNN_Atari
from Models.DNN_MinAtar import DNN_MinAtar



###############################################################################
######################## Class QR_DQN_Model_Atari #############################
###############################################################################

class QR_DQN_Model_Atari(nn.Module):
    """
    GOAL: Implementing the DL model for the QR-DQN distributional RL algorithm.
    
    VARIABLES:  - network: Deep Neural Network.
                                
    METHODS:    - __init__: Initialization of the Deep Neural Network.
                - forward: Forward pass of the Deep Neural Network.
    """

    def __init__(self, numberOfInputs, numberOfOutputs, numberOfQuantiles=200, minAtar=False):
        """
        GOAL: Defining and initializing the Deep Neural Network.
        
        INPUTS: - numberOfInputs: Number of inputs of the Deep Neural Network.
                - numberOfOutputs: Number of outputs of the Deep Neural Network.
                - numberOfQuantiles: Number of quantiles for approximating the distribution.
                - minAtar: Boolean specifying whether the env is "MinAtar" or not.
        
        OUTPUTS: /
        """

        # Call the constructor of the parent class (Pytorch torch.nn.Module)
        super(QR_DQN_Model_Atari, self).__init__()

        # Initialization of useful variables
        self.numberOfQuantiles = numberOfQuantiles
        self.numberOfActions = int(numberOfOutputs/numberOfQuantiles)
    
        # Initialization of the Deep Neural Network.
        if minAtar:
            self.network = DNN_MinAtar(numberOfInputs, numberOfOutputs)
        else:
            self.network = DNN_Atari(numberOfInputs, numberOfOutputs)

    
    def forward(self, x):
        """
        GOAL: Implementing the forward pass of the Deep Neural Network.
        
        INPUTS: - x: Input of the Deep Neural Network.
        
        OUTPUTS: - y: Output of the Deep Neural Network.
        """

        x = self.network(x)
        return x.view(x.size(0), self.numberOfActions, self.numberOfQuantiles)
