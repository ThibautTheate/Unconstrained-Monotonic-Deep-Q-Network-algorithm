# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import torch.nn as nn
# pylint: disable=E1101
# pylint: disable=E1102

from Models.FeedforwardDNN import FeedForwardDNN
from Models.CNN_Atari import CNN_Atari



###############################################################################
############################## Class DNN_Atari ################################
###############################################################################

class DNN_Atari(nn.Module):
    """
    GOAL: Implementing the orignal DNN designed for the DQN algorithm to
          succesfully play Atari games.
    
    VARIABLES:  - network: Deep Neural Network.
                                
    METHODS:    - __init__: Initialization of the Deep Neural Network.
                - forward: Forward pass of the Deep Neural Network.
    """

    def __init__(self, numberOfInputs, numberOfOutputs):
        """
        GOAL: Defining and initializing the Deep Neural Network.
        
        INPUTS: - numberOfInputs: Number of inputs of the Deep Neural Network.
                - numberOfOutputs: Number of outputs of the Deep Neural Network.
        
        OUTPUTS: /
        """

        # Call the constructor of the parent class (Pytorch torch.nn.Module)
        super(DNN_Atari, self).__init__()

        # Initialization of the Deep Neural Network.
        CNNOutputSize = CNN_Atari(numberOfInputs).getOutputSize()
        self.network = nn.Sequential(
            CNN_Atari(numberOfInputs),
            FeedForwardDNN(CNNOutputSize, numberOfOutputs, [512])
        )

    
    def forward(self, x):
        """
        GOAL: Implementing the forward pass of the Deep Neural Network.
        
        INPUTS: - x: Input of the Deep Neural Network.
        
        OUTPUTS: - y: Output of the Deep Neural Network.
        """

        return self.network(x)
