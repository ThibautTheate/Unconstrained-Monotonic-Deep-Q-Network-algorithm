# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import torch
import torch.nn as nn
# pylint: disable=E1101
# pylint: disable=E1102



###############################################################################
############################# Class CNN_MinAtar ###############################
###############################################################################

class CNN_MinAtar(nn.Module):
    """
    GOAL: Implementing the CNN part of the DNN designed for the DQN algorithm
          to successfully play Atari games (MinAtar version).
    
    VARIABLES:  - network: Convolutional Neural Network.
                                
    METHODS:    - __init__: Initialization of the Convolutional Neural Network.
                - forward: Forward pass of the Convolutional Neural Network.
    """

    def __init__(self, numberOfInputs):
        """
        GOAL: Defining and initializing the Convolutional Neural Network.
        
        INPUTS: - numberOfInputs: Number of inputs of the Convolutional Neural Network.
        
        OUTPUTS: /
        """

        # Call the constructor of the parent class (Pytorch torch.nn.Module)
        super(CNN_MinAtar, self).__init__()

        # Initialization of some variables
        self.channels = numberOfInputs
        self.size = 10
        self.filters = 16
        self.kernel = 3
        self.stride = 1

        # Initialization of the Convolutional Neural Network
        self.network = nn.Sequential(
            nn.Conv2d(self.channels, self.filters, self.kernel, self.stride),
            nn.ReLU()
        )

    
    def getOutputSize(self):
        """
        GOAL: Get the size of the Convolutional Neural Network output.
        
        INPUTS: /
        
        OUTPUTS: - size: Size of the Convolutional Neural Network. output.
        """

        newSize = ((self.size - self.kernel)/self.stride) + 1
        return int(newSize * newSize * self.filters)

    
    def forward(self, x):
        """
        GOAL: Implementing the forward pass of the Convolutional Neural Network.
        
        INPUTS: - x: Input of the Convolutional Neural Network.
        
        OUTPUTS: - y: Output of the Convolutional Neural Network.
        """
        
        x = self.network(x)
        return x.view(x.size(0), -1)
