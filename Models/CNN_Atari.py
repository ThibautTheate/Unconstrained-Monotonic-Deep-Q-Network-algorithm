# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import torch
import torch.nn as nn
# pylint: disable=E1101
# pylint: disable=E1102



###############################################################################
############################## Class CNN_Atari ################################
###############################################################################

class CNN_Atari(nn.Module):
    """
    GOAL: Implementing the CNN part of the DNN designed for the DQN algorithm
          to successfully play Atari games.
    
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
        super(CNN_Atari, self).__init__()

        # Initialization of the Convolutional Neural Network
        self.network = nn.Sequential(
            nn.Conv2d(numberOfInputs, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

    
    def getOutputSize(self):
        """
        GOAL: Get the size of the Convolutional Neural Network output.
        
        INPUTS: /
        
        OUTPUTS: - size: Size of the Convolutional Neural Network. output.
        """

        return self.network(torch.zeros(1, *(4, 84, 84))).view(1, -1).size(1)

    
    def forward(self, x):
        """
        GOAL: Implementing the forward pass of the Convolutional Neural Network.
        
        INPUTS: - x: Input of the Convolutional Neural Network.
        
        OUTPUTS: - y: Output of the Convolutional Neural Network.
        """

        x = self.network(x)
        return x.view(x.size(0), -1)
