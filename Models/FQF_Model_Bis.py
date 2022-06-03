# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import torch
import torch.nn as nn
# pylint: disable=E1101
# pylint: disable=E1102



###############################################################################
############################# Class FQF_Model_Bis #############################
###############################################################################

class FQF_Model_Bis(nn.Module):
    """
    GOAL: Implementing the DL model for the FQF distributional RL algorithm
          (Fraction Proposal Network).
    
    VARIABLES:  - network: Deep Neural Network.
                                
    METHODS:    - __init__: Initialization of the Deep Neural Network.
                - forward: Forward pass of the Deep Neural Network.
    """

    def __init__(self, numberOfInputs, numberOfOutputs, device='cpu'):
        """
        GOAL: Defining and initializing the Deep Neural Network.
        
        INPUTS: - numberOfInputs: Input shape (state embedding).
                - numberOfOutputs: Output shape (number of quantile fractions).
                - device: Running device (hardware acceleration).
        
        OUTPUTS: /
        """

        # Call the constructor of the parent class (Pytorch torch.nn.Module)
        super(FQF_Model_Bis, self).__init__()

        # Initialization of useful variables
        self.device = device
        self.N = numberOfOutputs

        # Initialization of the Deep Neural Network.
        self.network = nn.Sequential(
            nn.Linear(numberOfInputs, numberOfOutputs),
            nn.LogSoftmax(dim=1)
        )

    
    def forward(self, x):
        """
        GOAL: Implementing the forward pass of the Deep Neural Network.
        
        INPUTS: - x: Input of the Deep Neural Network. (state embedding).
        
        OUTPUTS: - taus: Quantile fractions generated.
                 - tausBis: Quantile fractions generated.
                 - entropy: Entropy associated with the DNN output.
        """

        # Generation of quantile fractions
        out = self.network(x)
        taus = torch.cumsum(out.exp(), dim=1)
        taus = torch.cat((torch.zeros((out.shape[0], 1)).to(self.device), taus), dim=1)
        tausBis = (taus[:, :-1] + taus[:, 1:]).detach() / 2.

        # Computation of the associated entropy
        entropy = -(out * out.exp()).sum(dim=-1, keepdim=True)

        return taus, tausBis, entropy
        