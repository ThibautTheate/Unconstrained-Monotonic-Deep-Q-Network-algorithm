# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import torch
import torch.nn as nn
# pylint: disable=E1101
# pylint: disable=E1102

from Models.DNN_Atari import DNN_Atari
from Models.MonotonicNN import MonotonicNN



###############################################################################
########################### Class UMDQN_W_Model_Atari #########################
###############################################################################

class UMDQN_W_Model_Atari(nn.Module):
    """
    GOAL: Implementing the DL model for the UMDQN-W distributional RL algorithm.
    
    VARIABLES:  - stateEmbeddingDNN: State embedding part of the Deep Neural Network.
                - UMNN: UMNN part of the Deep Neural Network.
                                
    METHODS:    - __init__: Initialization of the Deep Neural Network.
                - forward: Forward pass of the Deep Neural Network.
    """

    def __init__(self, numberOfInputs, numberOfOutputs,
                 structureUMNN, stateEmbedding,
                 numberOfSteps, device='cpu'):
        """
        GOAL: Defining and initializing the Deep Neural Network.
        
        INPUTS: - numberOfInputs: Number of inputs of the Deep Neural Network.
                - numberOfOutputs: Number of outputs of the Deep Neural Network.
                - structureUMNN: Structure of the UMNN for distribution representation.
                - stateEmbedding: Dimension of the state embedding.
                - numberOfSteps: Number of integration steps for the UMNN.
                - device: Hardware device (CPU or GPU).
        
        OUTPUTS: /
        """

        # Call the constructor of the parent class (Pytorch torch.nn.Module)
        super(UMDQN_W_Model_Atari, self).__init__()

        # Initialization of the Deep Neural Network
        self.stateEmbeddingDNN = DNN_Atari(numberOfInputs, stateEmbedding)
        self.UMNN = MonotonicNN(stateEmbedding+1, structureUMNN, numberOfSteps, numberOfOutputs, device)

    
    def forward(self, state, taus):
        """
        GOAL: Implementing the forward pass of the Deep Neural Network.
        
        INPUTS: - state: RL state.
                - taus: Samples of taus.
        
        OUTPUTS: - output: Output of the Deep Neural Network.
        """
        
        # State embedding part of the Deep Neural Network
        batchSize = state.size(0)
        x = self.stateEmbeddingDNN(state)
        x = x.repeat(1, int(len(taus)/len(state))).view(-1, x.size(1))

        # UMNNN part of the Deep Neural Network
        x = self.UMNN(taus, x)

        # Appropriate format
        return torch.cat(torch.chunk(torch.transpose(x, 0, 1), batchSize, dim=1), 0)
