# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import torch
import torch.nn as nn
# pylint: disable=E1101
# pylint: disable=E1102

from Models.DNN_Atari import DNN_Atari
from Models.DNN_MinAtar import DNN_MinAtar
from Models.MonotonicNN import OneDimensionnalNF



###############################################################################
######################### Class UMDQN_KL_Model_Atari ##########################
###############################################################################

class UMDQN_KL_Model_Atari(nn.Module):
    """
    GOAL: Implementing the DL model for the UMDQN-KL distributional RL algorithm.
    
    VARIABLES:  - stateEmbeddingDNN: State embedding part of the Deep Neural Network.
                - UMNN: UMNN part of the Deep Neural Network.
                                
    METHODS:    - __init__: Initialization of the Deep Neural Network.
                - forward: Forward pass of the Deep Neural Network.
                - getExpectation: Get the expectation of the PDF approximated by the UMNN.
    """

    def __init__(self, numberOfInputs, numberOfOutputs,
                 structureUMNN, stateEmbedding, numberOfSteps,
                 device='cpu', minAtar=False):
        """
        GOAL: Defining and initializing the Deep Neural Network.
        
        INPUTS: - numberOfInputs: Number of inputs of the Deep Neural Network.
                - numberOfOutputs: Number of outputs of the Deep Neural Network.
                - structureUMNN: Structure of the UMNN for distribution representation.
                - stateEmbedding: Dimension of the state embedding.
                - numberOfSteps: Number of integration steps for the UMNN.
                - device: Hardware device (CPU or GPU).
                - minAtar: Boolean specifying whether the env is "MinAtar" or not.
        
        OUTPUTS: /
        """

        # Call the constructor of the parent class (Pytorch torch.nn.Module)
        super(UMDQN_KL_Model_Atari, self).__init__()

        # Initialization of the Deep Neural Network
        if minAtar:
            self.stateEmbeddingDNN = DNN_MinAtar(numberOfInputs, stateEmbedding)
        else:
            self.stateEmbeddingDNN = DNN_Atari(numberOfInputs, stateEmbedding)
        self.UMNN = OneDimensionnalNF(stateEmbedding+1, structureUMNN, numberOfSteps, numberOfOutputs, device)

    
    def forward(self, state, q):
        """
        GOAL: Implementing the forward pass of the Deep Neural Network.
        
        INPUTS: - state: RL state.
                - q: Samples of potential returns.
        
        OUTPUTS: - output: Output of the Deep Neural Network.
        """
        
        # State embedding part of the Deep Neural Network
        batchSize = state.size(0)
        x = self.stateEmbeddingDNN(state)
        x = x.repeat(1, int(len(q)/len(state))).view(-1, x.size(1))

        # UMNN part of the Deep Neural Network
        x = self.UMNN(q, x)

        # Formatting of the output and post processing operations
        x = torch.cat(torch.chunk(torch.transpose(x, 0, 1), batchSize, dim=1), 0)
        x = torch.exp(x)
        x = x.clamp(min=1e-6)

        return x


    def getExpectation(self, state, minReturn, maxReturn, numberOfPoints):
        """
        GOAL: Get the expectation of the PDF internally computed by the UMNN.
        
        INPUTS: - state: RL state.
                - minReturn: Minimum return.
                - maxReturn: Maximum return.
                - numberOfPoints: Number of points for the computations (accuracy).
        
        OUTPUTS: - expectation: Expectation computed.
        """

        # State embedding part of the Deep Neural Network
        state = self.stateEmbeddingDNN(state)

        # Computation of the expectation of the PDF internally computed by the UMNN
        expectation = self.UMNN.expectation(state, lambda x: x, minReturn, maxReturn, numberOfPoints)
        return expectation
