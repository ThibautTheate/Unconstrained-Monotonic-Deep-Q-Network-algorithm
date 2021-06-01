# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import argparse
import importlib

from CustomEnvironments.stochasticGridWorld import StochasticGridWorld
from CustomEnvironments.stochasticGridWorldOptimal import StochasticGridWorldOptimal
from MonteCarloDistributions import MonteCarloDistributions
from AtariWrapper import AtariWrapper



###############################################################################
################################ Global variables #############################
###############################################################################

# Supported RL algorithms
algorithms = ['DQN', 'CDQN', 'QR_DQN', 'IQN', 'FQF',
              'UMDQN_KL', 'UMDQN_C', 'UMDQN_W']

# Supported RL environments
environments = ['StochasticGridWorld', 'PongNoFrameskip-v4',
                'BoxingNoFrameskip-v4', 'FreewayNoFrameskip-v4']



###############################################################################
##################################### MAIN ####################################
###############################################################################

if(__name__ == '__main__'):

    # Retrieve the paramaters sent by the user
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-algorithm", default='UMDQN_C', type=str, help="Name of the RL algorithm")
    parser.add_argument("-environment", default='StochasticGridWorld', type=str, help="Name of the RL environment")
    parser.add_argument("-episodes", default=10000, type=str, help="Number of episodes for training")
    parser.add_argument("-parameters", default='parameters', type=str, help="Name of the JSON parameters file")
    args = parser.parse_args()

    # Checking of the parameters validity
    algorithm = args.algorithm
    environment = args.environment
    episodes = args.episodes
    parameters = args.parameters
    if algorithm not in algorithms:
        print("The algorithm specified is not valid, only the following algorithms are supported:")
        for algo in algorithms:
            print("".join(['- ', algo]))
        quit()
    if environment not in environments:
        print("The environment specified is not valid, only the following environments are supported:")
        for env in environments:
            print("".join(['- ', env]))
        quit()
    if parameters == 'parameters':
        parameters = ''.join(['Parameters/parameters_', str(algorithm), '_', str(environment), '.json'])
    
    # Name of the file for saving the RL policy learned
    fileName = 'SavedModels/' + algorithm + '_' + environment
    
    # Initialization of the RL environment
    if environment == 'StochasticGridWorld':
        env = StochasticGridWorld()
    else:
        atariWrapper = AtariWrapper()
        env = atariWrapper.wrapper(environment, stickyActionsProba=0.25)
        fileName = 'SavedModels/' + algorithm + '_Atari57'
        parameters = ''.join(['Parameters/parameters_', algorithm, '_Atari57.json'])

    # Determination of the state and action spaces
    observationSpace = env.observation_space.shape[0]
    actionSpace = env.action_space.n

    # Initialization of the DRL algorithm
    algorithmModule = importlib.import_module(str(algorithm))
    className = getattr(algorithmModule, algorithm)
    RLAgent = className(observationSpace, actionSpace, environment, parameters)

    # Training of the RL agent
    RLAgent.training(env, episodes, verbose=True, rendering=False, plotTraining=False)
    #RLAgent.plotExpectedPerformance(env, episodes, iterations=10)
    
    # Saving of the RL model
    RLAgent.saveModel(fileName)

    # Loading of the RL model
    RLAgent.loadModel(fileName)

    # Testing of the RL agent
    RLAgent.testing(env, verbose=True, rendering=True)

    # Plotting of the true distribution of the random return via Monte Carlo
    """
    if environment == 'StochasticGridWorld':
        state = [int(7/2)-1, 7-1]
        optimalPolicy = StochasticGridWorldOptimal(env)
        MonteCarloDistributions = MonteCarloDistributions(env, optimalPolicy, 0.5)
        #MonteCarloDistributions = MonteCarloDistributions(env, RLAgent, 0.5)
        MonteCarloDistributions.plotDistributions(state)
    """
