# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import argparse
import importlib
import gym

from CustomEnvironments.stochasticGridWorld import StochasticGridWorld
from CustomEnvironments.stochasticGridWorldOptimal import StochasticGridWorldOptimal
from MonteCarloDistributions import MonteCarloDistributions
from AtariWrapper import AtariWrapper, MinAtarWrapper



###############################################################################
################################ Global variables #############################
###############################################################################

# Supported RL algorithms
algorithms = ['DQN', 'CDQN', 'QR_DQN', 'IQN', 'FQF',
              'UMDQN_KL', 'UMDQN_C', 'UMDQN_W']

# Supported RL environments
environments = ['StochasticGridWorld', 'CartPole-v0', 'Acrobot-v1',
                'LunarLander-v2', 'MountainCar-v0', 'MinAtar/Asterix-v0',
                'MinAtar/Breakout-v0', 'MinAtar/Freeway-v0', 'MinAtar/Seaquest-v0',
                'MinAtar/SpaceInvaders-v0', 'PongNoFrameskip-v4',
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
    episodes = int(args.episodes)
    parameters = args.parameters
    if algorithm not in algorithms:
        print("The algorithm specified is not valid, only the following algorithms are supported:")
        for algo in algorithms:
            print("".join(['- ', algo]))
    if environment not in environments:
        print("The environment specified is not valid, only the following environments are supported:")
        for env in environments:
            print("".join(['- ', env]))
    if parameters == 'parameters':
        parameters = ''.join(['Parameters/parameters_', str(algorithm), '_', str(environment), '.json'])
    
    # Name of the file for saving the RL policy learned
    fileName = 'SavedModels/' + algorithm + '_' + environment
    
    # Initialization of the RL environment
    if environment == 'StochasticGridWorld':
        env = StochasticGridWorld()
    elif environment in ['CartPole-v0', 'Acrobot-v1', 'LunarLander-v2', 'MountainCar-v0']:
        env = gym.make(environment)
        parameters = ''.join(['Parameters/parameters_', algorithm, '_ClassicControl.json'])
    elif environment in ['MinAtar/Asterix-v0','MinAtar/Breakout-v0', 'MinAtar/Freeway-v0', 'MinAtar/Seaquest-v0', 'MinAtar/SpaceInvaders-v0']:
        minAtarWrapper = MinAtarWrapper()
        env = minAtarWrapper.wrapper(environment)
        parameters = ''.join(['Parameters/parameters_', algorithm, '_MinAtar.json'])
    else:
        atariWrapper = AtariWrapper()
        env = atariWrapper.wrapper(environment, stickyActionsProba=0.25)
        parameters = ''.join(['Parameters/parameters_', algorithm, '_Atari57.json'])

    # Determination of the state and action spaces
    observationSpace = env.observation_space.shape[0]
    actionSpace = env.action_space.n

    # Initialization of the DRL algorithm
    algorithmModule = importlib.import_module(str(algorithm))
    className = getattr(algorithmModule, algorithm)
    RLAgent = className(observationSpace, actionSpace, environment, parameters)

    # Training of the RL agent
    RLAgent.training(env, episodes, verbose=False, rendering=False, plotTraining=False)
    #RLAgent.plotExpectedPerformance(env, episodes, iterations=5)
    
    # Saving of the RL model
    RLAgent.saveModel(fileName)

    # Loading of the RL model
    RLAgent.loadModel(fileName)

    # Testing of the RL agent
    RLAgent.testing(env, verbose=True, rendering=False)

    # Plotting of the true distribution of the random return via Monte Carlo
    """
    state = [int(7/2)-1, 7-1]
    optimalPolicy = StochasticGridWorldOptimal(env)
    MonteCarloDistributions = MonteCarloDistributions(env, optimalPolicy, 0.5)
    #MonteCarloDistributions = MonteCarloDistributions(env, RLAgent, 0.5)
    MonteCarloDistributions.plotDistributions(state)
    """
