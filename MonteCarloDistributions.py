# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import rc
rc('text', usetex=True)



###############################################################################
################################ Global variables #############################
###############################################################################

# Default parameters for the plotting of the distributions
numberOfSamples = 10000000
bins = 1000
density = True
plotRange = (-2.1, 2.1)
histtype = 'step'



###############################################################################
####################### Class MonteCarloDistributions #########################
###############################################################################

class MonteCarloDistributions():
    """
    GOAL: Implementing a technique based on Monte Carlo to estimate the true
          expected return associated with an environment and a policy.
    
    VARIABLES: - environment: Environment analysed.
               - policy: Policy analysed.
               - gamma: Discount factor.
                                
    METHODS: - __init__: Initialization of the class.
             - samplingMonteCarlo: Generate MC samples of the random return.
             - plotDistributions: PLot the distributions from the MC samples.
    """

    def __init__(self, environment, policy, gamma):
        """
        GOAL: Perform the initialization of the class.
        
        INPUTS: - environment: Environment analysed.
                - policy: Policy analysed.
                - gamma: Discount factor.
        
        OUTPUTS: /
        """

        # Initialization of important variables
        self.environment = environment
        self.policy = policy
        self.gamma = gamma

    
    def samplingMonteCarlo(self, initialState, initialAction, numberOfSamples=numberOfSamples):
        """
        GOAL: Collect Monte Carlo samples of the expected return associated
              with the state and action specified.
        
        INPUTS: - initialState: RL state to start from.
                - initialAction: RL action to start from.
                
                - numberOfSamples: Number of Monte Carlo samples to collect.
        
        OUTPUTS: - samples: Monte Carlo samples collected.
        """

        # Initialization of the memory storing the MC samples
        samples = []

        # Generation of the MC samples
        for _i in range(numberOfSamples):

            # Initialization of some variables
            expectedReturn = 0
            step = 0

            # Reset of the environment and initialization to the desired state
            self.environment.reset()
            state = self.environment.setState(initialState)

            # Execution of the action specified
            nextState, reward, done, info = self.environment.step(initialAction)

            # Update of the expected return
            expectedReturn += (reward * (self.gamma**step))
            step += 1

            # Loop until episode termination
            while done == 0:

                # Execute the next ation according to the policy selected
                state = self.policy.processState(nextState)
                policyAction = self.policy.chooseAction(state, plot=False)
                nextState, reward, done, info = self.environment.step(policyAction)

                # Update of the expected return
                expectedReturn += (reward * (self.gamma**step))
                step += 1
            
            # Add the MC sample to the memory
            samples.append(expectedReturn)

        # Output the MC samples collected
        return samples


    def plotDistributions(self, state, numberOfSamples=numberOfSamples):
        """
        GOAL: Collect Monte Carlo samples of the expected return associated
              with the state and action specified.
        
        INPUTS: - state: RL state to start from.
                - numberOfSamples: Number of Monte Carlo samples to collect.
        
        OUTPUTS: /
        """

        # Generation of the Monte Carlo samples
        samples = []
        actions = 4
        for action in range(actions):
            samples.append(self.samplingMonteCarlo(state, action, numberOfSamples))

        # Initialization of the figure
        colors = ['blue', 'red', 'orange', 'green', 'purple', 'brown']
        fig = plt.figure()
        
        # Plotting of the PDF of the random return
        ax1 = plt.subplot(3, 1, 1)
        for action in range(actions):
            plt.hist(samples[action], bins=bins, density=density, range=plotRange, histtype=histtype, color=colors[action])
        ax1.set_xlabel('Random return')
        ax1.set_ylabel('PDF')
        ax1.set(xlim=(-2, 2))

        # Plotting of the CDF of the random return
        ax2 = plt.subplot(3, 1, 2)
        for action in range(actions):
            plt.hist(samples[action], bins=bins, density=density, range=plotRange, histtype=histtype, cumulative=True, color=colors[action])
        ax2.set_xlabel('Random return')
        ax2.set_ylabel('CDF')
        ax2.set(xlim=(-2, 2))

        # Plotting of the QF of the random return
        ax3 = plt.subplot(3, 1, 3)
        CDF0 = plt.hist(samples[0], bins=bins, density=density, range=plotRange, histtype=histtype, cumulative=True, color='white')
        CDF1 = plt.hist(samples[1], bins=bins, density=density, range=plotRange, histtype=histtype, cumulative=True, color='white')
        CDF2 = plt.hist(samples[2], bins=bins, density=density, range=plotRange, histtype=histtype, cumulative=True, color='white')
        CDF3 = plt.hist(samples[3], bins=bins, density=density, range=plotRange, histtype=histtype, cumulative=True, color='white')
        ax3.clear()
        ax3.plot(CDF0[0], CDF0[1][1:], color=colors[0])
        ax3.plot(CDF1[0], CDF1[1][1:], color=colors[1])
        ax3.plot(CDF2[0], CDF2[1][1:], color=colors[2])
        ax3.plot(CDF3[0], CDF3[1][1:], color=colors[3])
        ax3.set_xlabel('Quantile fraction')
        ax3.set_ylabel('QF')
        ax3.set(xlim=(0, 1))
        ax3.legend(['Move right', 'Move down', 'Move left', 'Move up'])

        # Saving of the figure generated
        plt.savefig("Figures/Distributions/MonteCarloDistributions.pdf", format='pdf')

        # Generation of the figure for the PDF of the random return
        fig = plt.figure(figsize=(10, 4))
        ax1 = plt.subplot(1, 1, 1)
        for action in range(actions):
            plt.hist(samples[action], bins=bins, density=density, range=plotRange, histtype=histtype, color=colors[action])
        ax1.set_xlabel('Random return')
        ax1.set_ylabel('PDF')
        ax1.set(xlim=(-0.5, 1.5), ylim=(0, 3.5))
        plt.savefig("Figures/Distributions/MonteCarloDistributionsPDF.pdf", format='pdf')
        # Generation of the figure for the CDF of the random return
        fig = plt.figure(figsize=(10, 4))
        ax2 = plt.subplot(1, 1, 1)
        for action in range(actions):
            plt.hist(samples[action], bins=bins, density=density, range=plotRange, histtype=histtype, cumulative=True, color=colors[action])
        ax2.set_xlabel('Random return')
        ax2.set_ylabel('CDF')
        ax2.set(xlim=(-0.5, 1.5), ylim=(-0.1, 1.1))
        plt.savefig("Figures/Distributions/MonteCarloDistributionsCDF.pdf", format='pdf')
        # Generation of the figure for the QF of the random return
        fig = plt.figure(figsize=(10, 4))
        ax3 = plt.subplot(1, 1, 1)
        CDF0 = plt.hist(samples[0], bins=bins, density=density, range=plotRange, histtype=histtype, cumulative=True, color='white')
        CDF1 = plt.hist(samples[1], bins=bins, density=density, range=plotRange, histtype=histtype, cumulative=True, color='white')
        CDF2 = plt.hist(samples[2], bins=bins, density=density, range=plotRange, histtype=histtype, cumulative=True, color='white')
        CDF3 = plt.hist(samples[3], bins=bins, density=density, range=plotRange, histtype=histtype, cumulative=True, color='white')
        ax3.clear()
        ax3.plot(CDF0[0], CDF0[1][1:], color=colors[0])
        ax3.plot(CDF1[0], CDF1[1][1:], color=colors[1])
        ax3.plot(CDF2[0], CDF2[1][1:], color=colors[2])
        ax3.plot(CDF3[0], CDF3[1][1:], color=colors[3])
        ax3.set_xlabel('Quantile fraction')
        ax3.set_ylabel('QF')
        ax3.set(xlim=(0, 1), ylim=(-0.5, 1.5))
        plt.savefig("Figures/Distributions/MonteCarloDistributionsQF.pdf", format='pdf')

        # Saving of the data into external files
        PDF0 = plt.hist(samples[0], bins=bins, density=density, range=plotRange, histtype=histtype, color='white')
        PDF1 = plt.hist(samples[1], bins=bins, density=density, range=plotRange, histtype=histtype, color='white')
        PDF2 = plt.hist(samples[2], bins=bins, density=density, range=plotRange, histtype=histtype, color='white')
        PDF3 = plt.hist(samples[3], bins=bins, density=density, range=plotRange, histtype=histtype, color='white')
        CDF0 = plt.hist(samples[0], bins=bins, density=density, range=plotRange, histtype=histtype, cumulative=True, color='white')
        CDF1 = plt.hist(samples[1], bins=bins, density=density, range=plotRange, histtype=histtype, cumulative=True, color='white')
        CDF2 = plt.hist(samples[2], bins=bins, density=density, range=plotRange, histtype=histtype, cumulative=True, color='white')
        CDF3 = plt.hist(samples[3], bins=bins, density=density, range=plotRange, histtype=histtype, cumulative=True, color='white')
        dataPDF = {
            'Action0_x': PDF0[1][1:],
            'Action0_y': PDF0[0],
            'Action1_x': PDF1[1][1:],
            'Action1_y': PDF1[0],
            'Action2_x': PDF2[1][1:],
            'Action2_y': PDF2[0],
            'Action3_x': PDF3[1][1:],
            'Action3_y': PDF3[0],
        }
        dataCDF = {
            'Action0_x': CDF0[1][1:],
            'Action0_y': CDF0[0],
            'Action1_x': CDF1[1][1:],
            'Action1_y': CDF1[0],
            'Action2_x': CDF2[1][1:],
            'Action2_y': CDF2[0],
            'Action3_x': CDF3[1][1:],
            'Action3_y': CDF3[0],
        }
        dataQF = {
            'Action0_y': CDF0[1][1:],
            'Action0_x': CDF0[0],
            'Action1_y': CDF1[1][1:],
            'Action1_x': CDF1[0],
            'Action2_y': CDF2[1][1:],
            'Action2_x': CDF2[0],
            'Action3_y': CDF3[1][1:],
            'Action3_x': CDF3[0],
        }
        dataframePDF = pd.DataFrame(dataPDF)
        dataframeCDF = pd.DataFrame(dataCDF)
        dataframeQF = pd.DataFrame(dataQF)
        dataframePDF.to_csv('Figures/Distributions/MonteCarloPDF.csv')
        dataframeCDF.to_csv('Figures/Distributions/MonteCarloCDF.csv')
        dataframeQF.to_csv('Figures/Distributions/MonteCarloQF.csv')
