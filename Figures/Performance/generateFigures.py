# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import pandas as pd
from matplotlib import pyplot as plt

from matplotlib import rc
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}
rc('font', **font)
rc('text', usetex=True)



###############################################################################
###################### Script for generating the figures ######################
###############################################################################

# Setting of the parameters for processing data and generating figures
period = 300
numberOfPointsTraining = 10000
size = (10, 6)
colours = ['tab:red', 'tab:blue', 'tab:green', 'tab:grey', 'tab:cyan', 'tab:orange', 'tab:brown', 'tab:purple']
alpha1 = 0.75
alpha2 = 0.25

# Loading of the data
dataUMDQNKLTraining = pd.read_csv('UMDQN_KL/TrainingResults.csv')
dataUMDQNCTraining = pd.read_csv('UMDQN_C/TrainingResults.csv')
dataUMDQNWTraining = pd.read_csv('UMDQN_W/TrainingResults.csv')
dataUMDQNKLTesting = pd.read_csv('UMDQN_KL/TestingResults.csv')
dataUMDQNCTesting = pd.read_csv('UMDQN_C/TestingResults.csv')
dataUMDQNWTesting = pd.read_csv('UMDQN_W/TestingResults.csv')
dataCDQNTraining = pd.read_csv('CDQN/TrainingResults.csv')
dataCDQNTesting = pd.read_csv('CDQN/TestingResults.csv')

# Smoothing of the results for better readibility
dataUMDQNKLTraining = dataUMDQNKLTraining.rolling(period, min_periods=1).mean()
dataUMDQNCTraining = dataUMDQNCTraining.rolling(period, min_periods=1).mean()
dataUMDQNWTraining = dataUMDQNWTraining.rolling(period, min_periods=1).mean()
dataUMDQNKLTesting = dataUMDQNKLTesting.rolling(period, min_periods=1).mean()
dataUMDQNCTesting = dataUMDQNCTesting.rolling(period, min_periods=1).mean()
dataUMDQNWTesting = dataUMDQNWTesting.rolling(period, min_periods=1).mean()
dataCDQNTraining = dataCDQNTraining.rolling(period, min_periods=1).mean()
dataCDQNTesting = dataCDQNTesting.rolling(period, min_periods=1).mean()

# Selection of the number of points to plot (training and testing)
dataUMDQNKLTraining = dataUMDQNKLTraining[:numberOfPointsTraining]
dataUMDQNCTraining = dataUMDQNCTraining[:numberOfPointsTraining]
dataUMDQNWTraining = dataUMDQNWTraining[:numberOfPointsTraining]
numberOfPointsTesting = numberOfPointsTraining
dataUMDQNKLTesting = dataUMDQNKLTesting[:numberOfPointsTesting]
dataUMDQNCTesting = dataUMDQNCTesting[:numberOfPointsTesting]
dataUMDQNWTesting = dataUMDQNWTesting[:numberOfPointsTesting]
dataCDQNTraining = dataCDQNTraining[:numberOfPointsTraining]
dataCDQNTesting = dataCDQNTesting[:numberOfPointsTesting]

# Generation of the figure for training
fig = plt.figure(figsize=size)
ax = fig.add_subplot(111, ylabel='Cumulative reward', xlabel='Episode')
expectation = dataUMDQNKLTraining['Expectation']
std = dataUMDQNKLTraining['StandardDeviation']
ax.plot(expectation, color=colours[0], alpha=alpha1)
ax.fill_between(range(len(expectation)), expectation-std, expectation+std, alpha=alpha2, color=colours[0])
expectation = dataUMDQNCTraining['Expectation']
std = dataUMDQNCTraining['StandardDeviation']
ax.plot(expectation, color=colours[1], alpha=alpha1)
ax.fill_between(range(len(expectation)), expectation-std, expectation+std, alpha=alpha2, color=colours[1])
expectation = dataUMDQNWTraining['Expectation']
std = dataUMDQNWTraining['StandardDeviation']
ax.plot(expectation, color=colours[2], alpha=alpha1)
ax.fill_between(range(len(expectation)), expectation-std, expectation+std, alpha=alpha2, color=colours[2])
expectation = dataCDQNTraining['Expectation']
std = dataCDQNTraining['StandardDeviation']
ax.plot(expectation, color=colours[3], alpha=alpha1, linestyle='dashed')
ax.legend(['UMDQN-KL', 'UMDQN-C', 'UMDQN-W', 'CDQN'])
ax.set(ylim=(-0.3, 1.1))
plt.savefig('TrainingPerformance.pdf', format='pdf')

# Generation of the figure for testing
fig = plt.figure(figsize=size)
ax = fig.add_subplot(111, ylabel='Cumulative reward', xlabel='Episode')
expectation = dataUMDQNKLTesting['Expectation']
std = dataUMDQNKLTesting['StandardDeviation']
ax.plot(expectation, color=colours[0], alpha=alpha1)
ax.fill_between(range(len(expectation)), expectation-std, expectation+std, alpha=alpha2, color=colours[0])
expectation = dataUMDQNCTesting['Expectation']
std = dataUMDQNCTesting['StandardDeviation']
ax.plot(expectation, color=colours[1], alpha=alpha1)
ax.fill_between(range(len(expectation)), expectation-std, expectation+std, alpha=alpha2, color=colours[1])
expectation = dataUMDQNWTesting['Expectation']
std = dataUMDQNWTesting['StandardDeviation']
ax.plot(expectation, color=colours[2], alpha=alpha1)
ax.fill_between(range(len(expectation)), expectation-std, expectation+std, alpha=alpha2, color=colours[2])
expectation = dataCDQNTesting['Expectation']
std = dataCDQNTesting['StandardDeviation']
ax.plot(expectation, color=colours[3], alpha=alpha1, linestyle='dashed')
ax.legend(['UMDQN-KL', 'UMDQN-C', 'UMDQN-W', 'CDQN'])
ax.set(ylim=(-0.1, 1.1))
plt.savefig('TestingPerformance.pdf', format='pdf')
