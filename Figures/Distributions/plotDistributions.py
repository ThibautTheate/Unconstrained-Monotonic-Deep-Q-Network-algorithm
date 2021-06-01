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

# Setting of some paramters for the generation of figures
figsize = (10, 6)
colors = ['blue', 'red', 'orange', 'green']
alpha = 0.75
lw=2

# Loading of the data from csv files
FolderName = 'DistributionsVisualization/'
MonteCarloPDF = pd.read_csv('DistributionsVisualization/MonteCarloPDF.csv')
MonteCarloCDF = pd.read_csv('DistributionsVisualization/MonteCarloCDF.csv')
MonteCarloQF = pd.read_csv('DistributionsVisualization/MonteCarloQF.csv')
LearntPDF = pd.read_csv('DistributionsVisualization/UMDQN_KL.csv')
LearntCDF = pd.read_csv('DistributionsVisualization/UMDQN_C.csv')
LearntQF = pd.read_csv('DistributionsVisualization/UMDQN_W.csv')

# Plotting of the distributions (PDF, CDF and QF)
# PDF
fig = plt.figure(figsize=figsize)
ax1 = plt.subplot(1, 1, 1)
ax1.plot(MonteCarloPDF['Action0_x'], MonteCarloPDF['Action0_y'], color=colors[0], linestyle='dotted', alpha=alpha, lw=lw)
ax1.plot(MonteCarloPDF['Action1_x'], MonteCarloPDF['Action1_y'], color=colors[1], linestyle='dotted', alpha=alpha, lw=lw)
ax1.plot(MonteCarloPDF['Action2_x'], MonteCarloPDF['Action2_y'], color=colors[2], linestyle='dotted', alpha=alpha, lw=lw)
ax1.plot(MonteCarloPDF['Action3_x'], MonteCarloPDF['Action3_y'], color=colors[3], linestyle='dotted', alpha=alpha, lw=lw)
ax1.plot(LearntPDF['Action0_x'], LearntPDF['Action0_y'], color=colors[0], alpha=alpha, lw=lw)
ax1.plot(LearntPDF['Action1_x'], LearntPDF['Action1_y'], color=colors[1], alpha=alpha, lw=lw)
ax1.plot(LearntPDF['Action2_x'], LearntPDF['Action2_y'], color=colors[2], alpha=alpha, lw=lw)
ax1.plot(LearntPDF['Action3_x'], LearntPDF['Action3_y'], color=colors[3], alpha=alpha, lw=lw)
ax1.set_xlabel('Random return')
ax1.set_ylabel('PDF')
ax1.set(xlim=(-0.5, 1.5), ylim=(0, 3.5))
plt.savefig("RandomReturnPDF.pdf", format='pdf', dpi=1200)
# CDF
fig = plt.figure(figsize=figsize)
ax2 = plt.subplot(1, 1, 1)
ax2.plot(MonteCarloCDF['Action0_x'], MonteCarloCDF['Action0_y'], color=colors[0], linestyle='dotted', alpha=alpha, lw=lw)
ax2.plot(MonteCarloCDF['Action1_x'], MonteCarloCDF['Action1_y'], color=colors[1], linestyle='dotted', alpha=alpha, lw=lw)
ax2.plot(MonteCarloCDF['Action2_x'], MonteCarloCDF['Action2_y'], color=colors[2], linestyle='dotted', alpha=alpha, lw=lw)
ax2.plot(MonteCarloCDF['Action3_x'], MonteCarloCDF['Action3_y'], color=colors[3], linestyle='dotted', alpha=alpha, lw=lw)
ax2.plot(LearntCDF['Action0_x'], LearntCDF['Action0_y'], color=colors[0], alpha=alpha, lw=lw)
ax2.plot(LearntCDF['Action1_x'], LearntCDF['Action1_y'], color=colors[1], alpha=alpha, lw=lw)
ax2.plot(LearntCDF['Action2_x'], LearntCDF['Action2_y'], color=colors[2], alpha=alpha, lw=lw)
ax2.plot(LearntCDF['Action3_x'], LearntCDF['Action3_y'], color=colors[3], alpha=alpha, lw=lw)
ax2.set_xlabel('Random return')
ax2.set_ylabel('CDF')
ax2.set(xlim=(-0.5, 1.5), ylim=(-0.1, 1.1))
plt.savefig("RandomReturnCDF.pdf", format='pdf', dpi=1200)
# QF
fig = plt.figure(figsize=figsize)
ax3 = plt.subplot(1, 1, 1)
ax3.plot(MonteCarloQF['Action0_x'], MonteCarloQF['Action0_y'], color=colors[0], linestyle='dotted', alpha=alpha, lw=lw)
ax3.plot(MonteCarloQF['Action1_x'], MonteCarloQF['Action1_y'], color=colors[1], linestyle='dotted', alpha=alpha, lw=lw)
ax3.plot(MonteCarloQF['Action2_x'], MonteCarloQF['Action2_y'], color=colors[2], linestyle='dotted', alpha=alpha, lw=lw)
ax3.plot(MonteCarloQF['Action3_x'], MonteCarloQF['Action3_y'], color=colors[3], linestyle='dotted', alpha=alpha, lw=lw)
ax3.plot(LearntQF['Action0_x'], LearntQF['Action0_y'], color=colors[0], alpha=alpha, lw=lw)
ax3.plot(LearntQF['Action1_x'], LearntQF['Action1_y'], color=colors[1], alpha=alpha, lw=lw)
ax3.plot(LearntQF['Action2_x'], LearntQF['Action2_y'], color=colors[2], alpha=alpha, lw=lw)
ax3.plot(LearntQF['Action3_x'], LearntQF['Action3_y'], color=colors[3], alpha=alpha, lw=lw)
ax3.set_xlabel('Quantile fraction')
ax3.set_ylabel('QF')
ax3.set(xlim=(0.0, 1.0), ylim=(-0.5, 1.5))
plt.savefig("RandomReturnQF.pdf", format='pdf', dpi=1200)
# Legend
fig = plt.figure(figsize=figsize)
ax4 = plt.subplot(1, 1, 1)
ax4.plot(LearntQF['Action0_x'], LearntQF['Action0_y'], color=colors[0], alpha=alpha, lw=lw)
ax4.plot(LearntQF['Action1_x'], LearntQF['Action1_y'], color=colors[1], alpha=alpha, lw=lw)
ax4.plot(LearntQF['Action2_x'], LearntQF['Action2_y'], color=colors[2], alpha=alpha, lw=lw)
ax4.plot(LearntQF['Action3_x'], LearntQF['Action3_y'], color=colors[3], alpha=alpha, lw=lw)
ax4.set_xlabel('Quantile fraction')
ax4.set_ylabel('QF')
ax4.set(xlim=(0.0, 1.0), ylim=(10, 20))
ax4.legend(['Move right', 'Move down', 'Move left', 'Move up'])
plt.savefig("Legend.pdf", format='pdf', dpi=1200)
