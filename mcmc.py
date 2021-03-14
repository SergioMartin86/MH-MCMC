#! /usr/bin/env python3

# SHOWCASING SIMPLE MH-MCMC with Gaussian Proposals
# Adapted script from From A First Course in Machine Learning, Chapter 4. Simon Rogers, 01/11/11 
# [simon.rogers@glasgow.ac.uk] Example of Metropolis-Hastings
# Adapted from a MATLAB version by P. Angelikopoulos pangelik@inf.ethz.ch
# and into Python by Sergio Martin martiser@ethz.ch

import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--N',
    help='Number of MCMC iterations to run',
    default=1000,
    required=False)
parser.add_argument(
    '--BurnIn',
    help='Number of burn in samples to run',
    default=50,
    required=False)
parser.add_argument(
    '--initialX',
    help='Initial X position',
    default=5.0,
    required=False)
parser.add_argument(
    '--initialY',
    help='Initial Y position',
    default=10.0,
    required=False)
parser.add_argument(
    '--waiting',
    help='Pause between samples (in seconds)',
    default=0.5,
    required=False)    
args = parser.parse_args()


# Setting Gaussian Mean + Covariance matrix for the proposals
means = np.array([ 0.0, 0.0 ])
covMat = np.array([ [ 25.0, 0.0 ], [ 0.0, 1.0 ] ])

# Creating mesh grid
x = np.linspace(-20, 20, 200)
y = x
[Xv, Yv] = np.meshgrid(x,y)

# Pre-calculating constants for posterior plotting
c0 = -np.log(2.0*np.pi) - np.log(np.linalg.det(covMat))
v0 = np.subtract(Xv.flatten('F'), means[0])
v1 = np.subtract(Yv.flatten('F'), means[1])
m0 = np.array([v0, v1])
m1 = np.linalg.solve(covMat, m0)

mMult = np.multiply(np.transpose(m1), np.transpose(m0))
mAdd = np.sum(mMult, axis=1)
Probs = c0 - 0.5*(mAdd);
Z = np.transpose(np.reshape(np.exp(Probs), Xv.shape))
peak = np.max(Z)
conts=[0.05*peak,0.10*peak,0.25*peak,0.50*peak,0.75*peak,0.95*peak];

## Plotting contours
#fig = plt.figure()
#ax = fig.add_subplot(111)
#c = ax.contourf(Xv, Yv, Z, conts)
#ax.set_xlabel(r'${\theta}_1$')
#ax.set_ylabel(r'${\theta}_2$')  
#cbar = fig.colorbar(c, ax=ax)
#cbar.set_label('Posterior', rotation=270)

# Start running proposals

# Covariance of jumping Gaussian - try varying this and looking at the proportion of rejections/acceptances
jump_sigma = np.square(np.array(([1/3, 0], [0, 5/3]))) 
Naccept = 0;

# Initialize x
x = np.array((args.initialX, args.initialY));
MHSamples = x;

MHValA = -0.5*(x-means)
MHValB = np.linalg.inv(covMat)
MHValC = x-means
MHValue = np.sum(MHValA*MHValB*MHValC)

print(MHSamples)
print(MHValue)
#plt.show() 