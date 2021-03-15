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
import matplotlib.lines as mlines
import signal

# Setting termination signal handler
 
signal.signal(signal.SIGINT, lambda x, y: exit(0))
 
# Propose random sample function
def proprnd(theta, S, bounds):
 id=1;
 while id == 1:
  thetac = np.random.multivariate_normal(theta, S);
  if (len(theta)==1):
   if (thetac < bounds[0] or thetac > bounds[1]):
    id=1;
   else:
    id=0;
  else:
    if (np.any(thetac < bounds[0,:]) or np.any(thetac > bounds[1,:])):
      id=1;
    else:
      id=0;
 return thetac
       
parser = argparse.ArgumentParser()
parser.add_argument(
    '--N',
    help='Number of MCMC steps to run',
    default=1000,
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
    '--waitTime',
    help='Pause between samples (in seconds)',
    default=0.03,
    required=False)    
args = parser.parse_args()


# Set feasible parameter space
DataNth = 2
DataUnifbounds = np.concatenate((-10.0 * np.ones((1, DataNth)), 10.0 * np.ones((1, DataNth))));

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

fig = plt.figure(figsize=[12.8, 9.6])
ax = fig.add_subplot(111)
c = ax.contourf(Xv, Yv, Z, conts)
ax.set_xlabel(r'${\theta}_1$')
ax.set_ylabel(r'${\theta}_2$')  
cbar = fig.colorbar(c, ax=ax)
cbar.set_label('Posterior', rotation=270)

## Start running proposals

# Covariance of jumping Gaussian - try varying this and looking at the proportion of rejections/acceptances
jump_sigma = np.square(np.array(([1/3, 0], [0, 5/3]))) 
Naccept = 0;

# Initialize x
x = np.array((args.initialX, args.initialY));
MHSamples = [ x ];
MHValues = [ np.sum(-0.5*(x-means)*np.linalg.inv(covMat)*(x-means)) ]

ax.plot(x[0], x[1], '*',markersize=10, color='k');

legend_elements = [
 mlines.Line2D([], [], color='black', marker='*', linewidth=0, markersize=13, label='Initial Guess'),
 mlines.Line2D([], [], color='black', marker='o', markersize=10, label='Accepted Sample'),
 mlines.Line2D([], [], color='red', linestyle='--', marker='x', markersize=10, label='Rejected Sample')
 ]

ax.legend(handles=legend_elements, loc='lower left', ncol=1, fontsize=10)
ax.set_ylim([-5, 11])
ax.set_xlim([-13, 13])
plt.draw() 
 
## Now iterating over N MCMC cycles
for i in range(args.N):

 # Make a small pause before next sample
 plt.pause(args.waitTime)

 # Propose a new value
 xs = proprnd(x, jump_sigma, DataUnifbounds);

 # Using a Gaussian jump, jump ratios cancel
 # Compute ratio of densities (done in log space, constants cancel)
  
 pnew = np.sum(-0.5*(xs-means)*np.linalg.inv(covMat)*(xs-means))
 pold = np.sum(-0.5*(x-means)*np.linalg.inv(covMat)*(x-means))

 r = np.random.rand() 
 v = np.exp(pnew - pold)
 
 if r <= np.exp(pnew - pold): # Accept the sample
  ax.plot(xs[0], xs[1], 'o', markersize=5, color='k')
  ax.plot([ x[0], xs[0] ], [ x[1], xs[1] ], '-', color=[0.6, 0.6, 0.6])
  x = xs
  Naccept = Naccept + 1
 else: # Reject the sample
  ax.plot(xs[0], xs[1], 'x', markersize=5, color='r')
  ax.plot([ x[0], xs[0] ], [ x[1], xs[1] ], '--', color=[1.0, 0.6, 0.6])

 MHSamples.append(x);
 MHValues.append(pold);
  
 step = i+1
 acceptanceRatio = 100.0*float(Naccept)/float(step)
 plt.title('Current Acceptance Ratio: ' + "{:.3f}".format(acceptanceRatio) + '% at Step ' + str(step))
