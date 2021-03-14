#! /usr/bin/env python3

# SHOWCASING SIMPLE MH-MCMC with Gaussian Proposals
# Adapted script from From A First Course in Machine Learning, Chapter 4. Simon Rogers, 01/11/11 
# [simon.rogers@glasgow.ac.uk] Example of Metropolis-Hastings
# Adapted from a MATLAB version by P. Angelikopoulos pangelik@inf.ethz.ch
# And into Python by Sergio Martin martiser@ethz.ch

import os
import matplotlib.pyplot as plt
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
    default=0.0,
    required=False)
parser.add_argument(
    '--initialY',
    help='Initial Y position',
    default=0.0,
    required=False)
parser.add_argument(
    '--waiting',
    help='Pause between samples (in seconds)',
    default=0.5,
    required=False)    
args = parser.parse_args()


