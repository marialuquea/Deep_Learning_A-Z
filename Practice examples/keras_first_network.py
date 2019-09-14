# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:15:00 2019

@author: Maria
"""

# neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]