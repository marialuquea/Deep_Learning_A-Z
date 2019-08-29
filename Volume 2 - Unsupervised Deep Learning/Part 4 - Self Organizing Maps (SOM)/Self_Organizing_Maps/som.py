# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''In the dataset, all the customers are the input of the NN, they are going to be mapped
   to a new output space. Between the input/output space we have this NN composed of neurons, 
   each neuron being initialised as a vector of weights that is the same size as the vector of 
   customer, a vector of 15 elements (customer ID + 14 attributes) and for each observation 
   point/customer, the output of the customer will be the neuron that is the closest to the customer.
   Bascially in the neuron we pick the cluster that is closest to the customer. This neuron is called
   the winning node. For each customer, the winning node is the most similar neuron to the customer.
   Then we use a neighbour function to update the weights of the neighbours to move them closer
   to the point .
   This is done for all customers in the input space, and everytime it is done the output space
   decreases and loses dimensions until it reaches a point where the output space stops decreasing.
   That's the moment where the 2D map is obtained with all the winning nodes that were eventually
   identified.
   
   MID - Mean Interneuron Distance, for each neuron, the mean of the Euclidean distance will be 
   computed between this neuron and the neurons in its neighbourhood. We define the neighbourhood
   manually for each neuron. By doing this, we detect outliers because they will be far from all
   the neurons in its neighbourhood.
   Then, an inverse mapping function will identify which customers originally in the input space are associated
   to this winning node, the outlier.
'''

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values #all rows, all columns except the last one - the class
y = dataset.iloc[:, -1].values #all rows, only the last column
'''Since the last attribute of the dataset is the class (if the customer's application was succesful-1 or
   not-0), we separate that column from the dataset
'''

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
plt.title('Red circles -ve class, Green squares +ve class')
#plt.xlabel('Time')
#plt.ylabel('Google Stock Price')
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)