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
   not-0), we separate that column from the dataset.
   When we train the SOM, we only use X becuase this is unsupervised learning, a.k.a no dependent
   variable is considered.
'''

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) # normalisation- all features between 0 and 1
X = sc.fit_transform(X)
'''in most cases, feature scaling is compulsory for deep learning becuase they're high computations
   to make, since we start with a high dimensional dataset with lots of non-linear relationships and it 
   will be much easier for the model  to be trained if features are scaled.
'''

# Training the SOM
from minisom import MiniSom # the other python file containing the class
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5) 
'''10x10 grid, 15 features in the dataset, sigma-radius of neighbourhood, 0.5-by how much
   the weights are updated during each iteration, the higher, the faster there will be convergence
   the lower, the slower it will be built '''
som.random_weights_init(X) # initialise weights
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone() # the window that will contain the map
pcolor(som.distance_map().T) # all the values of the MID for all the winning nodes of the SOM 
# the T takes the transpose of this MID matrix 
colorbar() # leyend of colours
'''The white squares represent where the frauds are. Frauds are identified by outlying winning nodes,
   the ones that are far from the general rules.
   We want to find the customers who are a fraud but got approved (class 1)'''
markers = ['o', 's'] # o - circle, s - square
colors = ['r', 'g'] # red, green
for i, x in enumerate(X): # 2 looping variables (i and x)
    '''i - the different values of all the indexes of dataset 0 - 689 //rows
       x - the different vectors of customers (first customer, then second customer, etc...) //columns
       enumerate(X) - X is the dataset 
    '''
    w = som.winner(x) # winning node of customer x
    plot(w[0] + 0.5, # x coordinate of winning node, +0.5 to put it in the centre of the square
         w[1] + 0.5, # y coordinate of winning node
         markers[y[i]], # the y variable contains the column with the classes
         #  y - index of customer
         #  y[i] - value of dependent variable (0 or 1)
         #  if class in y == 0: 
         #      y[i] == 0
         #      markers[y[i]] = 0 == circle
         #  if class in y == 1:
         #      y[i] == 1
         #      markers[y[i]] = 1 = square
         markeredgecolor = colors[y[i]], # marker edge colour
         markerfacecolor = 'None', # marker face (inside) colour
         markersize = 10,
         markeredgewidth = 2,) # marker edge width
plt.title('Red circles -ve class, Green squares +ve class')
#plt.xlabel('Time')
#plt.ylabel('Google Stock Price')
show()


# Finding the frauds
mappings = som.win_map(X) # the data on which the SOM was trained
frauds = np.concatenate((mappings[(2,8)], mappings[(8,6)]), axis = 0) # get the 2 white spots and add lists to get list of all possible frauds
frauds = sc.inverse_transform(frauds) # inverse scaling, since values were scaled before 
# the first column in the frauds list is the customer ID of the possible fraud customers