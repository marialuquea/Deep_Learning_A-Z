# Boltzmann Machines

# Importing the libraries
import numpy as np # to work with arrays
import pandas as pd # to import datasets and create training/test set
import torch
import torch.nn as nn # used to implement neural networks
import torch.nn.parallel # parallel computations
import torch.optim as optim # for the optimizer
import torch.utils.data # tools that will be used
from torch.autograd import Variable # stochastic gradient descent

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
'''latin-1 encoding because some of the movie titles contain special characters that cannot be treated properly 
   with the classic encoding UTF-8'''

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t') # \t = tab
training_set = np.array(training_set, dtype = 'int') # dataframe to numpy array, data type int 
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = [] # a list of lists, each list for each user with its ratings
    for id_users in range(1, nb_users + 1): # the last one in the range is excluded, hence the +1
        id_movies = data[:,1][data[:,0] == id_users] # explained below
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies) # the movies that haven't been seen don't have an int, create list of 0s and change them if the movie has been rated
        ratings[id_movies - 1] = id_ratings 
        new_data.append(list(ratings)) # torch expects a list of lists
    return new_data
'''
    id_movies = data[:,1][data[:,0] == id_users]
        data[:,1] to get all the rows from the column index 1
        column index 1 contains the userID and the column 2 one of that user's rating
        we need to make a list of all the movies that the user has seen therefore
        
        [data[:,0] == id_users] 
        this line is a new condition
        the first column of data (index 0, the userID) == id_users 
'''
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors (similar to Numpy but used by PyTorch)
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1 # all the 0 values in the training set will now be a -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1
'''why convert ratings? because we are predicting binary ratings. 
   inputs must have the binary format the RBM will take the input vector, and inside the input
   vector it will predict the ratings for the movies that were not orginally rated by the user.
   Since the predicted ratings are computed originally from the existing ratings of the input
   vector, the predicted ratings of the outputmust have the same format as the existing ratings 
   in the input.
   Otherwise things would be inconsistent for the RBM.
   Inexistent ratings (0) will now become a -1
   1/2 -> not liked
   3/4/5 -> liked
'''

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))