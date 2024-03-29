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
'''
      many RBM models can be created and tested with different configurations (several numbers
      of hidden nodes because that's basically the main parameter)
      another parameter can be added to the RBM, like for example a learning rate to improve and tune the model
    '''
class RBM():
    def __init__(self, nv, nh): # default compulsory function, self is the object that will be created
        self.W = torch.randn(nh, nv) # initialises a tensor of size nh - nv according to a normal distribution, mean 0, variance 1
        self.a = torch.randn(1, nh) #bias for the probabilites of the hidden nodes given the visible nodes
        self.b = torch.randn(1, nv) #bias for the probs of the visible nodes given the hidden nodes
        '''
        Weights are going to be initialised in a torched tensor. These weights are all the 
        parameters of the probabilities of the visible nodes given the hidden nodes.
        Initilalised in a matrix of size NH and NV. Matrix is a torched tensor because pyTorch.
        Tensor: matrix of one single type
        There is some bias for the probaility of the visible node given the hidden node and viceversa.
        BIASES: create a vector of NH or NV element all initialised to some number that follow 
        a normal distribution. 
        1st dimension corresponds to batch, 2nd dimension corresponds to bias of the tensor
        This is becuase the functions that will be used in PyTorch can't accept a single input vector
        '''
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t()) # mm: product of 2 tensors, x and self.w.
        activation = wx + self.a.expand_as(wx) # activation function, the expands_as function adds a new dimension to the bias being added
        p_h_given_v = torch.sigmoid(activation) #probability that the hidden node will be activated according to the value of the visible node
        return p_h_given_v, torch.bernoulli(p_h_given_v) #the ith element of the vector in phgivenv is the probability that the ith hidden node is activated
        '''
        Sampling the hidden nodes according to the probabilities ph given v 
        ph given v is the sigmoid activation function
        needed because during training it will approximate the log likelihood gradient through Gibbs sampling.
        once we have the probability we can sample the activations of the hidden nodes
        If we have 100 hidden nodes, this function samples the activations of theses h, for each 
        of the 100 hidden nodes, it will activate them according to a certain probability computed 
        in this same function. For each hidden node the prob will be ph given v, P(hn == 1) given v, 
        this probability is equal to the activation function
        x - visible neurons v in the probabilities ph given v
        wx: product of w (vector of weights) times x (vector of visible neurons) 
        activation function = wx + bias of the nodes(a)
        For every deep learning model, what's inside the activation function is a linear
        function of the neurons, where the coefficients are the weights
        
        us probability to sample the activation of the hidden nodes
        '''
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
        '''same as before'''
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)).t() # adding the product of the probs that the hidden nodes equal 1 given the input vector v0
        # the probability that the hidden nodes equal onve given v0 is ph0
        # minus the torch product of the visible nodes obtained after k sampling 
        # this is the first update in the KCD algorithm that is in the contrastive divergence technique
        self.b += torch.sum((v0 - vk), 0) # update of weight B (B: bias of the probabilities pv_givenH)
        # the difference between the input vector of observations and the visible nodes after k sampling (v0 -vk), plus 0 to keep the format of a tensor of 2 dimensions
        self.a += torch.sum((ph0 - phk), 0) # bias of the probabilities ph given v
        # difference of the probabilities that the hidden nodes == 1 given the values of v0, the input vector of observations 
        # and the probabilities that the hidden nodes == 1 given the values of vk, the values of the visible nodes after k sampling
        '''
        v0: input vector containing ratings of all the movies by one user
        vk: visible nodes obtained after K sampling (after k round trips from the visible nodes
            to the hidden nodes first and then way back from the hidden nodes to the visible nodes)
            so visible nodes obtained after k iterations and k contrastive divergence
        ph0: vector of probabilities (at first iteration it equals 1 given the values of v0)
        phk: probabilities of the hidden nodes after k sampling given the values of the visible nodes vk
        
        update tensor of weights, then bias B and then bias A
        
        torch.mm to get the product of 2 tensors 
        .t() transpose function to make things mathematically correct 
        '''
        
nv = len(training_set[0]) # number os visible nodes = number of features in the data, it gets the lenfth of the first line of the dataset
nh = 100 # number of hidden nodes which corresponds to the no of features that we want to detect
batch_size = 100 # the algorithm is trained with the weights updated after several observations that will all go in a batch. There are 943 so batches of 100 will make the training go quick and this is good for the start
rbm = RBM(nv, nh) # new object of the class RBM with number of visible and hidden nodes

# Training the RBM
nb_epoch = 10 # because there are few observations and we only have binary value 0/1 so the convergence will be reached pretty fast
    '''make a for loop that will go through the 10 epochs and in each epoch, all the observations
       will go into the network and weights will be updated after the observations of each batch 
       passed through the network.
       In the end we will get the final visible node with the new ratings for the movies that 
       were not originally rated. '''
for epoch in range(1, nb_epoch + 1):
    train_loss = 0 # to measure the error
    s = 0. # the . to make it type float, counter that will increment after each epoch
    for id_user in range(0, nb_users - batch_size, batch_size): 
        ''' not taking user 1 by 1 but batches of users and since batch size = 100, the first batch is going to contain all users from index 0 to index 99, 2n batch from 100 to 199, etc...
            therefore last batch of users is 943 - 100 (nb_users - batch_size)
            in steps of 100 (batch_size)'''
        vk = training_set[id_user:id_user+batch_size] # users from id_user up to the next 100 users (a batch), this will then go into the Gibbs chain
        v0 = training_set[id_user:id_user+batch_size] # the batch of original ratings untouched, used to measure error
        '''input: ratings of all the movies by the specific user being dealt with in the loop
           target: at the beginning, the same as the input, then the input will go into the Gibbs
           chain and will be updated to get new ratings in each visible node so the input will change
           but the target will keep its same initial value'''
        ph0,_ = rbm.sample_h(v0) # initial probabilities, ,_ to only get the first element of the sample_h function
        for k in range(10):
           '''Gibbs sampling consists of making a Gibbs chain. There are several round trips
           from the visible nodes to the hidden nodes, and then from the hidden nodes to the 
           visible nodes.
           In each roundtrip the visible nodes are dated. Step after step it gets closer to the
           good predicted ratings
           '''
            _,hk = rbm.sample_h(vk) # _, to get the second element returned by the sample_h method. Since this is sampling of the first hidden nodes given values of first visible nodes (original ratings), first input for sample_h function in this first step of Gibbs sampling is visual, but visual (v) is the target (not to change)
            _,vk = rbm.sample_v(hk) # calling sample_v on the 1st sample of the hidden nodes (hk)
            vk[v0<0] = v0[v0<0] # the model shouldn't leatn where there is no rating, so to not include these rows --> freeze visible nodes containing the -1 rating
        phk,_ = rbm.sample_h(vk) 
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1. # the time counter 
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