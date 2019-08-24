# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np #allows to make some arrays, which are the only allowed input
import matplotlib.pyplot as plt #visualise results on nice charts
import pandas as pd #to be able to import the dataset and manage them easily

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv') #importing the data as a DataFrame
training_set = dataset_train.iloc[:, 1:2].values 
'''
    we need to select the right column that we need, which is the Open google stock price
    and also make it a NumPy array because only NumPy arrays can be the input of NNs in keras
    iloc method to get the right index of the column we want
     - : all the rows 
     - 1:2, ranges in python have their upper bound excluded, therefore the 2 here is excluded,
       only column 1 (Open) and this makes a NumPy array of 1 column
     this last line creates a DataFrame but only with the Open column
     - .values : to make it a numpy array
    
'''

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler 
sc = MinMaxScaler(feature_range = (0, 1)) #create an object sc from the class MinMaxScaler with the default feature range (0,1)
training_set_scaled = sc.fit_transform(training_set)
'''
    recommended/more relevant to use Normalisation than Standardisation for feature scaling in RNNs
    in RNNs, if there is a Sigmoid activation function in the output layer, its better to use normalisation
    feature range (0, 1) so that now all the stock prices will be between 0 and 1 
    apply the fit transform method from the MinMaxScaler class to the sc object
    fit sc object to the training set, which is inputted as argument 
    and transform it a.k.a scale it 
'''

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
'''
    60 timesteps means that at every time T, the RNN is going to look at the 60 stock prices 
    before time T, that is the stock prices between 60 days before time T and time T, and based on 
    the trends, it is capturing during these previous timesteps, it will try to predict the next output
    60 timesteps of the past information from which the RNN is gonna try to learn and understand some correlations
    based on understanding, it will try to predict the next output, the stock price at time T+1
    
    1 timestep is completely stupid because it will lead to overfitting
    20 timesteps not enough to capture trends
    the best one after trial and error is 60
    60 financial days, since there are 20 financial days in a month then 60 timesteps correspond to 3 months
    
    so for every day, the RNN will look at the 3 previous months to predict the next day 
    
    1 observation = 1 financial day
    for each observation, x_train will contain the 60 previews stock prices before that observation
    y_train will contain the stock price the next financial day
    
    x_train and y_train are lists, they have to be numpy arrays again so they can be accepted
    by the RNN
    
'''

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
'''
    adding more dimensionality to the data, adding the dimension "Unit", the number or predictors
    we can use to predict what we want (Google stock price at time T+1)
    before we could only predict T+1 by the last 60 days, but now we can add more indicators 
    that could help predict even better results 
    reshape function allows you to add a new dimension in a NumPy array without changing its data
    - x_train: the numpy array to reshape
    - 2ns argument: the new shape wanted
        - X_train rn is 2D (60 x 1198), we will add another one to make it 3D
        - (no of observations (rows), no of timesteps (columns), no of indicators we have)
          indicators: apple gets a lot of pieces from samsung, therfore apple depends on samsung
          and samsung depends on apply buying their pieces, hence their stock prices will be similar, 
          samsung is 1 indicator of apple stock prices
        - x_train.shape[0] gives the number of rows (observations)
        - x_train.shape[1] gives the number of columns (timesteps)
'''


# Part 2 - Building the RNN
'''
    Building the whole architecture of this NN 
    A stacked LSTM with some dropout regularization to prevent overfitting 
'''
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential() 
'''
    initialising the regressor
    a sequence of layers
    regressor because we are predicting a continuous value
'''

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
'''
    add LSTM layer 
    adding an object of the LSTM class
    - units: memory units to have in this LSTM layer (neurons) we want the model to have high dimensionality
      it is gonna have high dimensionality thanks to the multiple LSTM layers that we're gonna add, but we
      can increase the dimensionality by including a large number of neurons in each layer
    - return_sequences: true when there are more layers after this one, false when it is the last layer
    - input_shape: shape of x_train (3D - observations, timesteps, indicators) but we don't have
      to include the 3 dimensions, only the 2 last ones because the 1st one will be automatically taken
      into account 
'''
regressor.add(Dropout(0.2))
'''
    specifiy dropout rate, the rate of neurons to drop (ignore) to do the regularization
    to prevent overfitting
    0.2 = 20%
'''

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True)) # no need to specify the input shape again
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50)) 
regressor.add(Dropout(0.2))
'''
    after this layer comes the output layer with its output dimension
    since it is the last LSTM layer and we don't return any more sequences, we delete
    the return sequence parameter to make it false
'''

# Adding the output layer
regressor.add(Dense(units = 1))
''' here we don't add an LSTM layer, but a classic fully connected layer, becuase
    the classic layer is fully connected to the previous LSTM layer (4th one)
    to make a fully connection we use the Dense class 
    units: the number of neurons there needs to be in the output layer (can't choose)
    output has only 1 dimension therefore 1 unit 
'''

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
'''
    - .compile is a method of the Sequential class
    - for RNNs, an RMS prop optimizer is recommended, however with this model, after experimentation,
      the Adam optimizer gave better results, it is always a safe choice because it always performs
      relevant updates of the weights
    - mean_squared_error because this is a regression problem 
'''

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
'''
    since the data only has 5 past years, not many epochs are needed, 100 is enough
    batch_size: number of samples per gradient update 
'''


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
