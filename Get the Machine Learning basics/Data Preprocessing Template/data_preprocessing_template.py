# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Matrix of features
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

'''

#Taking care of missing data
from sklearn.preprocessing import Imputer # contains amazing libraries to make Machine Learning models
imputer = Imputer(missing_values = np.nan, strategy='mean', axis=0) # Ctrl+I to see how it works
imputer.fit(X[:, 1:3]) # 1:3 just means columns of index 1 and 2
X[:, 1:3] = imputer.transform(X[:, 1:3]) #method that replaces column ata with mean of column

# Encoding categorical data (e.g. from Spain to 1 or 0)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder() # turns nominal values to numerical
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # all rows, index 0
onehotencoder = OneHotEncoder(categorical_features = [0]) # encode column 0
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


'''

# Splitting the dataset into Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
 sc_X = StandardScaler() # scale the X matrix of features
 X_train = sc_X.fit_transform(X_train) # fit object to training set and then transform it
 X_test = sc_X.transform(X_test) # no need to fit it, because it is already fitted to the training set

 '''
 The feature scaling in x_test is the same as in x_train because the object 
 StandardScalar was fitted to x_train, that's why it's important to fit the 
 object to x_train first so that x_train and x_test are scaled on the same basis 
 
 Do we need to apply feature scaling to y?
 Not in this case, because it is a categorical/classification problem, 
 but for regression problems, when the dependent variable takes a huge
 range of variable, we also apply feature scaling to it.
 
 '''