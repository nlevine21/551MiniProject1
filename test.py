from proj1_data_loading import loadData
from construct_dataset import buildMatricies
import numpy as np
import time

from matplotlib import pyplot as plt
from mp1_linear_regression import LinearRegression

# Variables selecting which bonus features to add
include_length = False #orinially false
include_num_sentences = True #originally true
number_of_frequent_words = 160
include_hyperlink = True
include_wordsentence_ratio = True
include_swear_ratio = True


# First load the data using proj1_data_loading
data = loadData()
# Split the data into training, validation and test data according to specs
training_data = data[:10000]  # CHANGE THIS TO A SMALL SUBSET FOR CHECKING COVNERGENCE AND STUFF
validation_data = data[10000:11000]
test_data = data[11000:]


#Build the X and Y matricies for the three splits of data
Xtrain, Ytrain = buildMatricies(training_data, include_length, include_num_sentences)
Xval, Yval = buildMatricies(validation_data, include_length, include_num_sentences)
Xtest, Ytest = buildMatricies(test_data, include_length, include_num_sentences)
Ytrain = np.asarray([Ytrain]).T
Yval = np.asarray([Yval]).T


#MATRICES ARE CALCULATED ONCE THEN SAVED TO MATRIX.NPY
# np.save('matrix.npy', np.array((Xtrain,Ytrain,Xval,Yval,Xtest,Ytest)))
# Xtrain , Ytrain, Xval, Yval, Xtest , Ytest = np.load('matrix.npy')



# calculating closed form solution
myLinReg = LinearRegression(Xtrain, Ytrain)
w = myLinReg.exact_solution()

# evaluate closed form on training set using mean square difference, and absolute difference
Ytrain_pred = np.matmul(Xtrain, w)
N = Ytrain.shape[0]  # number training example
MAE = np.sum(np.abs(Ytrain_pred - Ytrain)) / N
MSE = np.sum(np.square(Ytrain_pred - Ytrain)) / N
print(
    'Exact solution evaluated on training data. \n Mean Absolute Error: {}\n Mean Square Error: {} \n'.format(MAE, MSE))

# evaluate closed form on validation set
Yval_pred = np.matmul(Xval, w)
N = Yval.shape[0]  # number validation example
MAE = np.sum(np.abs(Yval_pred - Yval)) / N
MSE = np.sum(np.square(Yval_pred - Yval)) / N
print('Exact solution evaluated on validation data. \n Mean Absolute Error: {}\n Mean Square Error: {} \n'.format(MAE,
                                                                                                                  MSE))

print("")