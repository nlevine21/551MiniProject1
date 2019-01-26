from proj1_data_loading import loadData
from construct_dataset import buildMatricies
import numpy as np
import time

from matplotlib import pyplot as plt
from mp1_linear_regression import LinearRegression

# Variables selecting which bonus features to add
include_length = False #orinially false
include_num_sentences = True #originally true
num_of_frequent_words = 160


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

    # print("a",len(Xtrain),"b",w,"c",Ytrain.shape,"d",Xtrain.shape)
w1 = w.copy() #josh
    # plt.figure()
    # plt.plot([Xtest[i][0] for i in range(1000)],[Ytest[i] for i in range(1000)], 'ro')
    # plt.show()

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

# step size function and other parameters for gradient descent
eta_nod = 0.001
beta = 0.01
step_size = lambda k: eta_nod / (1+beta*np.log(k))
max_iter = 600
eps = 1e-6

# calc solution using gradient descent
w, grad_norms, conv_flag = myLinReg.gradient_descent(step_size, eps, max_iter)

# analyzing how successful grad descent was and plotting its gradient
convString = "Yes." if conv_flag == 1 else "No."
plt.figure()
plt.plot(grad_norms[500:])
plt.title('Gradient norm vs iteration, tolerance: {0} \n Convergence: {1} Gradient norm at end:{2:9.3f}'.format(eps,
                                                                                                                convString,
                                                                                                                grad_norms[
                                                                                                                    -1]))
plt.show()

# calculating how close the grad descent solution is to the closed form solution
diff = np.linalg.norm(w1-w)
print(w.size)
avgdiff = diff/w.size
print("norm difference of vectors:{0:9.5f}: \n avg difference:{1:9.5f}\n".format(diff,avgdiff) )
print((np.sum(np.abs(w1)))/w1.size)

# print if grad decent converged close
print(conv_flag)

# evaluating grad descent solution on training set
Ytrain_pred = np.matmul(Xtrain, w)
MAE = np.sum(np.abs(Ytrain_pred - Ytrain)) / Ytrain.shape[0]
MSE = np.sum(np.square(Ytrain_pred - Ytrain)) / Ytrain.shape[0]
print(
    'Gradient descent solution evaluated on training data. \n Mean Absolute Error: {0:9.3f}\n Mean Square Error: {1:9.3f} \n'.format(
        MAE, MSE))

# evaluating grad descent solution on validation set
Yval_pred = np.matmul(Xval, w)
MAE = np.sum(np.abs(Yval_pred - Yval)) / Yval.shape[0]
MSE = np.sum(np.square(Yval_pred - Yval)) / Yval.shape[0]
print(
    'Gradient descent solution evaluated on validation data. \n Mean Absolute Error: {0:9.5f}\n Mean Square Error: {1:9.5f} \n'.format(
        MAE, MSE))


