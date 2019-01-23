from proj1_data_loading import loadData
from construct_dataset import buildMatricies
import numpy as np

from matplotlib import pyplot as plt
from mp1_linear_regression import LinearRegression
# Variables selecting which bonus features to add
include_length = False
include_num_sentences = True

# First load the data using proj1_data_loading
data = loadData()

# Split the data into training, validation and test data according to specs
training_data = data[:10000] #CHANGE THIS TO A SMALL SUBSET FOR CHECKING COVNERGENCE AND STUFF
validation_data = data[10000:11000]
test_data = data[11000:]

# Build the X and Y matricies for the three splits of data
Xtrain, Ytrain = buildMatricies(training_data, include_length, include_num_sentences)
Xval, Yval = buildMatricies(validation_data, include_length, include_num_sentences)
Xtest, Ytest = buildMatricies(test_data, include_length, include_num_sentences)

#print (X_test_data[998])

        

Ytrain=np.asarray([Ytrain]).T

w=np.random.rand(Xtrain.shape[1],Ytrain.shape[1])
myLinReg=LinearRegression(Xtrain, Ytrain)
w=myLinReg.exact_solution()
#print(np.linalg.norm(myLinReg.grad_e(w))) #to validate. should be really close to zero.

#Get accuracy on training data
w=myLinReg.exact_solution() #use exact solution

Ytrain_pred=np.matmul(Xtrain,w)
N=Ytrain.shape[0] #number training example
MAE=np.sum(np.abs(Ytrain_pred-Ytrain))/N
MSE=np.sum(np.square(Ytrain_pred-Ytrain))/N
print('Exact solution evaluated on training data. \n Mean Absolute Error: {}\n Mean Square Error: {} \n'.format(MAE, MSE))

Yval=np.asarray([Yval]).T
Yval_pred=np.matmul(Xval,w)
N=Yval.shape[0] #number validation example
MAE=np.sum(np.abs(Yval_pred-Yval))/N
MSE=np.sum(np.square(Yval_pred-Yval))/N
print('Exact solution evaluated on validation data. \n Mean Absolute Error: {}\n Mean Square Error: {} \n'.format(MAE, MSE))



eta_nod=1e-2
beta=1
step_size=lambda k: eta_nod/(1+beta*k)
max_iter=1000
eps=1e-5

w, grad_norms, conv_flag=myLinReg.gradient_descent(step_size, eps, max_iter)
convString="Yes." if conv_flag==1 else "No."
plt.figure()
plt.plot(grad_norms[20:])
plt.title('Gradient norm vs iteration, tolerance: {0} \n Convergence: {1} Gradient norm at end:{2:9.3f}'.format(eps, convString, grad_norms[-1]))
print(conv_flag)

Ytrain_pred=np.matmul(Xtrain,w)

MAE=np.sum(np.abs(Ytrain_pred-Ytrain))/Ytrain.shape[0]
MSE=np.sum(np.square(Ytrain_pred-Ytrain))/Ytrain.shape[0]
print('Gradient descent solution evaluated on training data. \n Mean Absolute Error: {0:9.3f}\n Mean Square Error: {1:9.3f} \n'.format(MAE, MSE))

Yval=np.asarray([Yval]).T
Yval_pred=np.matmul(Xval,w)
MAE=np.sum(np.abs(Yval_pred-Yval))/Yval.shape[0]
MSE=np.sum(np.square(Yval_pred-Yval))/Yval.shape[0]
print('Gradient descent solution evaluated on validation data. \n Mean Absolute Error: {0:9.3f}\n Mean Square Error: {1:9.3f} \n'.format(MAE, MSE))











