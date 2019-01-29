from proj1_data_loading import loadData
from construct_dataset import buildMatricies
import numpy as np
import time

from matplotlib import pyplot as plt
from mp1_linear_regression import LinearRegression

# Variables selecting which bonus features to add
include_length = False
include_num_sentences = False
number_of_top_words = 160
include_hyperlink = False
include_wordsentence_ratio = False
include_swear_ratio = False
include_puncs = False
include_children_squared = False

additional_feature_dict = {
    "0": "length",
    "1": "number of sentences",
    "2": "contains hyperlink",
    "3": "word sentence ratio",
    "4": "swear word ratio",
    "5": "number of question marks",
    "6": "children^2"
}


# First load the data using proj1_data_loading
data = loadData()
# Split the data into training, validation and test data according to specs
training_data = data[:10000]  
validation_data = data[10000:11000]
test_data = data[11000:]

def matrix_builder(boolean_tuple,words):
    include_length, include_num_sentences, include_hyperlink, include_wordsentence_ratio,include_swear_ratio,include_puncs, include_children_squared = boolean_tuple
    number_of_top_words = words

    # Build the X and Y matricies for the three splits of data
    Xtrain, Ytrain = buildMatricies(training_data, include_length, include_num_sentences, number_of_top_words,
                                    include_hyperlink,
                                    include_wordsentence_ratio, include_swear_ratio,include_puncs, include_children_squared)
    Xval, Yval = buildMatricies(validation_data, include_length, include_num_sentences, number_of_top_words,
                                include_hyperlink,
                                include_wordsentence_ratio, include_swear_ratio,include_puncs, include_children_squared)
    Xtest, Ytest = buildMatricies(test_data, include_length, include_num_sentences, number_of_top_words,
                                  include_hyperlink,
                                  include_wordsentence_ratio, include_swear_ratio,include_puncs, include_children_squared)
    Ytrain = np.asarray([Ytrain]).T
    Yval = np.asarray([Yval]).T
    Ytest = np.asarray([Ytest]).T

    return (Xtrain, Ytrain, Xval, Yval, Xtest, Ytest)


# train and test closed form solution. Prints results.
def test_exact(matrices):
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = matrices

    # calculating closed form solution
    myLinReg = LinearRegression(Xtrain, Ytrain)
    w = myLinReg.exact_solution()
    # evaluate closed form on training set using mean square difference, and absolute difference
    Ytrain_pred = np.matmul(Xtrain, w)
    N = Ytrain.shape[0]  # number training example
    MAE = np.sum(np.abs(Ytrain_pred - Ytrain)) / N
    MSE = np.sum(np.square(Ytrain_pred - Ytrain)) / N
    print(
        'Exact solution evaluated on training data. \n Mean Absolute Error: {}\n Mean Square Error: {} \n'.format(MAE,
                                                                                                                  MSE))

    # evaluate closed form on validation set
    Yval_pred = np.matmul(Xval, w)
    N = Yval.shape[0]  # number validation example
    MAE = np.sum(np.abs(Yval_pred - Yval)) / N
    MSE = np.sum(np.square(Yval_pred - Yval)) / N
    print(
        'Exact solution evaluated on validation data. \n Mean Absolute Error: {}\n Mean Square Error: {} \n'.format(MAE,
                                                                                                                    MSE))

# train and test using gradient descent
def test_grad(matrices,eta,beta,iter):
    # Build the X and Y matricies for the three splits of data
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = matrices

    myLinReg = LinearRegression(Xtrain, Ytrain)

    # step size function and other parameters for gradient descent
    eta = 0.001
    beta = 0.01
    step_size = lambda k: eta / (1 + beta*k)
    max_iter = iter
    eps = 1e-5

    # calc solution using gradient descent
    w, grad_norms, conv_flag = myLinReg.gradient_descent(step_size, eps, max_iter)

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


# Q1
print("Run Experiment 1\n")
# CALCULATE AVERAGE RUNTIME OF ALGORITHMS (only using the three non text features)

# calculate average runtime of closed form solution over 100 trials
avg_runtime = 0
matrices = matrix_builder((include_length,include_num_sentences,include_hyperlink,include_wordsentence_ratio,include_swear_ratio,include_puncs, include_children_squared),0)
Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = matrices
myLinReg = LinearRegression(Xtrain, Ytrain)
for i in range(100):
    start = time.time()
    w = myLinReg.exact_solution()
    end = time.time()
    avg_runtime += (end-start)
print("average time for exact solution is: ",avg_runtime/100)

# calculate average run time of gradient descent over 100 trials

# step size function and other parameters for gradient descent
eta = 0.03
beta = 0.001
step_size = lambda k: eta / (1 + beta * k)
max_iter = 1000
eps = 1e-5

avg_runtime=0
for i in range(100):
    start = time.time()
    # calc solution using gradient descent
    w, grad_norms, conv_flag = myLinReg.gradient_descent(step_size, eps, max_iter)
    end = time.time()
    avg_runtime += (end-start)
print("average time for gradient descent is: ",avg_runtime/100,"\n")

# PERFORMANCE (only using the three non text features)
test_exact(matrices)
test_grad(matrices,eta,beta,max_iter)

# Q2
print("Run Experiment 2 \n")

# three models tested on the training and validation data
print("testing model with only 3 non text features: \n")
test_exact(matrices) # no text features

print("testing model with 3 non text features and 60 top words: \n")
matrices_60 = matrix_builder((include_length,include_num_sentences,include_hyperlink,include_wordsentence_ratio,include_swear_ratio,include_puncs, include_children_squared),60)
test_exact(matrices_60) # includes 60 text features

print("testing model with 3 non text features and all 160 top words: \n")
matrices_160 = matrix_builder((include_length,include_num_sentences,include_hyperlink,include_wordsentence_ratio,include_swear_ratio,include_puncs, include_children_squared),160)
test_exact(matrices_160) # includes 160 word features



# Q3
print("Run Experiment 3")
print ("Testing each additional feature to see whehter it improves the model")
boolean_array=[] # array of all possible 5 boolean tuples
for i in range(8):
    arr = []
    for j in range(7):
        arr.append(i==j)
    boolean_array.append(tuple(arr))

MSE_validation_list = []

for i in boolean_array:
    #Build the X and Y matricies for the three splits of data
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = matrix_builder(i,60)

    # calculating closed form solution
    myLinReg = LinearRegression(Xtrain, Ytrain)
    
    w = myLinReg.exact_solution()

    # evaluate closed form on training set using mean square difference, and absolute difference
    Ytrain_pred = np.matmul(Xtrain, w)
    Nx = Ytrain.shape[0]  # number training example
    MAEx= np.sum(np.abs(Ytrain_pred - Ytrain)) / Nx
    MSEx = np.sum(np.square(Ytrain_pred - Ytrain)) / Nx


    # evaluate closed form on validation set
    Yval_pred = np.matmul(Xval, w)
    N = Yval.shape[0]  # number validation example
    MAE = np.sum(np.abs(Yval_pred - Yval)) / N
    MSE = np.sum(np.square(Yval_pred - Yval)) / N

    MSE_validation_list.append(MSE)

    print(i)
    print(
        'Exact solution evaluated on training data. \n Mean Absolute Error: {}\n Mean Square Error: {} \n'.format(
            MAEx, MSEx))
    print(
        'Exact solution evaluated on validation data. \n Mean Absolute Error: {}\n Mean Square Error: {} \n'.format(
            MAE,MSE))

#find which features improve model and build best model
boolean_arr = [False,False,False,False,False,False,False]
for i in range(7):
    if MSE_validation_list[i]<MSE_validation_list[7]:
        print ('Additional Feature {} improved the model'.format(
            additional_feature_dict[str(i)]))
        boolean_arr[i] = True

#Build the X and Y matricies for the best model
Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = matrix_builder(boolean_arr,60)

# calculating closed form solution
myLinReg = LinearRegression(Xtrain, Ytrain)

w = myLinReg.exact_solution()

# evaluate closed form on training set using mean square difference, and absolute difference
Ytrain_pred = np.matmul(Xtrain, w)
Nx = Ytrain.shape[0]  # number training example
MAEx= np.sum(np.abs(Ytrain_pred - Ytrain)) / Nx
MSEx = np.sum(np.square(Ytrain_pred - Ytrain)) / Nx


# evaluate closed form on validation set
Yval_pred = np.matmul(Xval, w)
N = Yval.shape[0]  # number validation example
MAE = np.sum(np.abs(Yval_pred - Yval)) / N
MSE = np.sum(np.square(Yval_pred - Yval)) / N

# evaluate closed form on test set
Ytest_pred = np.matmul(Xtest, w)
Nz = Ytest.shape[0]
MAEz = np.sum(np.abs(Ytest_pred - Ytest)) / N
MSEz = np.sum(np.square(Ytest_pred - Ytest)) / N

print(
    'Best exact solution evaluated on training data. \n Mean Absolute Error: {}\n Mean Square Error: {} \n'.format(
        MAEx, MSEx))
print(
    'Best exact solution evaluated on validation data. \n Mean Absolute Error: {}\n Mean Square Error: {} \n'.format(
        MAE,MSE))

print(
    'Best exact solution evaluated on validation data. \n Mean Absolute Error: {}\n Mean Square Error: {} \n'.format(
        MAEz,MSEz))


def generate_graph_images():
    print("\nnow calculating some interesting graphs, may take a few minutes...\n")
    include_length, include_num_sentences, include_hyperlink, include_wordsentence_ratio, include_swear_ratio, include_puncs, include_children_squared = (False, False, False, False, False, False, False)

    # GRAPHS FOR VASSILI
    # first is a graph of comparing the MSEs and MAEs of models using 0-160 top-word features
    MSE_training = []
    MAE_training = []
    MSE_validation = []
    MAE_validation = []
    for i in range(161):
        if i%20==0:
            print(i)
        matrices = matrix_builder(
            (include_length, include_num_sentences, include_hyperlink, include_wordsentence_ratio, include_swear_ratio,include_puncs, include_children_squared), i)
        Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = matrices
        myLinReg = LinearRegression(Xtrain, Ytrain)
        w = myLinReg.exact_solution()
        Ytrain_pred = np.matmul(Xtrain, w)
        N = Ytrain.shape[0]  # number training example
        MSE = np.sum(np.square(Ytrain_pred - Ytrain)) / N
        MAE = np.sum(np.abs(Ytrain_pred - Ytrain)) / N
        MSE_training.append(MSE)
        MAE_training.append(MAE)

        Yval_pred = np.matmul(Xval, w)
        N = Yval.shape[0]  # number validation example
        MAE = np.sum(np.abs(Yval_pred - Yval)) / N
        MSE = np.sum(np.square(Yval_pred - Yval)) / N
        MSE_validation.append(MSE)
        MAE_validation.append(MAE)


    plt.figure()
    plt.plot(MSE_training[:])
    #plt.plot(MAE_training[:])
    plt.title('MSE of model vs # of top words used in model. Using training data')
    #plt.legend(['MSE', 'MAE',], loc='upper left')
    plt.show()
    savefig('foo.png')

    plt.figure()
    plt.plot(MSE_validation[:])
    #plt.plot(MAE_validation[:])
    plt.title('MSE of model vs # of top words used in model. Using validation data')
    #plt.legend(['MSE', 'MAE',], loc='upper left')
    plt.show()
    savefig('foo2.png')

    #graph showing runtime vs amount of data
    runtimes = []
    x_axis = []
    i=170
    for j in range(14):
        x_axis.append(i)
        training_data = data[:i]
        matrices = matrix_builder((False,False,False,False,False,False,False),160)
        start = time.time()
        test_exact(matrices)
        runtimes.append(time.time()-start)
        i *= 1.3

    plt.figure()
    plt.plot(x_axis[:],runtimes[:])
    plt.title('runtime of closed form algorithm vs # amount of data')
    plt.show()
