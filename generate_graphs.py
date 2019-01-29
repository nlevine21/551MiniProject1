from main import matrix_builder


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