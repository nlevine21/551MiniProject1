#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 12:48:33 2019

@author: vassili
"""
import numpy as np

class LinearRegression:
    
    """LinearRegression object. 
    Attributes:
        X: Input data. M by N numpy array. M is number of data points, N number features
        Y: Output data. M by P numpy array. M number data points, P number of variables to predict. 
        XtX: X transpose times X. Expensive to compute. Computed once on initialization, then reused.  
    Notes: Y has to have two dimensions for the matrix multiplications to work. 
                So, e.g. not (10,) but rather need (10,1)
    """
    def __init__(self, X, Y):
        self.XtX=np.matmul(np.transpose(X),X)
        self.X=X;
        self.Y=Y;
    def grad_e(self, w):
        """Gradient of weights with respect to error. Returns N by 1 column matrix."""
        return 2*(np.matmul(self.XtX, w)-np.matmul(np.transpose(self.X),self.Y))
    def exact_solution(self):
        """Computes the optimal weights for linear regression.
            Returns weights vector w, N by P, where N number features, P number variables to predict
            X: M by N matrix where M is number data points
            Y: M by P matrix"""
        w=np.matmul(np.linalg.inv(self.XtX),np.transpose(self.X))
        w=np.matmul(w, self.Y)
        return w
    def gradient_descent(self, step_size, tol, max_iter):
        """ Gradient descent, takes step_size (a function! that returns step size vs iteration)
        and the maximum number of iterations."""
        conv_flag=0
        w=np.random.rand(self.X.shape[1],self.Y.shape[1])
        grad_norms=-1*np.ones([max_iter+1,1])
        for iter in range(max_iter):
            temp=w;
            w=w-step_size(iter+1)*self.grad_e(w)
            grad_norms[iter+1]=np.linalg.norm(self.grad_e(w))
            if np.linalg.norm(w-temp)<tol:
                conv_flag=1
                grad_norms=grad_norms[grad_norms!=-1.]
                break
            
        return w, grad_norms, conv_flag
    