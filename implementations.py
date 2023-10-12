from helper import *
import numpy as np

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    w = initial_w # initiate w_{t}
    for n_iter in range(max_iters):
        new_w = w - gamma * compute_gradient(y, tx, w) # w_{t+1} = w_{t} - gamma * \/L(w_{t})
        w = new_w # update w_{t} with the value of w_{t+1} for the next iteration
        return w, compute_mse(y, tx, w)

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):    
    w = initial_w # initiate w_{t}
    for n_iter in range(max_iters):
        new_w = w - gamma * compute_stoch_gradient(y, tx, w) # w_{t+1} = w_{t} - gamma * \/L_n(w_{t})
        w = new_w # update w_{t} with the value of w_{t+1} for the next iteration
    return w, compute_mse(y, tx, w)
    
def least_squares(y, tx):
    txT = tx.T #transpose calculation to avoid computing it twice in the arguments of np.linalg.solve
    w = np.linalg.solve(txT.dot(tx),txT.dot(y))
    return w, compute_mse(y, tx, w)

def ridge_regression(y, tx, lambda_):
    #Moreover, 
    # the loss returned by the regularized methods (ridge regression and reg logistic regression) should not include the penalty term.
    
def logistic_regression(y, tx, initial_w, max iters, gamma):
    
def reg_logistic_regression(y, tx, lambda_, initial w, max iters, gamma):