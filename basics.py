from implementations import *
from helper import *

def train(model, y, x, initial_w = 0, max_iters = 0, gamma = 0, lambda_ = 0, cost = 'mse'):
    if model == 'gradient descent':
        w, loss = mean_squared_error_gd(y,x, initial_w, max_iters, gamma)
        
    elif model == 'stochastic gradient descent':
        w, loss = mean_squared_error_sgd(y, x, initial_w, max_iters, gamma)
    elif model == 'least squares':
        w, loss = least_squares(y, x)
        
    elif model == 'ridge regression':
        w, loss = ridge_regression(y, x, lambda_, cost)
        
    elif model == 'logistic regression':
        w, loss = logistic_regression(y, x, initial_w, max_iters, gamma)
        
    elif model == 'reg logistic regression':
        w, loss = reg_logistic_regression(y, x, lambda_, initial_w, max_iters, gamma)
        
    else:
        print(f'model name -{model}- is incorrect')
        return 0
    return w, loss
        
def estimate(predictions, groundtruth): #to delete
    #if y is -1 or 1. need to change to 1 and -1 if y is 0,1
    #here assuming negative is -1 and positive 1
    difference = groundtruth - predictions
    accuracy =  np.count_nonzero(difference==0)/len(predictions)
    fp = np.count_nonzero(difference == -2)
    fn = np.count_nonzero(difference == 2)
    truepred = np.where(predictions == 1)
    truegt = np.where(groundtruth == 1)
    tp = len(np.intersect(truepred, truegt))
    f1 = tp / (tp + (fp+fn)/2)
    return f1, accuracy