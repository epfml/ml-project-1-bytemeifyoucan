import numpy as np
from processing import *
from visualisation import *

def compute_mse(e): 
    """Calculate the loss using MAE.

    Args:
        e: numpy array of shape=(N, )
        
    Returns:
        float: MAE across e
    """
    return 1/2*np.mean(e**2)


def compute_mae(e):
    """Calculate the loss using MAE.

    Args:
        e: numpy array of shape=(N, )
        
    Returns:
        float: MAE across e
    """
    return np.mean(np.abs(e))

def compute_gradient(y, tx, w):
    return - 1/len(y) * tx.T.dot(y - tx.dot(w))

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def compute_stoch_gradient(y, tx, w):
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
        return compute_gradient(minibatch_y, minibatch_tx, w)


#======
def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
            
    return 1.0 / (1.0 + np.exp(-t))

def cross_validation(model, y, x, k_fold, lambdas, initial_w = 0, max_iters = 0, gamma = 0, seed = 1, visualisation = False):
    """WORK IN PROGRESS?????

    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    for lambda_ in lambdas:
        avg_tr = []
        avg_te = []
        for k in range(k_fold):
            tr_err, te_err = cv_loss(model, y, x, k_indices, k, lambda_, initial_w , max_iters, gamma)
            avg_tr.append(tr_err)
            avg_te.append(te_err)
        rmse_tr.append(np.mean(avg_tr))
        rmse_te.append(np.mean(avg_te))
    
    best_rmse = min(rmse_te)
    best_lambda = lambdas[rmse_te.index(best_rmse)]
        
    cross_validation_visualization(lambdas, rmse_tr, rmse_te, visualisation)
    return best_lambda, best_rmse

def get_confusionm(predictions, groundtruth): 
    difference = groundtruth - predictions
    fp = np.count_nonzero(difference == -2)
    fn = np.count_nonzero(difference == 2)
    truepred = np.where(predictions == 1)
    truegt = np.where(groundtruth == 1)
    tp = len(np.intersect(truepred, truegt))
    tn = np.count_nonzero(difference==0) - tp
    return tp, fp, tn, fn

def find_error(predictions, groundtruth):
    indices = np.where(predictions != groundtruth)[0]
    return indices