from helper import *
import numpy as np

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Calculates the gradient of the unregularized loss and uses it in gradient descent to approximate optimal weights
    y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Args:
        y (np.ndarray): shape = (N,) contains the data we want to predict
        tx (np.ndarray): shape = (N,2) contains the features used to predict
        initial_w (np.ndarray): shape = (2,) the initial weight pair that will get updated with gradient
        max_iters (int): maximum number of steps
        gamma (float): learning rate

    Returns:
        np.ndarray : shape = (2,) optimal weights
        float : mean squared erros
    """
    w = initial_w # initiate w_{t}
    for n_iter in range(max_iters):
        e = y - np.dot(tx,w)
        gradient = -tx.T.dot(e) / len(e)
        new_w = w - gamma * gradient # w_{t+1} = w_{t} - gamma * \/L(w_{t})
        w = new_w # update w_{t} with the value of w_{t+1} for the next iteration
    e = y - np.dot(tx,w)
    return w, compute_mse(e)

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):  
    """Calculates the gradient of the unregularized loss and uses it in stochastic gradient descent to approximate optimal weights

    Args:
        y (np.ndarray): shape = (N,) contains the data we want to predict
        tx (np.ndarray): shape = (N,2) contains the features used to predict
        initial_w (np.ndarray): shape = (2,) the initial weight pair that will get updated with gradient
        max_iters (int): maximum number of steps
        gamma (float): learning rate

    Returns:
        np.ndarray : shape = (2,) optimal weights
        float : mean squared erros
    """
    w = initial_w # initiate w_{t}
    for n_iter in range(max_iters):
        new_w = w - gamma * compute_stoch_gradient(y, tx, w) # w_{t+1} = w_{t} - gamma * \/L_n(w_{t})
        w = new_w # update w_{t} with the value of w_{t+1} for the next iteration
    e = y - tx.dot(w)
    return w, compute_mse(e)
    

    
def least_squares(y, tx):
    """Computes optimal weights by solving the normal equation

    Args:
        y (np.ndarray): shape = (N,) contains the data we want to predict
        tx (np.ndarray): shape = (N,2) contains the features used to predict

    Returns:
        np.ndarray : shape = (2,) optimal weights
        float : mean squared error
    """
    txT = tx.T #transpose calculation to avoid computing it twice in the arguments of np.linalg.solve
    w = np.linalg.solve(txT.dot(tx),txT.dot(y))
    #return w, compute_mse(y, tx, w)
    e = y - tx.dot(w)
    return w, compute_mse(e)


def ridge_regression(y, tx, lambda_, cost = 'mse'):
    """Implements ridge regression with L2 regularisation to optimize weights by solving normal equation. 

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: penalization coefficient.
        cost: str {'mse', 'mae'}, default = 'mse', can also be 'mae

    Returns:
        np.ndarray : shape = (D, ) optimal weights, D is the number of features.
        float: mse or mae loss

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    N = y.size
    D = tx.shape[1]
    txT = tx.T
    w = np.linalg.solve(np.dot(txT, tx) + 2*N*lambda_ * np.identity(D) , np.dot(txT, y))
    e = y - tx.dot(w)
    if cost == 'mse':
        loss = compute_mse(e)
    else:
        loss = compute_mae(e)
    return w, loss
    

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Will perform logistic regression on tx and y, bu computing gradient and updating w max_iter times.

    Args:
        y (np.ndarray): shape = (N,) contains the data we want to predict
        tx (np.ndarray): shape = (N,2) contains the features used to predict
        initial_w (np.ndarray): shape = (2,) the initial weight pair that will get updated with gradient
        max_iters (int): maximum number of steps
        gamma (float): learning rate
        labels: True (default) if we use {0,1} labels, False if we use {-1,1} labels

     Returns:
        np.ndarray : shape = (2,) optimal weights
        float : mean squared error
    """
    w = initial_w

    for n_iter in range(max_iters):
        s = sigmoid(tx.dot(w))
        gradient = - (tx.T).dot(y - s) / len(y)
        new_w = w - gamma * gradient # w_{t+1} = w_{t} - gamma * \/L_n(w_{t})
        w = new_w # update w_{t} with the value of w_{t+1} for the next iteration

    s = sigmoid(tx.dot(w))
    loss = - np.mean(y * np.log(s) + (1 - y) * np.log(1 - s))

    return w, loss

    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """_summary_

    Args:
        y (np.ndarray): shape = (N,) contains the data we want to predict
        tx (np.ndarray): shape = (N,2) contains the features used to predict
        lambda_ (float): penalization coefficient
        initial_w (np.ndarray): shape = (2,) the initial weight pair that will get updated with gradient
        max_iters (int): maximum number of steps
        gamma (float): learning rate
        labels: True (default) if we use {0,1} labels, False if we use {-1,1} labels

    Returns:
        np.ndarray : shape = (2,) optimal weights
        float : mean squared error
    """

    w = initial_w

    for n_iter in range(max_iters):
        s = sigmoid(tx.dot(w))
        gradient = - (tx.T).dot(y - s) / len(y) +  2 * lambda_ * np.abs(w)
        new_w = w - gamma * gradient # w_{t+1} = w_{t} - gamma * \/L_n(w_{t})
        w = new_w # update w_{t} with the value of w_{t+1} for the next iteration

    s = sigmoid(tx.dot(w))
    loss = - np.mean(y * np.log(s) + (1 - y) * np.log(1 - s))

    return w, loss