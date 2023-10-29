import numpy as np
import csv
import os
from visualisation import *
from implementations import *


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.

    >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
    (array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]), array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]))
    """
    # set seed
    np.random.seed(seed)
    limit = int(np.floor(ratio*y.size))
    shuffled_indices = np.random.permutation(np.arange(y.size))

    x_shuffled = x[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    
    x_tr = x_shuffled[:limit]
    x_te = x_shuffled[limit:]
    
    y_tr = y_shuffled[:limit]
    y_te = y_shuffled[limit:]
    
    
    return x_tr, x_te, y_tr, y_te

def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cv_loss(model, y, x, k_indices, k, lambda_, max_iters, gamma): 
    """to complete ????

    Args:
        model:      str, ['gradient descent', 'stochastic gradient descent', 'ridge regression', 'logistic regression', 'reg logistic regression]
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        initial_w:  scalar, default to 0, needed for all models but ridge regression
        max_iters:  scalar, default to 0, needed for all models but ridge regression
        gamma:      learning rate, default to 0, needed for all models but ridge regression

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527597114, 0.33555914361295175)
    """
    # get k'th subgroup in test, others in train: 
    test = k_indices[k]
    train = np.delete(k_indices, k, axis=0)
    
    y_te = np.array([y[i] for i in test])
    x_te = np.array([x[i] for i in test])
    
    y_tr = np.array([y[i] for i in train.flatten()])
    x_tr = np.array([x[i] for i in train.flatten()])

    initial_w = np.zeros(len(y_tr))
    if model == 'gradient descent':
        w, _ = mean_squared_error_gd(y_tr,x_tr, initial_w, max_iters, gamma)
        
    elif model == 'stochastic gradient descent':
        w, _ = mean_squared_error_sgd(y_tr, x_tr, initial_w, max_iters, gamma)
        
    elif model == 'ridge regression':
        w, _ = ridge_regression(y_tr, x_tr, lambda_)
        
    elif model == 'logistic regression':
        w, _ = logistic_regression(y_tr, x_tr, initial_w, max_iters, gamma)
        
    elif model == 'reg logistic regression':
        w, _ = reg_logistic_regression(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)
    
    # calculate the loss for train and test data: TODO
    loss_te = np.sqrt(2*compute_mse(y_te, x_te, w))
    loss_tr = np.sqrt(2*compute_mse(y_tr, x_tr, w))
    return loss_tr, loss_te

def run_pca(x, n_components, threshold, visualisation = False):
    #if we want to work with threshold need to put n_components to False
    #use standardized data
    cov_mat = np.cov(x , rowvar = False)
    #Calculating Eigenvalues and Eigenvectors of the covariance matrix
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    #sort the eigenvalues in descending order
    sorted_index = np.argsort(eigen_values)[::-1]
    
    sorted_eigenvalues = eigen_values[sorted_index]
    #similarly sort the eigenvectors 
    sorted_eigenvectors = eigen_vectors[:,sorted_index]

    if n_components == False:
        #find limit tfor trheshold
        eigenvalues_relative = sorted_eigenvalues/np.sum(sorted_eigenvalues) * 100
        cum_eigenvalues = np.cumsum(eigenvalues_relative)
        above_threshold = [(value, i) for i, value in enumerate(cum_eigenvalues) if threshold <= value]
        perfect_components = min(above_threshold)[1]
        print(f'Finished running PCA: keeping { perfect_components +1 } components to explain >{threshold}% variance')
        eigenvector_subset = sorted_eigenvectors[:, 0:perfect_components]
        plot_pca(x.shape[1], sorted_eigenvalues, visualisation, 'PCA_total_variance_decomposition')
    else:
        eigenvector_subset = sorted_eigenvectors[:,0:n_components]
        plot_pca(x.shape[1], sorted_eigenvalues, visualisation, 'PCA_total_variance_decomposition') # to remove?
        
    x_reduced = np.dot(eigenvector_subset.transpose(),x.transpose()).transpose()
    return x_reduced

