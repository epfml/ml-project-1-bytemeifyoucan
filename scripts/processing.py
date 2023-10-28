import numpy as np
import csv
import os
from visualisation import *
from implementations import *

def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids

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

def cv_loss(model, y, x, k_indices, k, lambda_, initial_w, max_iters, gamma):
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

def run_pca(x, n_components, fig_name = 'PCA_variance_ratios', visualisation = False):
    #use standardized data
    cov_mat = np.cov(x , rowvar = False)
    #Calculating Eigenvalues and Eigenvectors of the covariance matrix
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    #sort the eigenvalues in descending order
    sorted_index = np.argsort(eigen_values)[::-1]
    
    sorted_eigenvalue = eigen_values[sorted_index]
    #similarly sort the eigenvectors 
    sorted_eigenvectors = eigen_vectors[:,sorted_index]

    eigenvector_subset = sorted_eigenvectors[:,0:n_components]
    x_reduced = np.dot(eigenvector_subset.transpose(),x.transpose()).transpose()
    
    plot_pca(n_components, sorted_eigenvalue, visualisation, fig_name)
    
    return x_reduced

def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})



