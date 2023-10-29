import numpy as np
from processing import *
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

def cross_validation(model, y, x, k_fold, lambdas, initial_w = 0, max_iters = 0, gamma = 0, seed = 1, visualisation = False): #to delete?
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


def find_error(predictions, groundtruth): #to delete 
    indices = np.where(predictions != groundtruth)[0]
    return indices

def compute_losses_for_hyperparameters(model, y, tx, k_fold, w_initial=0, max_iters=0, lambdas = ['Nan'], gammas = ['Nan'], seed = 1):
    """Process cross-validation with the chosen model 
        Calculate the test and train errors for every hyperparameters 

    Args:
        model (string): name of the regression technique chosen
        y (np.ndarray): shape = (N,) contains the data we want to predict
        tx (np.ndarray): shape = (N,D) contains the features used to predict
        initial_w (np.ndarray): shape = (D,) the initial weight pair that will get updated with gradient
        max_iters (int): maximum number of steps
        lambdas (np.ndarray): hyperparameter for the penalized loss for regularized regression
        gammas (np.ndarray): hyperparameter for GD and SGD implementation
        k_fold (int): K in K-fold, i.e. the fold num
        seed (int):  the random seed

    Returns:
        np.ndarray : shape(N,5) train and test errors for each hyperparameters of the chosen model
    """

    results = np.array(["model", "lambda", "gamma", "train error", "test error"])
    
    k_indices = build_k_indices(y, k_fold, seed)

    for lambda_ in lambdas:
        for gamma in gammas:
            losses_tr = []
            losses_te = []
            for k in range(k_fold): 
                loss_tr, loss_te = cv_loss(model, y, tx, k_indices, k, lambda_, w_initial, max_iters, gamma)
                losses_tr.append(loss_tr)
                losses_te.append(loss_te)
            loss_tr = np.mean(losses_tr)
            loss_te = np.mean(losses_te)
            results.append([model, lambda_, gamma, loss_tr, loss_te])

    return results

def find_best_hyperparameters(hyperparameter_losses):
    """Calculate the best hyperparameters based on the test errors computed previously

    Args:
        np.ndarray : shape(N,5) train and test errors for each hyperparameters of the chosen model

    Returns:
        np.ndarray : shape (1,5) train and test errors for the best hyperparameters of the chosen model
    """

    min_index = np.argmin(hyperparameter_losses[-1], axis=1)
    return hyperparameter_losses[min_index]