import numpy as np
# ========================

def compute_mse(e): 
    """Calculate the loss using MAE.

    Args:
        e: numpy array of shape=(N, )
        
    Returns:
        float: MAE across e
    """
    return 1/2*np.mean(e**2)
# ========================

def compute_mae(e):
    """Calculate the loss using MAE.

    Args:
        e: numpy array of shape=(N, )
        
    Returns:
        float: MAE across e
    """
    return np.mean(np.abs(e))

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
        e = minibatch_y - np.dot(minibatch_tx,w)
        return -tx.T.dot(e) / len(e)

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
            
    return 1.0 / (1.0 + np.exp(-t))
