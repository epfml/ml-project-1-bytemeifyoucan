#load
#clean by removing nan/adding values
#split
#remove const
#remove directly correlated
#standardize
#normalize
#split data for folds (CV)
#pca decomposition to get principal ocmponents
import numpy as np
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

#======= DATA CLEANING ==========
#get dictionnary with names DONE
#remove the useless defined by yann OK
#remove nan columns OK
#remove constant columns DONE
#remove correlated columns OK
#for categorical values do one hot encoding -----CHECK YANN
#for continuous values add mean DONE
#standardize OK
#clean function DONE

def get_type_features(data, type_):
    """
    This function returns the data with the selected features type 

    Args:
        data (np.ndarray): data
        type_ (string): type of the data, either 'continuous' or 'categorical'

    Returns:
        continuous_data or categorical_data (np.array): new data containing only the old data with the correct type
    """

    if(type_ != 'continuous' or 'categorical'):
        raise TypeError("Type of data must be either categorical or continuous")
    
    indices = [data_mapping[key][1] for key in data_mapping if data_mapping[key][0] == type_]
    type_data = data[:, indices]
    return type_data
    
def remove_useless_features(data):
    """
    This function removes the useless features of the dataset (e.g. a phone number, hidden data)
    These features are labelled as 'to_delete' in the data_mapping dictionnary

    Args:
        data (np.ndarray): data

    Returns:
        clean_data (np.ndarray): clean data
    """

    columns_to_remove = [data_mapping[key][1] for key in data_mapping if data_mapping[key][0] == 'delete']
    clean_data = np.delete(data, columns_to_remove, axis = 1)
    return clean_data
    
def remove_nan_columns(data,threshold=0.8):
    """
    This function removes the features containing more Nan values than the limit imposed by the treshold
    This function removes from the dictionary data_mapping the removed feature
    Then it updates their index

    Args:
        data (np.ndarray): data
        threshold (float): threshold of maximum Nan values percentage by column, default = 0.8

    Returns:
        new_data (np.ndarray): data without the features containing too much Nan values to be releavant 
    """

    nan_ratio = np.sum(np.isnan(data), axis = 0)/ data.shape[0]
    columns_to_remove = np.where(nan_ratio > threshold)[0]
    without_nan = np.delete(data, columns_to_remove, axis = 1)

    # Step 1: Identify keys to delete
    keys_to_delete = [key for key, value in data_mapping.items() if np.isin(value[1], columns_to_remove)]

    # Step 2: Remove keys with 'delete' type
    for key in keys_to_delete:
        del data_mapping[key]

    # Step 3: Adjust indexes for remaining features
    remaining_keys = list(data_mapping.keys())
    for i, key in enumerate(remaining_keys):
        data_mapping[key][1] = i

    return without_nan

def remove_constant_continuous(data_array, threshold_ratio=0.001):
    """
    Remove constant features from the data array based on a threshold ratio.

    Parameters:
    - data_array: NumPy array containing the data.
    - threshold_ratio: Threshold ratio for standard deviation. Features with
      standard deviation less than or equal to threshold_ratio * max_std will be removed.

    Returns:
    - array_without_constants: NumPy array with constant features removed.
    """
    std_values = np.std(data_array, axis=0)
    max_std = np.max(std_values)
    
    # Identify constant features based on the threshold ratio
    constant_features = np.where(std_values <= threshold_ratio * max_std)[0]

    # Remove constant features
    array_without_constants = np.delete(data_array, constant_features, axis=1)

    return array_without_constants

def remove_constant_categorical(data, threshold=0.001):
    constant_cols = []
    for i in range(data.shape[1]):

        unique_vals, counts = np.unique(~np.isnan(data[:, i]), return_counts=True)
        freq = counts.max() / len(data)
        if freq > 1 - threshold:
            constant_cols.append(i)
    
    data_filtered = np.delete(data, constant_cols, axis=1)

    return data_filtered
    
def delete_correlated_features(data):
    key_to_delete = []

    for key in correlated_with:
        is_correlated = []
        first_feature = data[:data_mapping[key][1]]
        for _key in correlated_with[key]:
            second_feature = data[:data_mapping[_key][1]]
            correlation = np.corrcoef(first_feature, second_feature)
            if correlation > 0.3 :
                is_correlated.append(True)
            else:
                is_correlated.append(False)
        if np.all(is_correlated):
            key_to_delete = key  

    return key_to_delete

def complete(data):
    """
    This function complete continuous features containing Nan values 

    Args:
        data (np.ndarray): data

    Returns:
        completed_data (np.ndarray): completed data
    """
    #avg add
    completed_data = data.copy()
    column_means = np.nanmean(completed_data, axis=0)

    # Find NaN values in the array
    nan_mask = np.isnan(completed_data)

    # Replace NaN values with the mean of their respective columns
    completed_data[nan_mask] = np.take(column_means, np.where(nan_mask)[1])
    return completed_data
    
def standardize(data):
    """
    This function standardizes the continuous features that had been previously completed to get rid of Nan values
    Standardize the data by subtracting the mean and dividing by the standard deviation.
    
    Args:
        data (np.ndarray): data

    Returns:
        standardized_data (np.ndarray): standardized data
    """

    means = np.mean(data, axis = 0)
    stds = np.std(data, axis = 0)
    
    standardized_data = (data - means)/ stds
    
    return standardized_data

def clean_data_mapping():
    """
    This function removes from the dictionary data_mapping the feature labelled as 'to_delete'
    Then it updates their index
    """
    # Step 1: Identify keys to delete
    keys_to_delete = [key for key, value in data_mapping.items() if value[0] == 'delete']

    # Step 2: Remove keys with 'delete' type
    for key in keys_to_delete:
        del data_mapping[key]

    # Step 3: Adjust indexes for remaining features
    remaining_keys = list(data_mapping.keys())
    for i, key in enumerate(remaining_keys):
        data_mapping[key][1] = i

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
    
def clean_VIVA(data, train = True):
    
    clean_data = data.copy()

    if train:
       split_ratio = 0.8
       x_tr, x_te, y_tr, y_te = split_data(x, y, split_ratio) # issue
       clean_data = x_tr

    clean_data = remove_useless_features(clean_data)
    clean_data_mapping()

    clean_data = remove_nan_columns(clean_data)

    for key, value in data_mapping.items():
        if value[0] == 'continuous':
            col_to_change = clean_data[:,value[1]]
            update_col = complete(col_to_change)
            update_col = standardize(update_col)
            clean_data[:,value[1]] = update_col

    keys_to_delete = delete_correlated_features(data)
    # Step 2: Remove keys with 'delete' type
    for key in keys_to_delete:
        del data_mapping[key]
    # Step 3: Adjust indexes for remaining features
    remaining_keys = list(data_mapping.keys())
    for i, key in enumerate(remaining_keys):
        data_mapping[key][1] = i

    categorical_data_encoded = []
    for key, value in data_mapping.items():
        if value[0] == 'categorical':
            col = clean_data[:,value[1]]
            new_cols = perform_one_hot_encoding(col)
            categorical_data_encoded = np.append(categorical_data_encoded, new_cols, axis=1)

    ### VARIANCE : CONST FEATURE ???

    continuous_data = get_type_features(clean_data, 'continuous')
    clean_data = np.concatenate(continuous_data, categorical_data_encoded, axis=1)

    return clean_data

def clean_YANN(data, train = True):
    
    clean_data = data.copy()

    if train:
       split_ratio = 0.8
       x_tr, x_te, y_tr, y_te = split_data(x, y, split_ratio) # issue
       clean_data = x_tr

    clean_data = remove_useless_features(clean_data)
    clean_data_mapping()

    clean_data = remove_nan_columns(clean_data)

    for key, value in data_mapping.items():
        if value[0] == 'continuous':
            col_to_change = clean_data[:,value[1]]
            update_col = complete(col_to_change)
            update_col = standardize(update_col)
            clean_data[:,value[1]] = update_col

    keys_to_delete = delete_correlated_features(data)
    # Step 2: Remove keys with 'delete' type
    for key in keys_to_delete:
        del data_mapping[key]
    # Step 3: Adjust indexes for remaining features
    remaining_keys = list(data_mapping.keys())
    for i, key in enumerate(remaining_keys):
        data_mapping[key][1] = i

    categorical_data_encoded = []
    for key, value in data_mapping.items():
        if value[0] == 'categorical':
            col = clean_data[:,value[1]]
            new_cols = perform_one_hot_encoding(col)
            categorical_data_encoded = np.append(categorical_data_encoded, new_cols, axis=1)

    ### VARIANCE : CONST FEATURE ???

    continuous_data = get_type_features(clean_data, 'continuous')
    clean_data = np.concatenate(continuous_data, categorical_data_encoded, axis=1)

    return clean_data

def perform_one_hot_encoding(feature):
    unique_labels = np.unique(feature)  # Find unique labels in the input list
    num_unique_labels = len(unique_labels)
    num_samples = len(feature)

    # Create an empty array to hold the one-hot encoded values
    encoded = np.zeros((num_samples, num_unique_labels))

    for i, label in enumerate(feature):
        index = np.where(unique_labels == label)[0][0]  # Get the index of the label
        encoded[i, index] = 1  # Set the corresponding value to 1 for the label

    return encoded 

def build_k_indices(y, k_fold, seed):
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

# The following dictionary maps the feature name to its type and its index in the raw data

data_mapping = {
    "_AIDTST3": ["categorical", 320],
    "_PNEUMO2": ["categorical", 319],
    "_FLSHOT6": ["categorical", 318],
    "_RFSEAT3": ["categorical", 317],
    "_RFSEAT2": ["categorical", 316],
    "_LMTSCL1": ["categorical", 315],
    "_LMTWRK1": ["categorical", 314],
    "_LMTACT1": ["categorical", 313],
    "_PASTAE1": ["categorical", 312],
    "_PAREC1": ["categorical", 311],
    "_PASTRNG": ["categorical", 310],
    "_PA30021": ["categorical", 309],
    "_PA300R2": ["categorical", 308],
    "_PA150R2": ["categorical", 307],
    "_PAINDX1": ["categorical", 306],
    "_PACAT1": ["categorical", 305],
    "PA1VIGM_": ["categorical", 304],
    "PAVIG21_": ["categorical", 303],
    "PAVIG11_": ["categorical", 302],
    "PA1MIN_": ["categorical", 301],
    "PAMIN21_": ["categorical", 300],
    "PAMIN11_": ["categorical", 299],
    "PAMISS1_": ["categorical", 298],
    "STRFREQ_": ["continuous", 297],
    "_MINAC21": ["categorical", 296],
    "_MINAC11": ["categorical", 295],
    "PAFREQ2_": ["continuous", 294],
    "PAFREQ1_": ["continuous", 293],
    "PADUR2_": ["categorical", 292],
    "PADUR1_": ["categorical", 291],
    "ACTIN21_": ["categorical", 290],
    "ACTIN11_": ["categorical", 289],
    "FC60_": ["continuous", 288],
    "MAXVO2_": ["continuous", 287],
    "METVL21_": ["continuous", 286],
    "METVL11_": ["continuous", 285],
    "_TOTINDA": ["categorical", 284],
    "_VEGETEX": ["categorical", 283],
    "_FRUITEX": ["categorical", 282],
    "_VEG23": ["categorical", 281],
    "_FRT16": ["categorical", 270],
    "_VEGLT1": ["categorical", 279],
    "_FRTLT1": ["categorical", 278],
    "_VEGESUM": ["continuous", 277],
    "_FRUTSUM": ["continuous", 276],
    "_VEGRESP": ["categorical", 275],
    "_FRTRESP": ["categorical", 274],
    "_MISVEGN": ["categorical", 273],
    "_MISFRTN": ["categorical", 272],
    "VEGEDA1_": ["continuous", 271],
    "ORNGDAY_": ["continuous", 270],
    "GRENDAY_": ["continuous", 269],
    "BEANDAY_": ["continuous", 268],
    "FRUTDA1_": ["continuous", 267],
    "FTJUDA1_": ["continuous", 266],
    "_RFDRHV5": ["categorical", 265],
    "_DRNKWEK": ["continuous", 264],
    "_RFBING5": ["categorical", 263],
    "DROCDY3_": ["categorical", 262],
    "DRNKANY5": ["categorical", 261],
    "_RFSMOK3": ["categorical", 260],
    "_SMOKER3": ["categorical", 259],
    "_INCOMG": ["categorical", 258],
    "_EDUCAG": ["categorical", 257],
    "_CHLDCNT": ["categorical", 256],
    "_RFBMI5": ["categorical", 255],
    "_BMI5CAT": ["categorical", 254],
    "_BMI5": ["continuous", 253],
    "WTKG3": ["continuous", 252],
    "HTM4": ["continuous", 251],
    "HTIN4": ["categorical", 250],
    "_AGE_G": ["categorical", 249],
    "_AGE80": ["categorical", 248],
    "_AGE65YR": ["categorical", 247],
    "_AGEG5YR": ["categorical", 246],
    "_RACE_G1": ["categorical", 245],
    "_RACEGR3": ["categorical", 244],
    "_RACEG21": ["categorical", 243],
    "_RACE": ["categorical", 242],
    "_HISPANC": ["categorical", 241],
    "_MRACE1": ["categorical", 240],
    "_PRACE1": ["categorical", 239],
    "_DRDXAR1": ["categorical", 238],
    "_ASTHMS1": ["categorical", 237],
    "_CASTHM1": ["categorical", 236],
    "_LTASTH1": ["categorical", 235],
    "_RFCHOL": ["categorical", 234],
    "_CHOLCHK": ["categorical", 233],
    "_RFHYPE5": ["categorical", 232],
    "_HCVU651": ["categorical", 231],
    "_RFHLTH": ["categorical", 230],
    "_LLCPWT": ["delete", 229],
    "_DUALCOR": ["delete", 228],
    "_DUALUSE": ["categorical", 227],
    "_CLLCPWT": ["delete", 226],
    "_CPRACE": ["categorical", 225],
    "_CRACE1": ["categorical", 224],
    "_CHISPNC": ["categorical", 223],
    "_WT2RAKE": ["delete", 222],
    "_RAWRAKE": ["delete", 221],
    "_STRWT": ["delete", 220],
    "_STSTR": ["delete", 219],
    "MSCODE": ["categorical", 218],
    "QSTLANG": ["categorical", 217],
    "QSTVER": ["categorical", 216],
    "ADANXEV": ["categorical", 215],
    "MISTMNT": ["categorical", 214],
    "ADMOVE": ["categorical", 213],
    "ADTHINK": ["categorical", 212],
    "ADFAIL": ["categorical", 211],
    "ADEAT1": ["categorical", 210],
    "ADENERGY": ["categorical", 209],
    "ADSLEEP": ["categorical", 208],
    "ADDOWN": ["categorical", 207],
    "ADPLEASR": ["categorical", 206],
    "LSATISFY": ["categorical", 205],
    "EMTSUPRT": ["categorical", 204],
    "CASTHNO2": ["categorical", 203],
    "CASTHDX2": ["categorical", 202],
    "RCSRLTN2": ["categorical", 201],
    "RCSGENDR": ["categorical", 200],
    "TRNSGNDR": ["categorical", 199],
    "SXORIENT": ["categorical", 198],
    "SCNTLWK1": ["categorical", 197],
    "SCNTLPAD": ["categorical", 196],
    "SCNTWRK1": ["categorical", 195],
    "SCNTPAID": ["categorical", 194],
    "SCNTMEL1": ["categorical", 193],
    "SCNTMNY1": ["categorical", 192],
    "PCDMDECN": ["categorical", 191],
    "PCPSADE1": ["categorical", 190],
    "PCPSARS1": ["categorical", 189],
    "PSATIME": ["categorical", 188],
    "PSATEST1": ["categorical", 187],
    "PCPSARE1": ["categorical", 186],
    "PCPSADI1": ["categorical", 185],
    "PCPSAAD2": ["categorical", 184],
    "LASTSIG3": ["categorical", 183],
    "HADSGCO1": ["categorical", 182],
    "HADSIGM3": ["categorical", 181],
    "LSTBLDS3": ["categorical", 180],
    "BLDSTOOL": ["categorical", 179],
    "LENGEXAM": ["categorical", 178],
    "PROFEXAM": ["categorical", 177],
    "HADHYST2": ["categorical", 176],
    "HPLSTTST": ["categorical", 175],
    "HPVTEST": ["categorical", 174],
    "LASTPAP2": ["categorical", 173],
    "HADPAP2": ["categorical", 172],
    "HOWLONG": ["categorical", 171],
    "HADMAM": ["categorical", 170],
    "SHINGLE2": ["categorical", 169],
    "HPVADSHT": ["categorical", 168],
    "HPVADVC2": ["categorical", 167],
    "TETANUS": ["categorical", 166],
    "ARTHEDU": ["categorical", 165],
    "ARTHEXER": ["categorical", 164],
    "ARTHWGT": ["categorical", 163],
    "ARTTODAY": ["categorical", 162],
    "RDUCSTRK": ["categorical", 161],
    "RDUCHART": ["categorical", 160],
    "RLIVPAIN": ["categorical", 159],
    "ASPUNSAF": ["categorical", 158],
    "CVDASPRN": ["categorical", 157],
    "STREHAB1": ["categorical", 156],
    "HAREHAB1": ["categorical", 155],
    "ASINHALR": ["categorical", 154],
    "ASTHMED3": ["categorical", 153],
    "ASNOSLEP": ["categorical", 152],
    "ASYMPTOM": ["categorical", 151],
    "ASACTLIM": ["categorical", 150],
    "ASRCHKUP": ["categorical", 149],
    "ASDRVIST": ["categorical", 148],
    "ASERVIST": ["categorical", 147],
    "ASATTACK": ["categorical", 146],
    "ASTHMAGE": ["categorical", 145],
    "DRADVISE": ["categorical", 144],
    "LONGWTCH": ["categorical", 143],
    "WTCHSALT": ["categorical", 142],
    "CDDISCUS": ["categorical", 141],
    "CDSOCIAL": ["categorical", 140],
    "CDHELP": ["categorical", 139],
    "CDASSIST": ["categorical", 138],
    "CDHOUSE": ["categorical", 137],
    "CIMEMLOS": ["categorical", 136],
    "VIMACDG2": ["categorical", 135],
    "VIGLUMA2": ["categorical", 134],
    "VICTRCT4": ["categorical", 133],
    "VIINSUR2": ["categorical", 132],
    "VIEYEXM2": ["categorical", 131],
    "VINOCRE2": ["categorical", 130],
    "VIPRFVS2": ["categorical", 129],
    "VIREDIF3": ["categorical", 128],
    "VIDFCLT2": ["categorical", 127],
    "CRGVEXPT": ["categorical", 126],
    "CRGVMST2": ["categorical", 125],
    "CRGVHOUS": ["categorical", 124],
    "CRGVPERS": ["categorical", 123],
    "CRGVPRB1": ["categorical", 122],
    "CRGVHRS1": ["categorical", 121],
    "CRGVLNG1": ["categorical", 120],
    "CRGVREL1": ["categorical", 119],
    "CAREGIV1": ["categorical", 118],
    "DIABEDU": ["categorical", 117],
    "DIABEYE": ["categorical", 116],
    "EYEEXAM": ["categorical", 115],
    "FEETCHK": ["categorical", 114],
    "CHKHEMO3": ["categorical", 113],
    "DOCTDIAB": ["categorical", 112],
    "FEETCHK2": ["categorical", 111],
    "BLDSUGAR": ["categorical", 110],
    "INSULIN": ["categorical", 109],
    "PREDIAB1": ["categorical", 108],
    "PDIABTST": ["categorical", 107],
    "WHRTST10": ["categorical", 106],
    "HIVTSTD3": ["categorical", 105],
    "HIVTST6": ["categorical", 104],
    "PNEUVAC3": ["categorical", 103],
    "IMFVPLAC": ["categorical", 102],
    "FLSHTMY2": ["categorical", 101],
    "FLUSHOT6": ["categorical", 100],
    "SEATBELT": ["delete", 99],
    "JOINPAIN": ["categorical", 98],
    "ARTHSOCL": ["categorical", 97],
    "ARTHDIS2": ["categorical", 96],
    "LMTJOIN3": ["categorical", 95],
    "STRENGTH": ["categorical", 94],
    "EXERHMM2": ["categorical", 93],
    "EXEROFT2": ["categorical", 92],
    "EXRACT21": ["categorical", 91],
    "EXERHMM1": ["categorical", 90],
    "EXEROFT1": ["categorical", 89],
    "EXRACT11": ["categorical", 88],
    "EXERANY2": ["categorical", 87],
    "VEGETAB1": ["categorical", 86],
    "FVORANG": ["categorical", 85],
    "FVGREEN": ["categorical", 84],
    "FVBEANS": ["categorical", 83],
    "FRUIT1": ["categorical", 82],
    "FRUITJU1": ["categorical", 81],
    "MAXDRNKS": ["categorical", 80],
    "DRNK3GE5": ["categorical", 79],
    "AVEDRNK2": ["categorical", 78],
    "ALCDAY5": ["categorical", 77],
    "USENOW3": ["categorical", 76],
    "LASTSMK2": ["categorical", 75],
    "STOPSMK2": ["categorical", 74],
    "SMOKDAY2": ["categorical", 73],
    "SMOKE100": ["categorical", 72],
    "DIFFALON": ["categorical", 71],
    "DIFFDRES": ["categorical", 70],
    "DIFFWALK": ["categorical", 69],
    "DECIDE": ["categorical", 68],
    "BLIND": ["categorical", 67],
    "USEEQUIP": ["categorical", 66],
    "QLACTLM2": ["categorical", 65],
    "PREGNANT": ["categorical", 64],
    "HEIGHT3": ["delete", 63],
    "WEIGHT2": ["delete", 62],
    "INTERNET": ["categorical", 61],
    "INCOME2": ["categorical", 60],
    "CHILDREN": ["categorical", 59],
    "EMPLOY1": ["categorical", 58],
    "VETERAN3": ["categorical", 57],
    "CPDEMO1": ["categorical", 56],
    "NUMPHON2": ["delete", 55],
    "NUMHHOL2": ["delete", 54],
    "RENTHOM1": ["delete", 53],
    "EDUCA": ["categorical", 52],
    "MARITAL": ["categorical", 51],
    "SEX": ["categorical", 50],
    "DIABAGE2": ["categorical", 49],
    "DIABETE3": ["categorical", 48],
    "CHCKIDNY": ["categorical", 47],
    "ADDEPEV2": ["categorical", 46],
    "HAVARTH3": ["categorical", 45],
    "CHCCOPD1": ["categorical", 44],
    "CHCOCNCR": ["categorical", 43],
    "CHCSCNCR": ["categorical", 42],
    "ASTHNOW": ["categorical", 41],
    "ASTHMA3": ["categorical", 40],
    "CVDSTRK3": ["categorical", 39],
    "TOLDHI2": ["categorical", 38],
    "CHOLCHK": ["categorical", 37],
    "BLOODCHO": ["categorical", 36],
    "BPMEDS": ["categorical", 35],
    "BPHIGH4": ["categorical", 34],
    "CHECKUP1": ["categorical", 33],
    "MEDCOST": ["categorical", 32],
    "PERSDOC2": ["categorical", 31],
    "HLTHPLN1": ["categorical", 30],
    "POORHLTH": ["categorical", 29],
    "MENTHLTH": ["categorical", 28],
    "PHYSHLTH": ["categorical", 27],
    "GENHLTH": ["categorical", 26],
    "HHADULT": ["categorical", 25],
    "LANDLINE": ["delete", 24],
    "CSTATE": ["categorical", 23],
    "CCLGHOUS": ["delete", 22],
    "PVTRESD2": ["categorical", 21],
    "CADULT": ["delete", 20],
    "CELLFON2": ["delete", 19],
    "CTELNUM1": ["delete", 18],
    "NUMWOMEN": ["categorical", 17],
    "NUMMEN": ["categorical", 16],
    "NUMADULT": ["categorical", 15],
    "LADULT": ["delete", 14],
    "CELLFON3": ["delete", 13],
    "STATERES": ["delete", 12],
    "COLGHOUS": ["delete", 11],
    "PVTRESD1": ["categorical", 10],
    "CTELENUM": ["delete", 9],
    "_PSU": ["delete", 8],
    "SEQNO": ["delete", 7],
    "DISPCODE": ["delete", 6],
    "IYEAR": ["categorical", 5],
    "IDAY": ["delete", 4],
    "IMONTH": ["categorical", 3],
    "IDATE": ["delete", 2],
    "FMONTH": ["delete", 1],
    "_STATE": ["categorical", 0]
}

# The following dictionary maps some features to the ones they are supposed to be correlated with
# in order to remove them from the data (it will be checked in the process of cleaning the data)

correlated_with = {
    "_AIDTST3": "HIVTST6",
    "_PNEUMO2": "PNEUVAC3",
    "_FLSHOT6": "FLUSHOT6",
    "_PASTAE1": "_PAREC1",
    "_PA30021": "_PA300R2",
    "_TOTINDA": "EXERANY2",
    "_RFDRHV5": "SEX, _DRNKWEK",
    "_DRNKWEK": ("ALCDAY5", "AVEDRNK2", "DROCDY3_"),
    "_RFBING5": ("ALCDAY", "DRNK3GE5"),
    "DRNKANY5": "ALCDAY5",
    "_RFSMOK3": "_SMOKER3",
    "_SMOKER3": ("SMOKE100", "SMOKEDAY"),
    "_INCOMG": "INCOME2",
    "_EDUCAG": "EDUCA",
    "_CHLDCNT": "CHILDREN",
    "_RFBMI5": "_BMI5",
    "_BMI5CAT": "_BMI5",
    "HTIN4": "HEIGHT3",
    "_AGE_G": "_IMPAGE",
    "_AGE65YR": "AGE",
    "_AGEG5YR": "AGE",
    "_RACEGR3": "_RACE_G1",
    "_RACEG21": "_RACE",
    "_RACE": ("_HISPANC", "_MRACE1"),
    "_MRACE1": "MRACASC1",
    "_PRACE1": "MRACASC1",
    "_DRDXAR1": "HAVARTH3",
    "_CASTHM1": ("ASTHMA3", "ASTHNOW"),
    "_LTASTH1": "ASTHMA3",
    "_RFCHOL": ("BLOODCHO", "TOLDHI2"),
    "_CHOLCHK": ("BLOODCHO", "CHOLCHK"),
    "_RFHYPE5": "BPHIGH4",
    "_HCVU651": ("AGE", "HLTHPLN1"),
    "_RFHLTH": "GENHLTH"
}
