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
#remove the useless defined by yann
#remove nan columns
#remove constant columns
#remove correlated columns
#for categorical values do one hot encoding
#for continuous values add value from same distrib/moyenne -> go check indices 
#split avant de standardize -> no split for testx
#standardize


def remove_nan_columns(data,threshold=0.8):
    #remove features with threshold for nan
    nan_ratio = np.sum(np.isnan(data), axis = 0)/ data.shape[0]
    columns_to_remove = np.where(nan_ratio > threshold)[0]
    without_nan = np.delete(data, columns_to_remove, axis = 1)
    return without_nan



# complete datapoints for numerical features w const std et mean -> or knn? knn might take too much to run

#remove correlated columns (try to always kick our the catsgorical one)

#remove data with less than 0.25% variation - look at std


# standardize numerical

#find 10 columns that explain the most variance (not pc)

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

def standardize_data():
    return 0
    
def normalise_data():
    return 0
    
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

data_mapping = {
    "_AIDTST3": ["categorical", 330],
    "_PNEUMO2": ["categorical", 329],
    "_FLSHOT6": ["categorical", 328],
    "_RFSEAT3": ["categorical", 327],
    "_RFSEAT2": ["categorical", 326],
    "_LMTSCL1": ["categorical", 325],
    "_LMTWRK1": ["categorical", 324],
    "_LMTACT1": ["categorical", 323],
    "_PASTAE1": ["categorical", 322],
    "_PAREC1": ["categorical", 321],
    "_PASTRNG": ["categorical", 320],
    "_PA30021": ["categorical", 319],
    "_PA300R2": ["categorical", 318],
    "_PA150R2": ["categorical", 317],
    "_PAINDX1": ["categorical", 316],
    "_PACAT1": ["categorical", 315],
    "PA1VIGM_": ["categorical", 314],
    "PAVIG21_": ["categorical", 313],
    "PAVIG11_": ["categorical", 312],
    "PA1MIN_": ["categorical", 311],
    "PAMIN21_": ["categorical", 310],
    "PAMIN11_": ["categorical", 309],
    "PAMISS1_": ["categorical", 308],
    "STRFREQ_": ["continuous", 307],
    "_MINAC21": ["categorical", 306],
    "_MINAC11": ["categorical", 305],
    "PAFREQ2_": ["continuous", 304],
    "PAFREQ1_": ["continuous", 303],
    "PADUR2_": ["categorical", 302],
    "PADUR1_": ["categorical", 301],
    "ACTIN21_": ["categorical", 300],
    "ACTIN11_": ["categorical", 299],
    "FC60_": ["continuous", 298],
    "MAXVO2_": ["continuous", 297],
    "METVL21_": ["continuous", 296],
    "METVL11_": ["continuous", 295],
    "_TOTINDA": ["categorical", 294],
    "_VEGETEX": ["categorical", 293],
    "_FRUITEX": ["categorical", 292],
    "_VEG23": ["categorical", 291],
    "_FRT16": ["categorical", 290],
    "_VEGLT1": ["categorical", 289],
    "_FRTLT1": ["categorical", 288],
    "_VEGESUM": ["continuous", 287],
    "_FRUTSUM": ["continuous", 286],
    "_VEGRESP": ["categorical", 285],
    "_FRTRESP": ["categorical", 284],
    "_MISVEGN": ["categorical", 283],
    "_MISFRTN": ["categorical", 282],
    "VEGEDA1_": ["continuous", 281],
    "ORNGDAY_": ["continuous", 280],
    "GRENDAY_": ["continuous", 279],
    "BEANDAY_": ["continuous", 278],
    "FRUTDA1_": ["continuous", 277],
    "FTJUDA1_": ["continuous", 276],
    "_RFDRHV5": ["categorical", 275],
    "_DRNKWEK": ["continuous", 274],
    "_RFBING5": ["categorical", 273],
    "DROCDY3_": ["categorical", 272],
    "DRNKANY5": ["categorical", 271],
    "_RFSMOK3": ["categorical", 270],
    "_SMOKER3": ["categorical", 269],
    "_INCOMG": ["categorical", 268],
    "_EDUCAG": ["categorical", 267],
    "_CHLDCNT": ["categorical", 266],
    "_RFBMI5": ["categorical", 265],
    "_BMI5CAT": ["categorical", 264],
    "_BMI5": ["continuous", 263],
    "WTKG3": ["continuous", 262],
    "HTM4": ["continuous", 261],
    "HTIN4": ["categorical", 260],
    "_AGE_G": ["categorical", 259],
    "_AGE80": ["categorical", 258],
    "_AGE65YR": ["categorical", 257],
    "_AGEG5YR": ["categorical", 256],
    "_RACE_G1": ["categorical", 255],
    "_RACEGR3": ["categorical", 254],
    "_RACEG21": ["categorical", 253],
    "_RACE": ["categorical", 252],
    "_HISPANC": ["categorical", 251],
    "_MRACE1": ["categorical", 250],
    "_PRACE1": ["categorical", 249],
    "_DRDXAR1": ["categorical", 248],
    "_ASTHMS1": ["categorical", 247],
    "_CASTHM1": ["categorical", 246],
    "_LTASTH1": ["categorical", 245],
    "_MICHD": ["categorical", 244],
    "_RFCHOL": ["categorical", 243],
    "_CHOLCHK": ["categorical", 242],
    "_RFHYPE5": ["categorical", 241],
    "_HCVU651": ["categorical", 240],
    "_RFHLTH": ["categorical", 239],
    "_LLCPWT": ["delete", 238],
    "_DUALCOR": ["delete", 237],
    "_DUALUSE": ["categorical", 236],
    "_CLLCPWT": ["delete", 235],
    "_CPRACE": ["categorical", 234],
    "_CRACE1": ["categorical", 233],
    "_CHISPNC": ["categorical", 232],
    "_WT2RAKE": ["delete", 231],
    "_RAWRAKE": ["delete", 230],
    "_STRWT": ["delete", 229],
    "_STSTR": ["delete", 228],
    "MSCODE": ["categorical", 227],
    "EXACTOT2": ["delete", 226],
    "EXACTOT1": ["delete", 225],
    "QSTLANG": ["categorical", 224],
    "QSTVER": ["categorical", 223],
    "ADANXEV": ["categorical", 222],
    "MISTMNT": ["categorical", 221],
    "ADMOVE": ["categorical", 220],
    "ADTHINK": ["categorical", 219],
    "ADFAIL": ["categorical", 218],
    "ADEAT1": ["categorical", 217],
    "ADENERGY": ["categorical", 216],
    "ADSLEEP": ["categorical", 215],
    "ADDOWN": ["categorical", 214],
    "ADPLEASR": ["categorical", 213],
    "LSATISFY": ["categorical", 212],
    "EMTSUPRT": ["categorical", 211],
    "CASTHNO2": ["categorical", 210],
    "CASTHDX2": ["categorical", 209],
    "RCSRLTN2": ["categorical", 208],
    "RCSGENDR": ["categorical", 207],
    "TRNSGNDR": ["categorical", 206],
    "SXORIENT": ["categorical", 205],
    "SCNTLWK1": ["categorical", 204],
    "SCNTLPAD": ["categorical", 203],
    "SCNTWRK1": ["categorical", 202],
    "SCNTPAID": ["categorical", 201],
    "SCNTMEL1": ["categorical", 200],
    "SCNTMNY1": ["categorical", 199],
    "PCDMDECN": ["categorical", 198],
    "PCPSADE1": ["categorical", 197],
    "PCPSARS1": ["categorical", 196],
    "PSATIME": ["categorical", 195],
    "PSATEST1": ["categorical", 194],
    "PCPSARE1": ["categorical", 193],
    "PCPSADI1": ["categorical", 192],
    "PCPSAAD2": ["categorical", 191],
    "LASTSIG3": ["categorical", 190],
    "HADSGCO1": ["categorical", 189],
    "HADSIGM3": ["categorical", 188],
    "LSTBLDS3": ["categorical", 187],
    "BLDSTOOL": ["categorical", 186],
    "LENGEXAM": ["categorical", 185],
    "PROFEXAM": ["categorical", 184],
    "HADHYST2": ["categorical", 183],
    "HPLSTTST": ["categorical", 182],
    "HPVTEST": ["categorical", 181],
    "LASTPAP2": ["categorical", 180],
    "HADPAP2": ["categorical", 179],
    "HOWLONG": ["categorical", 178],
    "HADMAM": ["categorical", 177],
    "SHINGLE2": ["categorical", 176],
    "HPVADSHT": ["categorical", 175],
    "HPVADVC2": ["categorical", 174],
    "TETANUS": ["categorical", 173],
    "ARTHEDU": ["categorical", 172],
    "ARTHEXER": ["categorical", 171],
    "ARTHWGT": ["categorical", 170],
    "ARTTODAY": ["categorical", 169],
    "RDUCSTRK": ["categorical", 168],
    "RDUCHART": ["categorical", 167],
    "RLIVPAIN": ["categorical", 166],
    "ASPUNSAF": ["categorical", 165],
    "CVDASPRN": ["categorical", 164],
    "STREHAB1": ["categorical", 163],
    "HAREHAB1": ["categorical", 162],
    "ASINHALR": ["categorical", 161],
    "ASTHMED3": ["categorical", 160],
    "ASNOSLEP": ["categorical", 159],
    "ASYMPTOM": ["categorical", 158],
    "ASACTLIM": ["categorical", 157],
    "ASRCHKUP": ["categorical", 156],
    "ASDRVIST": ["categorical", 155],
    "ASERVIST": ["categorical", 154],
    "ASATTACK": ["categorical", 153],
    "ASTHMAGE": ["categorical", 152],
    "DRADVISE": ["categorical", 151],
    "LONGWTCH": ["categorical", 150],
    "WTCHSALT": ["categorical", 149],
    "CDDISCUS": ["categorical", 148],
    "CDSOCIAL": ["categorical", 147],
    "CDHELP": ["categorical", 146],
    "CDASSIST": ["categorical", 145],
    "CDHOUSE": ["categorical", 144],
    "CIMEMLOS": ["categorical", 143],
    "VIMACDG2": ["categorical", 142],
    "VIGLUMA2": ["categorical", 141],
    "VICTRCT4": ["categorical", 140],
    "VIINSUR2": ["categorical", 139],
    "VIEYEXM2": ["categorical", 138],
    "VINOCRE2": ["categorical", 137],
    "VIPRFVS2": ["categorical", 136],
    "VIREDIF3": ["categorical", 135],
    "VIDFCLT2": ["categorical", 134],
    "CRGVEXPT": ["categorical", 133],
    "CRGVMST2": ["categorical", 132],
    "CRGVHOUS": ["categorical", 131],
    "CRGVPERS": ["categorical", 130],
    "CRGVPRB1": ["categorical", 129],
    "CRGVHRS1": ["categorical", 128],
    "CRGVLNG1": ["categorical", 127],
    "CRGVREL1": ["categorical", 126],
    "CAREGIV1": ["categorical", 125],
    "QLHLTH2": ["delete", 124],
    "QLSTRES2": ["delete", 123],
    "QLMENTL2": ["delete", 122],
    "PAINACT2": ["delete", 121],
    "DIABEDU": ["categorical", 120],
    "DIABEYE": ["categorical", 119],
    "EYEEXAM": ["categorical", 118],
    "FEETCHK": ["categorical", 117],
    "CHKHEMO3": ["categorical", 116],
    "DOCTDIAB": ["categorical", 115],
    "FEETCHK2": ["categorical", 114],
    "BLDSUGAR": ["categorical", 113],
    "INSULIN": ["categorical", 112],
    "PREDIAB1": ["categorical", 111],
    "PDIABTST": ["categorical", 110],
    "WHRTST10": ["categorical", 109],
    "HIVTSTD3": ["categorical", 108],
    "HIVTST6": ["categorical", 107],
    "PNEUVAC3": ["categorical", 106],
    "IMFVPLAC": ["categorical", 105],
    "FLSHTMY2": ["categorical", 104],
    "FLUSHOT6": ["categorical", 103],
    "SEATBELT": ["delete", 102],
    "JOINPAIN": ["categorical", 101],
    "ARTHSOCL": ["categorical", 100],
    "ARTHDIS2": ["categorical", 99],
    "LMTJOIN3": ["categorical", 98],
    "STRENGTH": ["categorical", 97],
    "EXERHMM2": ["categorical", 96],
    "EXEROFT2": ["categorical", 95],
    "EXRACT21": ["categorical", 94],
    "EXRACT21": ["categorical", 93],
    "EXERHMM1": ["categorical", 92],
    "EXEROFT1": ["categorical", 91],
    "EXRACT11": ["categorical", 90],
    "EXERANY2": ["categorical", 89],
    "VEGETAB1": ["categorical", 88],
    "FVORANG": ["categorical", 87],
    "FVGREEN": ["categorical", 86],
    "FVBEANS": ["categorical", 85],
    "FRUIT1": ["categorical", 84],
    "FRUITJU1": ["categorical", 83],
    "MAXDRNKS": ["categorical", 82],
    "DRNK3GE5": ["categorical", 81],
    "AVEDRNK2": ["categorical", 80],
    "ALCDAY5": ["categorical", 79],
    "USENOW3": ["categorical", 78],
    "LASTSMK2": ["categorical", 77],
    "STOPSMK2": ["categorical", 76],
    "SMOKDAY2": ["categorical", 75],
    "SMOKE100": ["categorical", 74],
    "DIFFALON": ["categorical", 73],
    "DIFFDRES": ["categorical", 72],
    "DIFFWALK": ["categorical", 71],
    "DECIDE": ["categorical", 70],
    "BLIND": ["categorical", 69],
    "USEEQUIP": ["categorical", 68],
    "QLACTLM2": ["categorical", 67],
    "PREGNANT": ["categorical", 66],
    "HEIGHT3": ["delete", 65],
    "WEIGHT2": ["delete", 64],
    "INTERNET": ["categorical", 63],
    "INCOME2": ["categorical", 62],
    "CHILDREN": ["categorical", 61],
    "EMPLOY1": ["categorical", 60],
    "VETERAN3": ["categorical", 59],
    "CPDEMO1": ["categorical", 58],
    "NUMPHON2": ["delete", 57],
    "NUMHHOL2": ["delete", 56],
    "RENTHOM1": ["delete", 55],
    "EDUCA": ["categorical", 54],
    "MARITAL": ["categorical", 53],
    "SEX": ["categorical", 52],
    "DIABAGE2": ["categorical", 51],
    "DIABETE3": ["categorical", 50],
    "CHCKIDNY": ["categorical", 49],
    "ADDEPEV2": ["categorical", 48],
    "HAVARTH3": ["categorical", 47],
    "CHCCOPD1": ["categorical", 46],
    "CHCOCNCR": ["categorical", 45],
    "CHCSCNCR": ["categorical", 44],
    "ASTHNOW": ["categorical", 43],
    "ASTHMA3": ["categorical", 42],
    "CVDSTRK3": ["categorical", 41],
    "CVDCRHD4": ["categorical", 40],
    "CVDINFR4": ["categorical", 39],
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
    "_STATE": ["categorical", 23],
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
    "_MICHD": ("CVDINFR4", "CVDCRHD4"),
    "_RFCHOL": ("BLOODCHO", "TOLDHI2"),
    "_CHOLCHK": ("BLOODCHO", "CHOLCHK"),
    "_RFHYPE5": "BPHIGH4",
    "_HCVU651": ("AGE", "HLTHPLN1"),
    "_RFHLTH": "GENHLTH"
}

def delete_correlated_features(y, x):

    key_to_delete = []

    for key in correlated_with:
        is_correlated = []
        for _key in correlated_with[key]:
            correlation = np.corrcoef(data_mapping[_key][1], data_mapping[key][1])
            if correlation > 0.3 :
                is_correlated.append(True)
            else:
                is_correlated.append(False)
        if np.all(is_correlated):
            key_to_delete = key  

    return key_to_delete