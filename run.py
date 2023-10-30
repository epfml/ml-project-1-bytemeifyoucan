from basics import *
from implementations import *
from processing import *
from cleaning import *
from metrics import *
from helper import *
from definitions import ROOT_DIR

import implementations as imp
import helper as hp

dataset_path = os.path.join(ROOT_DIR, 'dataset_to_release')
x_tr, x_te, y_tr, tr_ids, te_ids = load_csv_data(dataset_path, False)

cv = False #for the sake of time, if set to False will use premade list of best parameters already computed. Those same parameters can be found by keeping cv to True. 
split_ratio = 0.8 #80% of the dataset will be used to train the model, the rest will be used to evaluate performance
nan_threshold = 0.8 #to filter our features with more nan than the threshold
max_unique_values = 50 
remove_const = False
const_thresholds = [0.0000001, 0.000001] #to filter constant continuous , when filtering out features w too many categories
n_components = False
PCA = False #if set to True will run PCA on continuous features and keep principal components until 99% of variance is explained
pca_thresholds = 99 #in % explained variance
correlation_threshold = 0.9 #correlation threshold

models = ['gradient descent', 'stochastic gradient descent', 'least squares', 'ridge regression',  'logistic regression', 'reg logistic regression']
mapping_threshold = 0

#clean and split
clean_train = clean(x_tr, nan_threshold, 
                    remove_const, const_thresholds, 
                    PCA, n_components, pca_thresholds, 
                    max_unique_values, correlation_threshold)

clean_x_tr, clean_x_te, y_tr_set, y_te_set = split_data(clean_train, y_tr, split_ratio)

#tune hyperparameters
best_params = {'gradient descent' : [0.01, 0.7], 
               'stochastic gradient descent': [0.01, 0.7],
               'least squares': [0.01, 0.7],
               'ridge regression': [0.01, 0.7],
               'logistic regression': [0.01, 0.7],
               'reg logistic regression': [0.01, 0.7]}

k_fold = 4
max_iters = 500000
lambdas = np.linspace(0, 1, 10)
gammas = np.linspace(0.5,1.5,5)

if cv:
    best_params = {}
    for model in models:
        losses = compute_losses_for_hyperparameters(model, y_tr_set, clean_x_tr, k_fold, max_iters, lambdas, gammas)
        print(losses.shape)
        best_params['model'] = [find_best_hyperparameters(losses)[1:3]]

#use entirety of train and make submission
limit = x_tr.shape[0]

total_data = np.concatenate((x_tr, x_te), axis = 0)

clean_total = clean(total_data, nan_threshold, 
                    remove_const, const_thresholds, 
                    PCA, n_components, pca_thresholds, 
                    max_unique_values, correlation_threshold)

tot_clean_train = clean_total[:limit]

tot_clean_test = clean_total[limit:]

#run on train and predict + submit prediction
for model in models:
    lambda_ = best_params[model][0]
    gamma = best_params[model][1]
    initial_w = np.zeros(y_tr)
    w, loss = train(model, y_tr, tot_clean_train, initial_w, max_iters, gamma, lambda_)
    y_pred = np.where(np.dot(tot_clean_test,w) < mapping_threshold, -1, 1)
    create_csv_submission([i for i in range(len(y_pred))], y_pred, model.replace( , '_') + '_submission.csv')
    
