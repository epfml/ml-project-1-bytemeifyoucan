[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/U9FTc9i_)


### Code structure

The dataset is expected to be in a folder 'dataset_to_release' 

#### `cleaning.py`:
- data mapping dictionnary as {'feature_ID' : [type, index]} with type in ['continuous', 'categorical', 'delete'] which were estimated from hand annotation of the dataset, 'delete' corresponding to unusable data
- correlated_with dictionnary, containing feature ID for correlated features
- obtain 'categorical' or 'continuous' data: get_type_features(data, type_)
- remove the 'delete' type features: remove_useless_features(data)
- remove features with ratio of NaNs above threshold parameter: remove_nan_columns(data, threshold=0.8)
- mean imputation on (continuous) data: complete(data)
- removing low-deviation continuous features with std < threshold_ratio * maximum std: remove_constant_continuous(data_array, threshold_ratio=0.001)
- removing the features which have a high maximal frequency for a class: remove_constant_categorical(data, threshold=0.001)
- remove correlated features in correlated_with dictionnary: delete_correlated_features(data)
- OneHotEncode data: OneHotEncoder(data) 
- standardizing (continuous) data: standardize(data)
- updating data_mapping dictionnary indexes after removing 'delete' features: clean_data_mapping()
- delete correlated columns calculating correlation matrix: remove_correlated_columns(data, correlation_threshold)
- applying preprocessing steps to data: clean(data, nan_threshold, remove_const, const_thresholds, PCA, n_components, pca_threshold, max_unique_values,correlation_threshold)
- apply preprocessing steps to continuous features: clean_continuous(data, nan_threshold, n_components, pca_threshold, correlation_threshold)

#### `definitions.py`:
contains absolue path to repository

#### `helper.py`:
- load dataset: load_csv_data(data_path, sub_sample=False)
- create submission file: create_csv_submission(ids, y_pred, name)
- compute mse:compute_mse(e)
- compute mae:compute_mae(e)
- compute gradient:compute_gradient(y, tx, w)
- Generate a minibatch iterator for a dataset: batch_iter(y, tx, batch_size, num_batches=1, shuffle=True)
- compute stochastic gradient: compute_stoch_gradient(y, tx, w)
- sigmoid function: sigmoid(t)
- obtain for a model, a choice of lambdas and gamma, the test and train losses: compute_losses_for_hyperparameters(model, y, tx, k_fold, max_iters=0, lambdas = ['Nan'], gammas = ['Nan'], seed = 1)
- obtain best hyperparameters based on test loss: find_best_hyperparameters(hyperparameter_losses)
- train model : train(model, y, x, initial_w = 0, max_iters = 0, gamma = 0, lambda_ = 0, cost = 'mse')  


#### `implementation.py`:
It contains the 6 functions in project1 description
- linear regression using gradient descent: mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)
- linear regression using stochastic gradient descent: mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)
- least squares regression using normal equations: least_squares(y, tx)
- ridge regression using normal equations: ridge_regression(y, tx, lambda_, cost = 'mse')
- logistic regression using gradient descent or SGD: logistic_regression(y, tx, initial_w, max_iters, gamma)
- regularized logistic regression using gradient descent or SGD: reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)


#### `metrics.py`:
- obtain confusion matrice: confusion_matrix(y_true, y_pred)  
- compute F1 score: calculate_f1_score(y_true, y_pred) 
- compute precision: calcuate_precision(y_true, y_pred) 
- compute specificity: calculate_specificity(y_true, y_pred)
- compute sensitivity: calculate_sensitivity(y_true, y_pred)
- compute accuracy: calculate_accuracy(y_true, y_pred)
- compute all of the metrics above: calculate_list_of_metrics(y_true, y_pred)


#### `processing.py`:
- split training data into train and test sets: split_data(x, y, ratio, seed=1)
- split data for k-folds cross validation: build_k_indices(y, k_fold, seed=1)
- obtain loss for a given index k for k-fold cross validation: cv_loss(model, y, x, k_indices, k, lambda_, max_iters, gamma)
- PCA decomposition: run_pca(x, n_components, fig_name = 'PCA_variance_ratios', visualisation = False)


#### `run.py`:
Runing this file will create our submission csv (not on AIcrowd because failed to submit)


#### `visualisation.py`:
- plotting pca results: plot_pca(n_components, eigenvalues, visualisation, fig_name)
- plotting test vs train error when runing cross validation: plot_train_test(model, train_errors, test_errors, lambdas, gammas=0, visualisation = False)
- plotting metrics for a given model: plot_metrics(model, metrics, thresholds, visualisation = False)

#### `obtain_metrics.ipynb`:
Run to obtain and print metrics for training and testing on sampled training dataset



