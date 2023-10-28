[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/U9FTc9i_)


### Code structure

#### `scripts/basics.py`:

- train model : train(model, y, x, initial_w = 0, max_iters = 0, gamma = 0, lambda_ = 0, cost = 'mse')              
- evaluate accuracy + f1 score: ???? jsp à quel point on le veut ou on garde metrics?

#### `scripts/cleaning.py`:
- data mapping dictionnary as {'feature_ID' : [type, index]} with type in ['continuous', 'categorical', 'delete'] which were estimated from hand annotation of the dataset, 'delete' corresponding to unusable data (????? à preciser)
- obtain 'categorical' or 'continuous' data: get_type_features(data, type_)
- remove the 'delete' type features: remove_useless_features(data)
- remove features with ratio of NaNs above threshold parameter: remove_nan_columns(data, threshold=0.8)???? checker le threshold par defaut
- mean imputation on (continuous) data: complete(data)
- removing low-deviation continuous features with std < threshold_ratio * maximum std: remove_constant_continuous(data_array, threshold_ratio=0.001)??????? 
- removing the features which have a high maximal frequency for a class: remove_constant_categorical(data, threshold=0.001)
- ????YANN DECRIRE ????? delete_correlated_features(data)
- OneHotEncoder(categorical_data)
- standardizing (continuous) data: standardize(data)
- ????YANN DECRIRE ????? clean_data_mapping()
- ????? dire pour le dictionnaire sans les delete pour les indexes
- applying preprocessing steps to data: ???? clean data function


#### `scripts/helper.py`:
- compute mse                       ????a completer
- compute mae                       ????a completer
- compute gradient                  ????a completer
- batch-iter                        ????a completer
- compute stoch gradient            ????a completer
- load csv des exos (to move)       ????a completer
- create csv submissions (to move)  ????a completer
- cross validation (in progress)    ????a completer
- find error points indices     DONE   ????? jsp à quel point utile


#### `scripts/implementation.py`:
- mean_squared_error_gd     ????a completer
- mean_squared_error_sgd        ????a completer
- least_squares     ????a completer
- ridge_regression      ????a completer
- logistic_regression           check for loss -1,1/0,1???      ????a completer
- reg_logistic_regression       check for loss -1,1/0,1????     ????a completer

#### `scripts/metrics.py`:
- confusion_matrix(y_true, y_pred)  - ???? maybe useless
- calculate_f1_score(y_true, y_pred) - ????a completer
- calcuate_precision(y_true, y_pred) - ????a completer
- calculate_specificity(y_true, y_pred)- ????a completer
- caclculate_sensitivity(y_true, y_pred)- ????a completer
- calculate_accuracy(y_true, y_pred)- ????a completer
- calculate_list_of_metrics(y_true, y_pred):- ????a completer


#### `scripts/preprocessing.py`:
- loading the csv data into an array: load_csv_data(data_path, sub_sample=False)
- split training data into train and test sets: split_data(x, y, ratio, seed=1)
- split data for k-folds cross validation: build_k_indices(y, k_fold, seed=1)
- PCA decomposition: run_pca(x, n_components, fig_name = 'PCA_variance_ratios', visualisation = False)
- create submission file from prediction array: create_csv_submission(ids, y_pred, name)

#### `scripts/run.py`:
can run and will give all submission files we submit and talk about + the confusion matrices + our plots in a plots folder
- ????a completer

#### `scipts/visualisation.py`:
- plotting confusion matrices/f1/accuracy?  - ????a completer
- plotting pca results          DONE        - ????a completer
- plotting data with color code true false etc  - ????a completer
- plotting cross validation results figure with different losses - ????a completer





