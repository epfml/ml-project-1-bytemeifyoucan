#preprocessing

#ad params for preprocessing like pca etc

# for each model 
#find best parameters w cross validation it param set to true
#hyperparam_tuning: True
#if false use the ones we already found 

#for each model
# find best hyperparam
# run to get predictions
# evaluate our predictions
# make submission 

#do multiple runs with different preprocessing steps (ratio split, ratio for nan, threshold for const)



# we need to end up w/
# - figures for cv for each param with other ones fixed
# - best hyperparameter list if cv set to true 
# - figures for pca value decomposition evolution to show the optimal components
# - plot of metrics for each model 