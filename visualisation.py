#plotting confusion matrices
#plotting pca results
#plotting data with color code true false etc
import matplotlib.pyplot as plt
import numpy as np
from definitions import ROOT_DIR
import os


def cross_validation_visualization(lambds, rmse_tr, rmse_te, visualisation, fig_name):
    #need to change path for saving
    """visualization the curves of rmse_tr and rmse_te."""
    plt.semilogx(lambds, rmse_tr, marker=".", color="b", label="train error")
    plt.semilogx(lambds, rmse_te, marker=".", color="r", label="test error")
    plt.xlabel("lambda")
    plt.ylabel("r mse")
    # plt.xlim(1e-4, 1)
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    if visualisation == False:
        plt.close()
    fig_path = os.path.join(ROOT_DIR, 'figures')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.savefig("cross_validation")
    
def plot_pca(n_components, eigenvalues, visualisation, fig_name):
    eigenvalues_relative = eigenvalues/np.sum(eigenvalues) * 100
    cum_eigenvalues = np.cumsum(eigenvalues_relative[: n_components])
    plt.plot(range(n_components), cum_eigenvalues, color = 'blue', marker = 'x')
    plt.ylabel("Variance explained (%)")
    plt.xlabel("Principal Component")
    plt.ticklabel_format(axis='y', style='plain')
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.title('Principal Component Analysis')
    plt.grid(True)
    if visualisation == False:
        plt.close()
    fig_path = os.path.join(ROOT_DIR, 'figures')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(os.path.join(fig_path, fig_name))


def plot_train_test(model, train_errors, test_errors, lambdas, gammas=0):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set
    """
    if(len(gammas) > 7):
        raise ValueError("You can plot up to 7 different gammas on the same graph")
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    for i, gamma in enumerate(gammas):
        plt.semilogx(lambdas, train_errors[i], color=colors[i], linestyle='--', marker='.', label="Train error for gamma = " + str(gamma))
        plt.semilogx(lambdas, test_errors[i], color=colors[i], linestyle='-', marker='.', label="Test error for gamma = " + str(gamma))
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    plt.title(f'{model}')

    leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=2)  # Position legend beneath the plot
    leg.draw_frame(False)
    plt.savefig(f'{model}')