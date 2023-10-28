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
    print(cum_eigenvalues)
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