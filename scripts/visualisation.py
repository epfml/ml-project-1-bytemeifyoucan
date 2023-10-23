#plotting confusion matrices
#plotting pca results
#plotting data with color code true false etc
import matplotlib.pyplot as plt
import numpy as np


def cross_validation_visualization(lambds, rmse_tr, rmse_te, visualisation):
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
    plt.savefig("cross_validation")
    
def plot_pca(n_components, eigenvalues, visualisation):
    plt.plot(range(n_components, eigenvalues/np.sum(eigenvalues) * 100, color = 'blue', marker = 'x'))
    plt.xlabel("Variance explained (%)")
    plt.ylabel("Principal Component")
    plt.title('Principal Component Analysis')
    plt.grid(True)
    if visualisation == False:
        plt.close()
    plt.savefig('PCA')