import matplotlib.pyplot as plt
import numpy as np
from definitions import ROOT_DIR
import os


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

def plot_train_test(model, train_errors, test_errors, lambdas, gammas=0, visualisation = False):
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

    if visualisation == False:
        plt.close()
    fig_path = os.path.join(ROOT_DIR, 'figures')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(os.path.join(fig_path, f'errors_{model}'))

def plot_metrics(model, metrics, thresholds, visualisation = False):
   
    if(len(thresholds) > 7):
        raise ValueError("You can plot up to 7 differents thresholds on the same graph, you tried to plot " + str(len(thresholds)))
    
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']

    # Plotting each metric as a bar plot
    for i, metric in enumerate(metrics):
        positions = np.arange(len(metric)) + i * 0.2
        plt.bar(positions, metric, width=0.2, color=colors[i], label=f'Threshold {thresholds[i]}')

    custom_labels = ['accuracy', 'f1 score', 'specificity', 'sensitivity', 'precision']
    # Replace x-axis numeric indices with custom labels
    plt.xticks(np.arange(len(custom_labels)) + 0.2, custom_labels)

    plt.title(f'Metrics obtained with {model}')
    leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=len(thresholds))  # Position legend beneath the plot
    leg.draw_frame(False)

    if visualisation == False:
        plt.close()
    fig_path = os.path.join(ROOT_DIR, 'figures')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(os.path.join(fig_path, f'metrics_{model}.png'))