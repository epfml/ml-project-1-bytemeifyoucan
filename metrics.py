import numpy as np

def confusion_matrix(y_true, y_pred):
    """Calculates the confusion matrix of the model

    Args:
        y_true (np.ndarray): shape = (N,) contains the data we are provided with
        y_pred (np.ndarray): shape = (N,) contains the data we predicted

    Returns:
        np.ndarray : shape = (2,2) confusion matrix
    """

    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    num_classes = len(unique_labels)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_label = y_pred[i]

        true_index = np.where(unique_labels == true_label)[0][0]
        pred_index = np.where(unique_labels == pred_label)[0][0]

        conf_matrix[true_index, pred_index] += 1

    return conf_matrix


def calculate_f1_score(y_true, y_pred):
    """Calculates the f1 score of the model
       It is the harmonic mean of precision and recall, providing a balance between the two metrics.

    Args:
        y_true (np.ndarray): shape = (N,) contains the data we are provided with
        y_pred (np.ndarray): shape = (N,) contains the data we predicted

    Returns:
        float : f1 score
    """
    
    # Binary classification: assume 1 positive class and 0 negative class
    precision = calcuate_precision(y_true, y_pred)
    recall = caclculate_sensitivity(y_true, y_pred)
        
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score


def calcuate_precision(y_true, y_pred):
    """Calculates the precision of the model
       Precision measures the proportion of true positive predictions out of all positive predictions.

    Args:
        y_true (np.ndarray): shape = (N,) contains the data we are provided with
        y_pred (np.ndarray): shape = (N,) contains the data we predicted

    Returns:
        float : precision
    """

    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays have different lengths")

    conf_matrix = confusion_matrix(y_true, y_pred)
    TP = conf_matrix[1, 1]
    FN = conf_matrix[1, 0]

    precision = TP / (TP + FP)

    return precision


def calculate_specificity(y_true, y_pred):
    """Calculates the specificity of the model
       True Negative Rate/Specificity measures the proportion of true negative predictions out of all actual negative instances.

    Args:
        y_true (np.ndarray): shape = (N,) contains the data we are provided with
        y_pred (np.ndarray): shape = (N,) contains the data we predicted

    Returns:
        float : specificity
    """

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Extract True Negatives (TN) and False Positives (FP) from the confusion matrix
    TN = conf_matrix[0, 0]  # Row 0 (actual negatives), Column 0 (predicted negatives)
    FP = conf_matrix[0, 1]  # Row 0 (actual negatives), Column 1 (predicted positives)

    # Calculate Specificity (TNR)
    specificity = TN / (TN + FP)

    return specificity
    

def calculate_sensitivity(y_true, y_pred):
    """Calculates the sensitivity/recall of the model
       Recall measures the proportion of true positive predictions out of all actual positive instances. 

    Args:
        y_true (np.ndarray): shape = (N,) contains the data we are provided with
        y_pred (np.ndarray): shape = (N,) contains the data we predicted

    Returns:
        float : sensitivity
    """

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Extract True Positives (TP) and False Negatives (FN) from the confusion matrix
    TP = conf_matrix[1, 1]  # Row 1 (actual positives), Column 1 (predicted positives)
    FN = conf_matrix[1, 0]  # Row 1 (actual positives), Column 0 (predicted negatives)

    # Calculate Sensitivity (Recall)
    sensitivity = TP / (TP + FN)

    return sensitivity
 

def calculate_accuracy(y_true, y_pred):
    """Calculates the accuracy of the model

    Args:
        y_true (np.ndarray): shape = (N,) contains the data we are provided with
        y_pred (np.ndarray): shape = (N,) contains the data we predicted

    Returns:
        float : accuracy
    """

    # Check if the lengths of the true and predicted labels match
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays have different lengths")

    # Calculate the number of correctly predicted instances
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1

    # Calculate accuracy as the proportion of correct predictions
    accuracy = correct / len(y_true)

    return accuracy

def calculate_metrics(y_true, y_pred):
    """Calculates a list of metrics for a given set of predictions and groundtruth in the following order
        accuracy, f1 score, specificity, sensitivity, precision

    Args:
        y_true (np.ndarray): shape = (N,) contains the data we are provided with
        y_pred (np.ndarray): shape = (N,) contains the data we predicted

    Returns:
        list (float): list of metrics
    """

    metrics = [calculate_accuracy(y_true, y_pred), 
               calculate_f1_score(y_true, y_pred),
               calculate_specificity(y_true, y_pred),
               calculate_sensitivity(y_true, y_pred),
               calcuate_precision(y_true, y_pred)]
    return metrics

def prettyprint(metrics):
    print(f'Accuracy: {metrics[0]} - F1 score {metrics[1]}')
    print(f'Specificity: {metrics[2]} - Sensitivity: {metrics[3]}')
    print(f'Precision: {metrics[4]}')