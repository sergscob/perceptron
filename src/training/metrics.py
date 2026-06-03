import numpy as np

def compute_classification_metrics(y_true_onehot, predictions):
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(predictions, axis=1)

    eps = 1e-15
    accuracy = float(np.mean(y_pred == y_true))

    pos = 1
    tp = int(np.sum((y_pred == pos) & (y_true == pos)))
    fp = int(np.sum((y_pred == pos) & (y_true != pos)))
    fn = int(np.sum((y_pred != pos) & (y_true == pos)))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return accuracy, precision, recall, f1
