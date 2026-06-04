import numpy as np


class MetricSeries:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_precisions = []
        self.train_recalls = []
        self.train_f1s = []
        self.train_mses = []
        self.train_rmses = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []
        self.val_mses = []
        self.val_rmses = []

    def add_train(self, loss, accuracy, precision, recall, f1, mse, rmse):
        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)
        self.train_precisions.append(precision)
        self.train_recalls.append(recall)
        self.train_f1s.append(f1)
        self.train_mses.append(mse)
        self.train_rmses.append(rmse)

    def add_val(self, loss, accuracy, precision, recall, f1, mse, rmse):
        self.val_losses.append(loss)
        self.val_accuracies.append(accuracy)
        self.val_precisions.append(precision)
        self.val_recalls.append(recall)
        self.val_f1s.append(f1)
        self.val_mses.append(mse)
        self.val_rmses.append(rmse)

    def to_history(self):
        return {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "train_accuracy": self.train_accuracies,
            "val_accuracy": self.val_accuracies,
            "train_precision": self.train_precisions,
            "train_recall": self.train_recalls,
            "train_f1": self.train_f1s,
            "train_mse": self.train_mses,
            "train_rmse": self.train_rmses,
            "val_precision": self.val_precisions,
            "val_recall": self.val_recalls,
            "val_f1": self.val_f1s,
            "val_mse": self.val_mses,
            "val_rmse": self.val_rmses,
        }

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


def compute_regression_metrics(y_true_onehot, predictions):
    mse = float(np.mean((predictions - y_true_onehot) ** 2))
    rmse = float(np.sqrt(mse))
    return mse, rmse

def lossCrossEntropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
