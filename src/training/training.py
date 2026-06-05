import json
import os
import numpy as np
from training.activations import softmax
from training.plots import plot_figure, plot_layer_stats
from training.metrics import MetricSeries, compute_classification_metrics, compute_regression_metrics, lossCrossEntropy


def snapshot_layer_params(network):
    return [(layer.W.copy(), layer.b.copy()) for layer in network.layers]


def restore_layer_params(network, snapshot):
    for layer, (weights, bias) in zip(network.layers, snapshot):
        layer.W = weights
        layer.b = bias


def train(network, X_train, y_train, X_valid, y_valid, args):

    learning_rate = float(args.learning_rate)
    momentum = float(getattr(args, "momentum", 0.9))
    optimizer_name = getattr(args, "optimizer", "sgd")
    epochs = int(args.epochs) 
    batch_size = int(args.batch) 
    patience = int(getattr(args, "patience", 0))
    min_delta = float(getattr(args, "min_delta", 0.0))

    chart_file = f"{args.activation}_b{batch_size}_e{epochs}_lr{learning_rate}_w{args.w_init}_layers{'-'.join(args.layers.split())}"

    metrics = MetricSeries()

    best_val_loss = float("inf")
    best_epoch = -1
    best_epoch_state = snapshot_layer_params(network)
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(epochs):
        num_batches = 0

        # SHUFFLE
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0

        for start in range(0, len(X_train), batch_size):
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            logits = network.forward(X_batch)
            # print (f"batch {num_batches+1}: logits={logits}")
            predictions = softmax(logits)
            # print (f"batch {num_batches+1}: predictions={predictions}")
            train_loss = lossCrossEntropy(y_batch, predictions)

            # BACK
            gradient = predictions - y_batch
            network.backward(gradient)

            epoch_loss += train_loss
            num_batches += 1

        epoch_loss /= num_batches

        # VALIDATION
        val_logits = network.forward(X_valid)
        val_predictions = softmax(val_logits)
        val_loss = lossCrossEntropy(y_valid, val_predictions)

        # METRICS
        train_logits_full = network.forward(X_train)
        train_predictions_full = softmax(train_logits_full)

        train_mse, train_rmse = compute_regression_metrics(y_train, train_predictions_full)
        train_accuracy, train_precision, train_recall, train_f1 = compute_classification_metrics(y_train, train_predictions_full)
        metrics.add_train(epoch_loss, train_accuracy, train_precision, train_recall, train_f1, train_mse, train_rmse)
        
        val_mse, val_rmse = compute_regression_metrics(y_valid, val_predictions)
        val_accuracy, val_precision, val_recall, val_f1 = compute_classification_metrics(y_valid, val_predictions)
        metrics.add_val(val_loss, val_accuracy, val_precision, val_recall, val_f1, val_mse, val_rmse)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_epoch_state = snapshot_layer_params(network)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if args.verbose:
            print(
                f"epoch {epoch+1}/{epochs} "
                f"- loss: {epoch_loss:.4f} "
                f"- accuracy: {train_accuracy:.4f} "
                f"- val_loss: {val_loss:.4f} "
                f"- val_accuracy: {val_accuracy:.4f} "
                f"- val_f1: {val_f1:.4f}"
            )

        if patience > 0 and epochs_without_improvement >= patience:
            stopped_early = True
            print(f"early stopping at epoch {epoch+1} (best val_loss: {best_val_loss:.4f} at epoch {best_epoch+1})")
            break

    if stopped_early:
        restore_layer_params(network, best_epoch_state)
        val_logits = network.forward(X_valid)  
        val_predictions = softmax(val_logits)
        val_loss = lossCrossEntropy(y_valid, val_predictions)
        val_mse, val_rmse = compute_regression_metrics(y_valid, val_predictions)
        val_accuracy, val_precision, val_recall, val_f1 = compute_classification_metrics(y_valid, val_predictions)

    # CHARTS
    plot_figure("Loss", chart_file+ "_loss.png", metrics.train_losses, "Train Loss", metrics.val_losses, "Val Loss")
    plot_figure("Accuracy", chart_file+ "_accuracy.png", metrics.train_accuracies, "Train Accuracy", metrics.val_accuracies, "Val Accuracy")
    plot_figure("F1-Score", chart_file+ "_f1.png", metrics.val_precisions, "Val Precision", metrics.val_recalls, "Val Recall", metrics.train_f1s, "Train F1-Score")
    # plot_layer_stats(network.layer_stats(X_valid), chart_file)
    plot_figure("Loss", "last_loss.png", metrics.train_losses, "Train Loss", metrics.val_losses, "Val Loss")
    plot_figure("Accuracy", "last_accuracy.png", metrics.train_accuracies, "Train Accuracy", metrics.val_accuracies, "Val Accuracy")

    history = {
        "model_name": chart_file,
        "epochs": epochs,
        "stopped_epoch": len(metrics.train_losses),
        "best_epoch": best_epoch + 1 if best_epoch >= 0 else None,
        "best_val_loss": best_val_loss if best_epoch >= 0 else None,
        "early_stopping": {
            "enabled": patience > 0,
            "patience": patience,
            "min_delta": min_delta,
            "stopped": stopped_early,
        },
        "config": {
            "activation": args.activation,
            "optimizer": optimizer_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "w_init": args.w_init,
            "layers": args.layers,
        },
    }
    history.update(metrics.to_history())

    os.makedirs("result/hist", exist_ok=True)
    history_path = f"result/hist/{chart_file}.json"
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file)

    print (f"\nTraining complete.")
    print (f"\nFinal training loss: {metrics.train_losses[-1]:.4f}")
    print (f"Final training accuracy: {metrics.train_accuracies[-1]:.4f}")
    print (f"Final training MSE/RMSE: {train_mse:.6f}/{train_rmse:.6f}")
    print (f"Final training precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-score: {train_f1:.4f}")
    
    print (f"\nFinal validation loss: {metrics.val_losses[-1]:.4f}")
    print (f"Final validation accuracy: {val_accuracy:.4f}")
    print (f"Final validation MSE/RMSE: {val_mse:.6f}/{val_rmse:.6f}")
    print (f"Final validation precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1:.4f}\n")
    
    # print (f"\nSaved learning curves to {chart_file}_loss.png and {chart_file}_accuracy.png")
    # print (f"Saved classification metrics plot to {chart_file}_classification_metrics.png")
    # print (f"Saved activation stats plot to {chart_file}_layer_stats.png")
    print (f"Saved history to {history_path}")