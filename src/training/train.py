import json
import os
import numpy as np
from training.activations import softmax
from training.loss import lossCrossEntropy
from training.plots import plot_classification_metrics, plot_layer_stats, plot_learning_curves
from training.metrics import compute_classification_metrics 


def snapshot_layer_params(network):
    return [(layer.W.copy(), layer.b.copy()) for layer in network.layers]


def restore_layer_params(network, snapshot):
    for layer, (weights, bias) in zip(network.layers, snapshot):
        layer.W = weights
        layer.b = bias


def train(network, X_train, y_train, X_valid, y_valid, args):

    learning_rate = float(args.learning_rate)
    momentum = float(getattr(args, "momentum", 0.9))
    optimizer_name = getattr(args, "optimizer", "nesterov")
    epochs = int(args.epochs) 
    batch_size = int(args.batch) 
    patience = int(getattr(args, "patience", 0))
    min_delta = float(getattr(args, "min_delta", 0.0))
    loss_fn = lossCrossEntropy

    chart_file = f"{args.activation}_b{batch_size}_e{epochs}_lr{learning_rate}_w{args.w_init}_layers{'-'.join(args.layers.split())}"

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_precisions = []
    train_recalls = []
    train_f1s = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

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
            train_loss = loss_fn(y_batch, predictions)

            # BACK
            gradient = predictions - y_batch
            network.backward(gradient)

            epoch_loss += train_loss
            num_batches += 1

        epoch_loss /= num_batches

        # VALIDATION
        val_logits = network.forward(X_valid)
        val_predictions = softmax(val_logits)
        val_loss = loss_fn(y_valid, val_predictions)

        # METRICS
        train_logits_full = network.forward(X_train)
        train_predictions_full = softmax(train_logits_full)

        train_accuracy, train_precision, train_recall, train_f1 = compute_classification_metrics(y_train, train_predictions_full)
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)

        val_accuracy, val_precision, val_recall, val_f1 = compute_classification_metrics(y_valid, val_predictions)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

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

    # CHARTS
    plot_learning_curves( train_losses, val_losses, train_accuracies, val_accuracies, chart_file)
    plot_classification_metrics(val_precisions, val_recalls, val_f1s, chart_file)
    # plot_layer_stats(network.layer_stats(X_valid), chart_file)

    history = {
        "model_name": chart_file,
        "epochs": epochs,
        "stopped_epoch": len(train_losses),
        "best_epoch": best_epoch + 1 if best_epoch >= 0 else None,
        "best_val_loss": best_val_loss if best_epoch >= 0 else None,
        "early_stopping": {
            "enabled": patience > 0,
            "patience": patience,
            "min_delta": min_delta,
            "stopped": stopped_early,
        },
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_accuracy": train_accuracies,
        "val_accuracy": val_accuracies,
        "train_precision": train_precisions,
        "train_recall": train_recalls,
        "train_f1": train_f1s,
        "val_precision": val_precisions,
        "val_recall": val_recalls,
        "val_f1": val_f1s,
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

    os.makedirs("result/hist", exist_ok=True)
    history_path = f"result/hist/{chart_file}.json"
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file)

    print (f"\nTraining complete.")
    print (f"Final training loss: {train_losses[-1]:.4f}")
    print (f"Final training accuracy: {train_accuracies[-1]:.4f}")
    print (f"Final validation loss: {val_losses[-1]:.4f}")
    print (f"Final validation accuracy: {val_accuracy:.4f}")
    print (f"Saved learning curves to {chart_file}_loss.png and {chart_file}_accuracy.png")
    print (f"Saved classification metrics plot to {chart_file}_classification_metrics.png")
    print (f"Saved activation stats plot to {chart_file}_layer_stats.png")
    print (f"Saved history to {history_path}")