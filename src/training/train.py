import numpy as np
from training.activations import softmax
from training.loss import lossCrossEntropy
from training.plots import plot_layer_stats, plot_learning_curves


def train(network, X_train, y_train, X_valid, y_valid, args):

    learning_rate = float(args.learning_rate)
    epochs = int(args.epochs) 
    batch_size = int(args.batch) 
    loss_fn = lossCrossEntropy

    chart_file = f"{args.activation}_b{batch_size}_e{epochs}_lr{learning_rate}_w{args.w_init}_layers{'-'.join(args.layers.split())}"

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):

        num_batches = 0

        # SHUFFLE
        indices = np.random.permutation(len(X_train))

        X_train = X_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0
        epoch_accuracy = 0

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
            network.backward(gradient, learning_rate)

            # ACCURACY
            pred_classes = np.argmax(predictions, axis=1)
            # print (f"batch {num_batches+1}: pred_classes={pred_classes}")
            true_classes = np.argmax(y_batch, axis=1)
            # print (f"batch {num_batches+1}: true_classes={true_classes}")
            accuracy = np.mean(pred_classes == true_classes)

            epoch_loss += train_loss
            epoch_accuracy += accuracy

            num_batches += 1

        # METRICS

        epoch_loss /= num_batches
        epoch_accuracy /= num_batches

        # VALIDATION

        val_logits = network.forward(X_valid)
        val_predictions = softmax(val_logits)
        val_loss = loss_fn(y_valid, val_predictions)
        val_pred_classes = np.argmax(val_predictions, axis=1)
        val_true_classes = np.argmax(y_valid, axis=1)

        val_accuracy = np.mean(val_pred_classes == val_true_classes)

        # CHARTS
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        train_accuracies.append(epoch_accuracy)
        val_accuracies.append(val_accuracy)

        # PRINT

        if args.verbose:
            print(
                f"epoch {epoch+1}/{epochs} "
                f"- loss: {epoch_loss:.4f} "
                f"- accuracy: {epoch_accuracy:.4f} "
                f"- val_loss: {val_loss:.4f} "
                f"- val_accuracy: {val_accuracy:.4f}"
            )

    # CHARTS
    plot_learning_curves(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies, 
        chart_file
    )

    stats = network.layer_stats(X_valid)
    plot_layer_stats(stats, chart_file)

    if args.verbose:
        for i, (s, m, a) in enumerate(zip(stats["stds"], stats["means"], stats["alive"])):
            print(f"layer {i+1}: std={s:.4f}, mean={m:.4f}, alive={a:.4f}")

    print (f"\nTraining complete.")
    # print (f"num_batches: {num_batches}")
    print (f"Final training loss: {train_losses[-1]:.4f}")
    print (f"Final training accuracy: {train_accuracies[-1]:.4f}")
    print (f"Final validation loss: {val_losses[-1]:.4f}")
    print (f"Final validation accuracy: {val_accuracy:.4f}")
    print (f"Saved learning curves to {chart_file}_loss.png and {chart_file}_accuracy.png")
    print (f"Saved activation stats plot to {chart_file}_layer_stats.png")