import matplotlib.pyplot as plt


def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, chart_file):
    epochs = range(1, len(train_losses) + 1)

    # LOSS
    plt.figure()
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Val")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig("result/charts/" + chart_file+ "_loss.png")
    plt.close()    

    # ACCURACY
    plt.figure()
    plt.plot(epochs, train_accs, label="Train")
    plt.plot(epochs, val_accs, label="Val")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig("result/charts/" + chart_file+ "_accuracy.png")
    plt.close()


def plot_layer_stats(stats, chart_file):
    layers = range(1, len(stats["stds"]) + 1)

    plt.figure()
    plt.plot(layers, stats["stds"], marker="o", label="x.std()")
    plt.plot(layers, stats["means"], marker="o", label="x.mean()")
    plt.plot(layers, stats["alive"], marker="o", label="(z > 0).mean()")
    plt.title("Stats by layer")
    plt.xlabel("Layer")
    plt.ylabel("Value")
    plt.xticks(list(layers))
    plt.legend()
    plt.grid(True)

    plt.savefig("result/charts/" + chart_file + "_layer_stats.png")
    plt.close()