import matplotlib.pyplot as plt


def plot_learning_curves(train_losses, val_losses,
                          train_accs, val_accs, chart_file):

    epochs = range(1, len(train_losses) + 1)

    # =========================
    # LOSS
    # =========================
    plt.figure()
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Val")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig("charts/" + chart_file+ "_loss.png")
    plt.close()    

    # =========================
    # ACCURACY
    # =========================
    plt.figure()
    plt.plot(epochs, train_accs, label="Train")
    plt.plot(epochs, val_accs, label="Val")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig("charts/" + chart_file+ "_accuracy.png")
    plt.close()