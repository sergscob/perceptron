import matplotlib.pyplot as plt



def plot_figure(title, chart_file, line1, label1, line2, label2, line3=None, label3=None):
    epochs = range(1, len(line1) + 1)

    plt.figure()
    plt.plot(epochs, line1, label=label1)
    plt.plot(epochs, line2, label=label2)
    if line3 is not None and label3 is not None:
        plt.plot(epochs, line3, label=label3)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlabel("Epoch")

    plt.savefig("result/charts/" + chart_file)
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


def plot_model_comparison(hists, metric, output_path, title=None):
    plt.figure()

    for history in hists:
        values = history[metric]
        epochs = range(1, len(values) + 1)
        label = history.get("label") or history.get("model_name", "model")
        plt.plot(epochs, values, label=label)

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(title or f"Model comparison: {metric}")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()