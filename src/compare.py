import argparse
import glob
import json
import os
import sys

from training.plots import plot_model_comparison


METRIC_CHOICES = ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]


def load_hist(patterns):
    files = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            files.extend(matches)
        elif os.path.isfile(pattern):
            files.append(pattern)

    uniq_files = sorted(set(files))
    if not uniq_files:
        raise ValueError("No history files found.")

    hists = []
    for path in uniq_files:
        with open(path, "r", encoding="utf-8") as file:
            history = json.load(file)

        history["label"] = os.path.splitext(os.path.basename(path))[0]
        hists.append(history)

    return hists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hist", nargs="+", help="History JSON files or glob patterns (for example: result/charts/hist/*.json)")
    parser.add_argument("--metric", default="val_loss", choices=METRIC_CHOICES, help="Metric to compare")
    parser.add_argument("--output", default=None, help="Output PNG path (default: result/charts/compare/<metric>.png)")
    args = parser.parse_args()

    try:
        hists = load_hist(args.hist)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    output_path = args.output or f"result/charts/compare/{args.metric}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    missing_metric = [h.get("label", "model") for h in hists if args.metric not in h]
    if missing_metric:
        print("Error: metric not found in histories for: " + ", ".join(missing_metric))
        sys.exit(1)

    plot_model_comparison(hists, args.metric, output_path)
    print(f"Saved comparison chart to {output_path}")


if __name__ == "__main__":
    main()
