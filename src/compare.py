import argparse
import glob
import json
import os
import sys

from training.plots import plot_model_comparison


METRIC_CHOICES = [
    "train_loss",
    "val_loss",
    "train_accuracy",
    "val_accuracy",
    "train_precision",
    "train_recall",
    "train_f1",
    "val_precision",
    "val_recall",
    "val_f1",
]
DEFAULT_METRIC = "val_loss"
DEFAULT_HISTORY_GLOB = "result/hist/*.json"


def load_hist():
    files = []
    matches = glob.glob(DEFAULT_HISTORY_GLOB)
    if matches:
        files.extend(matches)

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
    parser.add_argument("-m", "--metric", default=DEFAULT_METRIC, choices=METRIC_CHOICES, help="Metric to compare")
    parser.add_argument("--output", default=None, help="Output PNG path (default: result/charts/compare/<metric>.png)")
    args = parser.parse_args()

    try:
        hists = load_hist()
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
