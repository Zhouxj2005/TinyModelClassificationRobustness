import os
import argparse
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd

# Data extracted from result.md
CORRUPTIONS: List[str] = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]

RESULTS: Dict[str, Dict] = {
    "mobilenet_v2_standard": {
        "clean": 69.35,
        "robust": 30.79,
        "corruptions": {
            "gaussian_noise": 4.30,
            "shot_noise": 4.05,
            "impulse_noise": 6.62,
            "defocus_blur": 5.07,
            "glass_blur": 8.11,
            "motion_blur": 13.25,
            "zoom_blur": 41.84,
            "snow": 35.13,
            "frost": 34.27,
            "fog": 54.39,
            "brightness": 64.01,
            "contrast": 41.68,
            "elastic_transform": 53.22,
            "pixelate": 48.32,
            "jpeg_compression": 47.59,
        },
    },
    "mobilenet_v2_augmix": {
        "clean": 70.60,
        "robust": 41.41,
        "corruptions": {
            "gaussian_noise": 15.71,
            "shot_noise": 15.77,
            "impulse_noise": 29.12,
            "defocus_blur": 11.69,
            "glass_blur": 21.17,
            "motion_blur": 24.41,
            "zoom_blur": 61.87,
            "snow": 41.73,
            "frost": 39.30,
            "fog": 57.03,
            "brightness": 66.70,
            "contrast": 54.62,
            "elastic_transform": 62.59,
            "pixelate": 60.28,
            "jpeg_compression": 59.21,
        },
    },
    "mobilenet_v2_distill": {
        "clean": 74.20,
        "robust": 43.18,
        "corruptions": {
            "gaussian_noise": 19.03,
            "shot_noise": 18.09,
            "impulse_noise": 29.47,
            "defocus_blur": 9.33,
            "glass_blur": 18.07,
            "motion_blur": 23.10,
            "zoom_blur": 64.14,
            "snow": 45.32,
            "frost": 45.03,
            "fog": 62.10,
            "brightness": 70.95,
            "contrast": 55.99,
            "elastic_transform": 65.02,
            "pixelate": 60.87,
            "jpeg_compression": 61.15,
        },
    },
    "mobilenet_v2_distill_improved": {
        "clean": 68.56,
        "robust": 39.50,
        "corruptions": {
            "gaussian_noise": 13.79,
            "shot_noise": 13.94,
            "impulse_noise": 24.23,
            "defocus_blur": 11.06,
            "glass_blur": 20.73,
            "motion_blur": 22.89,
            "zoom_blur": 59.46,
            "snow": 40.02,
            "frost": 37.34,
            "fog": 55.33,
            "brightness": 64.14,
            "contrast": 51.73,
            "elastic_transform": 60.72,
            "pixelate": 59.11,
            "jpeg_compression": 57.95,
        },
    },
    "resnet18_augmix": {
        "clean": 76.30,
        "robust": 46.68,
        "corruptions": {
            "gaussian_noise": 17.25,
            "shot_noise": 16.76,
            "impulse_noise": 33.00,
            "defocus_blur": 12.23,
            "glass_blur": 21.69,
            "motion_blur": 30.20,
            "zoom_blur": 70.45,
            "snow": 47.49,
            "frost": 48.93,
            "fog": 66.11,
            "brightness": 73.82,
            "contrast": 66.18,
            "elastic_transform": 68.86,
            "pixelate": 64.14,
            "jpeg_compression": 63.05,
        },
    },
    "resnet18_prime": {
        "clean": 77.60,
        "robust": 53.20,
        "corruptions": {
            "gaussian_noise": 51.43,
            "shot_noise": 48.44,
            "impulse_noise": 35.76,
            "defocus_blur": 13.72,
            "glass_blur": 23.90,
            "motion_blur": 29.98,
            "zoom_blur": 69.54,
            "snow": 51.05,
            "frost": 58.19,
            "fog": 67.43,
            "brightness": 75.15,
            "contrast": 71.21,
            "elastic_transform": 69.26,
            "pixelate": 66.78,
            "jpeg_compression": 66.17,
        },
    },
}

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def save_table_image(df: pd.DataFrame, title: str, out_path: str, col_width: float = 1.8, row_height: float = 0.5) -> str:
    """Render a DataFrame as a matplotlib table and save as PNG."""
    fig, ax = plt.subplots(figsize=(col_width * len(df.columns), row_height * (len(df) + 1)))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.1)
    ax.set_title(title, pad=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_overall(models: List[str], out_dir: str) -> str:
    clean = [RESULTS[m]["clean"] for m in models]
    robust = [RESULTS[m]["robust"] for m in models]
    x = range(len(models))

    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], clean, width, label="Clean", color=COLORS[0])
    ax.bar([i + width / 2 for i in x], robust, width, label="Robust", color=COLORS[1])
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Clean vs Robust Accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "overall_clean_vs_robust.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def table_overall(models: List[str], out_dir: str) -> Dict[str, str]:
    data = {
        "model": [],
        "clean": [],
        "robust": [],
    }
    for m in models:
        data["model"].append(m)
        data["clean"].append(RESULTS[m]["clean"])
        data["robust"].append(RESULTS[m]["robust"])

    df = pd.DataFrame(data).set_index("model")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "overall_table.csv")
    df.to_csv(csv_path)

    img_path = os.path.join(out_dir, "overall_table.png")
    save_table_image(df.round(2), "Clean / Robust Accuracy", img_path)
    return {"csv": csv_path, "png": img_path}


def plot_corruptions(models: List[str], out_dir: str) -> str:
    x = range(len(CORRUPTIONS))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(14, 6))

    for idx, model in enumerate(models):
        offsets = [i + (idx - (len(models) - 1) / 2) * width for i in x]
        vals = [RESULTS[model]["corruptions"][c] for c in CORRUPTIONS]
        ax.bar(offsets, vals, width, label=model, color=COLORS[idx % len(COLORS)])

    ax.set_xticks(list(x))
    ax.set_xticklabels(CORRUPTIONS, rotation=35, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Corruption Accuracy Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "per_corruption_comparison.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def table_corruptions(models: List[str], out_dir: str) -> Dict[str, str]:
    # rows: corruption types; columns: models
    rows = []
    for c in CORRUPTIONS:
        row = {"corruption": c}
        for m in models:
            row[m] = RESULTS[m]["corruptions"][c]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("corruption")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "corruption_table.csv")
    df.to_csv(csv_path)

    img_path = os.path.join(out_dir, "corruption_table.png")
    save_table_image(df.round(2), "Per-Corruption Accuracy", img_path, col_width=1.4, row_height=0.45)
    return {"csv": csv_path, "png": img_path}


def main():
    parser = argparse.ArgumentParser(description="Plot CIFAR-100-C results from result.md data")
    parser.add_argument("--models", nargs="+", default=list(RESULTS.keys()), help="Models to include in plots")
    parser.add_argument("--out", type=str, default="./outputs", help="Output directory for figures")
    args = parser.parse_args()

    for m in args.models:
        if m not in RESULTS:
            raise ValueError(f"未知模型: {m}")

    overall_path = plot_overall(args.models, args.out)
    corr_path = plot_corruptions(args.models, args.out)
    overall_tables = table_overall(args.models, args.out)
    corr_tables = table_corruptions(args.models, args.out)

    print(f"Saved overall figure to: {overall_path}")
    print(f"Saved corruption figure to: {corr_path}")
    print(f"Saved overall tables: CSV={overall_tables['csv']}, PNG={overall_tables['png']}")
    print(f"Saved corruption tables: CSV={corr_tables['csv']}, PNG={corr_tables['png']}")


if __name__ == "__main__":
    main()
