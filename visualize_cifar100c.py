import argparse
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets


CORRUPTION_TYPES: List[str] = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]


def load_corruption_image(root: str, corruption_type: str, base_index: int, severity: int) -> np.ndarray:
    """
    Load one corrupted image for a given base test index and severity.

    CIFAR-100-C for each corruption has shape (50000, 32, 32, 3), arranged as
    5 severity blocks of 10000 images each. The mapping is:
        idx_in_corruption = base_index + (severity - 1) * 10000
    """
    assert 1 <= severity <= 5, "severity 必须在 1~5 之间"
    file_path = os.path.join(root, f"{corruption_type}.npy")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到 {file_path}")

    arr = np.load(file_path)  # (50000, 32, 32, 3), uint8
    idx = base_index + (severity - 1) * 10000
    if idx >= arr.shape[0]:
        raise IndexError(f"索引超出范围: idx={idx}, arr.shape[0]={arr.shape[0]}")
    img = arr[idx]
    return img  # uint8, HWC


def visualize_grid(index: int = 0,
                   severity: int = 3,
                   c_root: str = './data/cifar100-c',
                   clean_root: str = './data/cifar100',
                   save_path: str = None,
                   show: bool = False) -> str:
    """
    Visualize one clean CIFAR-100 test image and its 15 corrupted variants at a given severity.

    Returns the path to the saved figure.
    """
    # Load clean test set sample (PIL Image, label)
    clean_ds = datasets.CIFAR100(root=clean_root, train=False, download=True)
    if not (0 <= index < len(clean_ds)):
        raise IndexError(f"index 必须在 0~{len(clean_ds)-1} 之间")

    clean_img, clean_label = clean_ds[index]
    clean_img_np = np.array(clean_img)
    class_name = clean_ds.classes[clean_label] if hasattr(clean_ds, 'classes') else str(clean_label)

    # Prepare grid: 4x4 (1 clean + 15 corruptions)
    nrows, ncols = 4, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 16))
    axes = axes.ravel()

    # First cell: clean
    axes[0].imshow(clean_img_np)
    axes[0].set_title(f"clean\n{class_name}")
    axes[0].axis('off')

    # Next 15 cells: each corruption (at the same severity)
    for i, ctype in enumerate(CORRUPTION_TYPES, start=1):
        try:
            img_np = load_corruption_image(c_root, ctype, index, severity)
            axes[i].imshow(img_np)
            axes[i].set_title(f"{ctype}\nsev={severity}", fontsize=10)
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error\n{e}", ha='center', va='center', fontsize=9)
        axes[i].axis('off')

    fig.suptitle(f"CIFAR-100-C visualization | index={index}, class={class_name} (label={clean_label})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Determine save path
    if save_path is None:
        os.makedirs('./outputs', exist_ok=True)
        save_path = os.path.join('./outputs', f'cifar100c_index{index}_sev{severity}.png')

    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    return save_path


def main():
    parser = argparse.ArgumentParser(description='Visualize CIFAR-100-C: one sample across all corruptions.')
    parser.add_argument('--index', type=int, default=0, help='CIFAR-100 test index (0-9999)')
    parser.add_argument('--severity', type=int, default=3, help='Corruption severity (1-5)')
    parser.add_argument('--c-root', type=str, default='./data/cifar100-c', help='Path to CIFAR-100-C folder')
    parser.add_argument('--clean-root', type=str, default='./data/cifar100', help='Path to CIFAR-100 folder')
    parser.add_argument('--save', type=str, default=None, help='Output path for the figure (PNG)')
    parser.add_argument('--show', action='store_true', help='Show the window after saving')
    args = parser.parse_args()

    out = visualize_grid(index=args.index,
                         severity=args.severity,
                         c_root=args.c_root,
                         clean_root=args.clean_root,
                         save_path=args.save,
                         show=args.show)
    print(f"Saved visualization to: {out}")


if __name__ == '__main__':
    main()
