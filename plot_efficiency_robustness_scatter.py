import os
import math
import torch
import matplotlib.pyplot as plt

# 本脚本根据 result.md 中的 mAcc，计算模型参数量并绘制效率-鲁棒性散点图。
# 轴：X=Parameters (M)，Y=CIFAR-100-C Mean Robust Accuracy (%)


def count_params_million(model: torch.nn.Module) -> float:
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6


def get_arch_params_million(arch: str) -> float:
    # 延迟导入，避免不必要的依赖初始化
    from models import get_model
    model = get_model(arch, num_classes=100)
    model.eval()
    with torch.no_grad():
        params_m = count_params_million(model)
    return params_m


def main():
    # 从 result.md 提取的鲁棒性均值（mAcc）——四个关键点
    points = [
        {"name": "MobileNetV2-Standard", "arch": "mobilenet_v2", "macc": 30.79},
        {"name": "MobileNetV2-AugMix",   "arch": "mobilenet_v2", "macc": 41.41},
        {"name": "MobileNetV2-Ours",     "arch": "mobilenet_v2", "macc": 43.18},
        {"name": "ResNet18-AugMix",      "arch": "resnet18",     "macc": 46.68},
    ]

    # 计算各架构的参数量（百万）
    arch_to_params = {}
    for p in points:
        arch = p["arch"]
        if arch not in arch_to_params:
            arch_to_params[arch] = get_arch_params_million(arch)
        p["params_m"] = arch_to_params[arch]

    # 颜色与样式：突出 Ours
    style = {
        "MobileNetV2-Standard": {"color": "#7f8c8d", "marker": "o", "size": 80},
        "MobileNetV2-AugMix":   {"color": "#2980b9", "marker": "o", "size": 100},
        "MobileNetV2-Ours":     {"color": "#ff0000", "marker": "*", "size": 220, "edgecolor": "black", "linewidth": 1.2},
        "ResNet18-AugMix":      {"color": "#e67e22", "marker": "s", "size": 120},
    }

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=140)

    # 绘制散点
    for p in points:
        st = style[p["name"]]
        ax.scatter(
            [p["params_m"]], [p["macc"]],
            s=st.get("size", 100),
            c=st["color"],
            marker=st.get("marker", "o"),
            edgecolors=st.get("edgecolor", None),
            linewidths=st.get("linewidth", None),
            label=p["name"] if p["name"] != "MobileNetV2-Ours" else None,  # Ours 单独标注
        )

    # 单独给 Ours 添加图例条目（以更醒目的样式）
    ours = next(p for p in points if p["name"] == "MobileNetV2-Ours")
    ax.scatter([ours["params_m"]], [ours["macc"]], s=style["MobileNetV2-Ours"]["size"],
               c=style["MobileNetV2-Ours"]["color"], marker=style["MobileNetV2-Ours"]["marker"],
               edgecolors=style["MobileNetV2-Ours"].get("edgecolor", None), linewidths=style["MobileNetV2-Ours"].get("linewidth", None),
               label="MobileNetV2-Ours")

    # # 为三个 MobileNetV2 的点加标签，避免重叠
    # for p in points:
    #     if "MobileNetV2" in p["name"]:
    #         ax.annotate(p["name"], (p["params_m"], p["macc"]),
    #                     textcoords="offset points", xytext=(6, -10), ha="left", fontsize=10, color="#2c3e50")

    # # 添加箭头：从 Standard/AugMix 指向 Ours（体现“向左上角移动”——此处为同一 x、更高 y 的上移）
    # std = next(p for p in points if p["name"] == "MobileNetV2-Standard")
    # aug = next(p for p in points if p["name"] == "MobileNetV2-AugMix")
    # ax.annotate("", xy=(ours["params_m"], ours["macc"]), xytext=(std["params_m"], std["macc"]),
    #             arrowprops=dict(arrowstyle="->", color="#34495e", lw=1.5))
    # ax.annotate("", xy=(ours["params_m"], ours["macc"]), xytext=(aug["params_m"], aug["macc"]),
    #             arrowprops=dict(arrowstyle="->", color="#34495e", lw=1.5))


    # 轴与网格设置
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("CIFAR-100-C Mean Robust Accuracy (%)")

    # X 轴范围：留些边距，突出 MobileNet 与 ResNet 的差异
    x_vals = [p["params_m"] for p in points]
    y_vals = [p["macc"] for p in points]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    ax.set_xlim(max(0, x_min - 0.5), x_max + 0.8)
    ax.set_ylim(max(0, y_min - 5), min(100, y_max + 5))

    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(loc="lower right", frameon=False)

    # 输出文件
    out_dir = os.path.join("outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "efficiency_robustness_scatter.png")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
