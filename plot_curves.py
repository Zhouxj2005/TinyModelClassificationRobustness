import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

def plot_training_log(log_file):
    if not os.path.exists(log_file):
        print(f"错误: 找不到文件 {log_file}")
        return

    # 1. 读取数据
    df = pd.read_csv(log_file)
    epochs = df['epoch']
    loss = df['loss']
    acc = df['clean_acc']

    # 2. 设置画风 (类似 Seaborn/论文风格)
    plt.style.use('bmh') 
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 3. 绘制 Loss (左轴，红色)
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(epochs, loss, color=color, linewidth=2, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # 4. 绘制 Accuracy (右轴，蓝色)
    ax2 = ax1.twinx()  # 共享 x 轴
    color = 'tab:blue'
    ax2.set_ylabel('Clean Accuracy (%)', color=color)
    ax2.plot(epochs, acc, color=color, linewidth=2, linestyle='--', label='Validation Acc')
    ax2.tick_params(axis='y', labelcolor=color)

    # 5. 添加标题和图例
    plt.title('Training Progress (MobileNetV2 + AugMix)', fontsize=14)
    fig.tight_layout()  # 防止标签重叠

    # 6. 保存或显示
    save_name = log_file.replace('.csv', '.png')
    plt.savefig(save_name, dpi=300)
    print(f"✅ 曲线图已保存至: {save_name}")
    # plt.show() # 如果要在服务器上跑，注释掉这行

if __name__ == "__main__":
    # 使用示例：直接指定你的 csv 路径
    log_path = './checkpoints/mobilenet_v2_augmix_log.csv'
    plot_training_log(log_path)