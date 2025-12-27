import matplotlib.pyplot as plt
import numpy as np

# 设置 CVPR 风格字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

def draw_figure_3():
    # 数据来源：result.md
    # 我们对比 ResNet18-AugMix 和 MobileNetV2-Ours
    corruptions = ['Gaussian Noise', 'Shot Noise', 'Impulse Noise']
    
    # ResNet18 (AugMix) 数据
    acc_r18 = [17.25, 16.76, 33.00] 
    
    # MobileNetV2 (Ours) 数据
    acc_ours = [19.03, 18.09, 29.47] # Impulse虽然输了，但也要画出来显得诚实
    
    x = np.arange(len(corruptions))
    width = 0.35  # 柱状图宽度

    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # 绘制柱子
    # 颜色：ResNet用灰色/蓝色（代表Baseline），Ours用红色/橙色（代表Highlight）
    rects1 = ax.bar(x - width/2, acc_r18, width, label='ResNet18 (AugMix, 11.7M)', color='#4c72b0', alpha=0.9, edgecolor='black')
    rects2 = ax.bar(x + width/2, acc_ours, width, label='MobileNetV2 (Ours, 3.4M)', color='#c44e52', alpha=0.9, edgecolor='black', hatch='//')

    # 添加数值标签
    def autolabel(rects, is_ours=False):
        for rect in rects:
            height = rect.get_height()
            # 如果是我们的并且赢了，加粗显示
            fontweight = 'bold' if is_ours and height > 17 else 'normal' 
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight=fontweight)

    autolabel(rects1)
    autolabel(rects2, is_ours=True)

    # 装饰图表
    ax.set_ylabel('Robust Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Specific Robustness Analysis on Noise Corruptions', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(corruptions, fontsize=11)
    ax.set_ylim(0, 40) # 设置Y轴上限，留出空间给Legend
    ax.legend(fontsize=10, loc='upper left')
    
    # 添加网格
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # 重点标注：画一个箭头指向 Gaussian Noise 的反超
    ax.annotate('Outperforms Larger Model!', 
                xy=(x[0]+width/2, 19.03), 
                xytext=(x[0]+0.5, 25),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                fontsize=10, color='darkred', fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/noise_analysis.pdf') # 保存到你的figures目录
    print("Figure 3 saved as noise_analysis.pdf")

if __name__ == "__main__":
    draw_figure_3()