import matplotlib.pyplot as plt
import seaborn as sns

# 设置 seaborn 风格，让图表看起来像顶会论文
sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'serif'

# 数据准备 (基于你的 result.md)
models = [
    {'name': 'MobileNetV2\n(Standard)', 'params': 3.4, 'robust': 30.79, 'clean': 69.35, 'type': 'Baseline'},
    {'name': 'MobileNetV2\n(AugMix)', 'params': 3.4, 'robust': 41.41, 'clean': 70.60, 'type': 'Baseline'},
    {'name': 'MobileNetV2\n(Ours)', 'params': 3.4, 'robust': 43.18, 'clean': 74.20, 'type': 'Ours'},
    {'name': 'ResNet18\n(AugMix)', 'params': 11.7, 'robust': 46.68, 'clean': 76.30, 'type': 'Competitor'},
    {'name': 'ResNet18\n(Prime/Teacher)', 'params': 11.7, 'robust': 53.20, 'clean': 77.60, 'type': 'Teacher'},
]

# 提取绘图数据
x = [m['params'] for m in models]
y = [m['robust'] for m in models]
sizes = [m['clean']**1.5 * 0.5 for m in models] # 气泡大小与 Clean Acc 相关

plt.figure(figsize=(8, 5))

# 绘制不同类型的点
colors = {'Baseline': 'gray', 'Ours': '#d62728', 'Competitor': '#1f77b4', 'Teacher': 'green'}
markers = {'Baseline': 'o', 'Ours': '*', 'Competitor': 's', 'Teacher': '^'}

for i, model in enumerate(models):
    plt.scatter(model['params'], model['robust'], 
                s=sizes[i], 
                c=colors[model['type']], 
                marker=markers[model['type']],
                alpha=0.8, edgecolors='black', linewidth=1.5,
                label=model['type'] if model['type'] not in plt.gca().get_legend_handles_labels()[1] else "")
    
    # 添加标签文字
    offset_y = 1.5
    if 'Standard' in model['name']: offset_y = -3
    if 'Ours' in model['name']: offset_y = 2
    
    plt.text(model['params'], model['robust'] + offset_y, model['name'], 
             fontsize=10, ha='center', va='center', fontweight='bold' if 'Ours' in model['name'] else 'normal')

# 装饰图表
plt.xlabel('Parameters (Millions)', fontsize=12)
plt.ylabel('CIFAR-100-C Mean Robust Accuracy (%)', fontsize=12)
plt.title('Robustness vs. Efficiency Frontier', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(1, 14)
plt.ylim(25, 60)

# 绘制箭头表示提升
plt.annotate('', xy=(3.4, 43.18), xytext=(3.4, 30.79),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
plt.text(3.6, 36, '+12.4% Robustness', fontsize=10, color='darkred', rotation=90)

# 保存
plt.tight_layout()
plt.savefig('./outputs/teaser.pdf', bbox_inches='tight')
print("Figure saved as ./outputs/teaser.pdf")