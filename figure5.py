import pandas as pd
import matplotlib.pyplot as plt
import io

# 你的 CSV 数据（我已经根据你提供的内容做了简化处理，你可以直接读取文件）
def plot_real_dynamics():
    # 读取你提供的 CSV 文件
    # 假设文件名分别是 'log_ours.csv' 和 'log_rkd.csv'
    # 如果你没有改文件名，请修改这里的路径
    try:
        df_ours = pd.read_csv('./checkpoint/mobilenet_v2_distill_15.0_200_0.05_log.csv')
        df_rkd = pd.read_csv('./checkpoint/mobilenet_v2_distill_improved_log.csv')
    except FileNotFoundError:
        print("请确保csv文件在当前目录下。")
        return

    plt.figure(figsize=(7, 5))
    
    # 设置风格
    plt.rcParams['font.family'] = 'serif'
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # 绘制曲线
    plt.plot(df_ours['epoch'], df_ours['clean_acc'], 
             label='Ours (Logits KD)', color='#d62728', linewidth=2)
    plt.plot(df_rkd['epoch'], df_rkd['clean_acc'], 
             label='Ours + RKD (Feature KD)', color='#1f77b4', linewidth=2)
    
    # 标注最终结果
    final_acc_ours = df_ours['clean_acc'].iloc[-1]
    final_acc_rkd = df_rkd['clean_acc'].iloc[-1]
    
    plt.scatter(200, final_acc_ours, color='#d62728')
    plt.text(205, final_acc_ours, f'{final_acc_ours:.1f}%', va='center', color='#d62728', fontweight='bold')
    
    plt.scatter(200, final_acc_rkd, color='#1f77b4')
    plt.text(205, final_acc_rkd, f'{final_acc_rkd:.1f}%', va='center', color='#1f77b4', fontweight='bold')

    # 装饰
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Clean Validation Accuracy (%)', fontsize=12)
    plt.title('Training Dynamics: Impact of Feature Alignment', fontsize=13)
    plt.legend(fontsize=11, loc='lower right')
    plt.xlim(0, 230) # 给右边的文字留点位置
    plt.ylim(10, 80)

    plt.tight_layout()
    plt.savefig('outputs/training_dynamics_real.pdf')
    print("Saved outputs/training_dynamics_real.pdf")

if __name__ == "__main__":
    plot_real_dynamics()