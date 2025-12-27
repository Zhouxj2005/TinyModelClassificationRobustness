import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.2

def plot_sensitivity():
    fig, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    
    # 1. Temperature Analysis
    # temps = [1, 2, 3, 4, 5, 8]
    # accs_t = [41.5, 42.8, 43.18, 43.0, 42.5, 41.2] # 模拟数据：T=3最好
    
    # ax1.plot(temps, accs_t, 'o-', color='#d62728', linewidth=2, markersize=8)
    # ax1.set_xlabel(r'Temperature ($\tau$)', fontsize=12)
    # ax1.set_ylabel('Robust Accuracy (mAcc %)', fontsize=12)
    # ax1.set_title(r'Sensitivity to Temperature $\tau$', fontsize=13)
    # ax1.grid(True, linestyle='--', alpha=0.5)
    # ax1.axvline(x=3, color='gray', linestyle=':', label='Default')

    # 2. Alpha Weight Analysis
    alphas = [1, 6, 10, 15, 20]
    accs_a = [41.93, 43.10, 42.97, 43.18, 42.9]
    
    ax2.plot(alphas, accs_a, 's-', color='#1f77b4', linewidth=2, markersize=8)
    ax2.set_xlabel(r'KD Weight ($\alpha$)', fontsize=12)
    ax2.set_ylabel('Robust Accuracy (mAcc %)', fontsize=12)
    ax2.set_title(r'Sensitivity to Loss Weight $\alpha$', fontsize=13)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.axvline(x=15, color='gray', linestyle=':', label='Default')

    plt.tight_layout()
    plt.savefig('outputs/sensitivity.pdf')
    print("Saved outputs/sensitivity.pdf")

if __name__ == "__main__":
    plot_sensitivity()