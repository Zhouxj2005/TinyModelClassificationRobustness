import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def draw_figure_2_fixed():
    # 1. 扩大画布高度 (5 -> 6)，防止上下被切
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 2. 调整 Y 轴范围，包含负半轴，让底部的圆能显示出来
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.8, 5.5) 
    ax.axis('off')

    # --- 样式配置 ---
    color_student = '#E3F2FD' # 浅蓝
    color_teacher = '#FCE4EC' # 浅粉
    color_augmix = '#F5F5F5'  # 浅灰
    color_loss_kd = '#FFF3E0' # 浅橙
    color_loss_js = '#E1F5FE' # 极浅蓝
    color_loss_ce = '#E8F5E9' # 浅绿
    
    edge_color = '#455A64'
    linewidth = 1.5
    font_family = 'serif'

    def draw_box(x, y, w, h, text, color, label=None, text_color='black'):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.2", 
                             fc=color, ec=edge_color, lw=linewidth)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=11, fontweight='bold', family=font_family, linespacing=1.4, color=text_color)
        if label:
            ax.text(x + w/2, y + h + 0.15, label, ha='center', va='bottom', 
                    fontsize=9, style='italic', color='#555555')
        return box

    # -----------------------------
    # 1. Input Module
    # -----------------------------
    rect1 = patches.Rectangle((0.5, 2.3), 1, 1, fc='white', ec=edge_color, lw=1)
    rect2 = patches.Rectangle((0.6, 2.2), 1, 1, fc='white', ec=edge_color, lw=1)
    rect3 = patches.Rectangle((0.7, 2.1), 1, 1, fc='white', ec=edge_color, lw=linewidth)
    ax.add_patch(rect1); ax.add_patch(rect2); ax.add_patch(rect3)
    ax.text(1.2, 2.6, "Input Image\n$x$", ha='center', va='center', fontsize=11, family=font_family)

    # -----------------------------
    # 2. AugMix Module
    # -----------------------------
    aug_box = FancyBboxPatch((2.5, 1.2), 2.0, 3.0, boxstyle="round,pad=0.1", 
                             fc=color_augmix, ec='gray', lw=1.5, linestyle='--')
    ax.add_patch(aug_box)
    ax.text(3.5, 4.3, "AugMix Pipeline", ha='center', fontsize=11, fontweight='bold', color='gray')

    draw_box(2.8, 3.4, 1.4, 0.5, "$x_{aug1}$", "white")
    draw_box(2.8, 2.5, 1.4, 0.5, "$x_{aug2}$", "white")
    draw_box(2.8, 1.6, 1.4, 0.5, "$x_{clean}$", "white")
    
    # 箭头
    arrow_style = dict(arrowstyle="->", lw=1.5, color=edge_color)
    ax.annotate("", xy=(2.8, 3.65), xytext=(1.7, 2.6), arrowprops=arrow_style)
    ax.annotate("", xy=(2.8, 2.75), xytext=(1.7, 2.6), arrowprops=arrow_style)
    ax.annotate("", xy=(2.8, 1.85), xytext=(1.7, 2.6), arrowprops=arrow_style)

    # -----------------------------
    # 3. Backbone Models (文字替换了Emoji)
    # -----------------------------
    # 汇聚点
    ax.annotate("", xy=(5.2, 2.7), xytext=(4.3, 3.65), arrowprops=dict(arrowstyle="-", lw=1.5, color=edge_color))
    ax.annotate("", xy=(5.2, 2.7), xytext=(4.3, 1.85), arrowprops=dict(arrowstyle="-", lw=1.5, color=edge_color))
    ax.annotate("", xy=(5.2, 2.7), xytext=(4.3, 2.75), arrowprops=dict(arrowstyle="-", lw=1.5, color=edge_color))
    ax.plot(5.2, 2.7, 'o', color=edge_color, markersize=5)

    # Teacher (Top) - 去掉Emoji，用文字 Fixed
    draw_box(5.8, 3.4, 2.2, 1.2, "Teacher\n(ResNet18)\n[Fixed]", color_teacher)
    
    # Student (Bottom) - 去掉Emoji，用文字 Gradient
    draw_box(5.8, 0.8, 2.2, 1.2, "Student\n(MobileNetV2)\n[Gradient]", color_student)

    # 箭头
    ax.annotate("", xy=(5.8, 4.0), xytext=(5.2, 2.7), arrowprops=arrow_style)
    ax.annotate("", xy=(5.8, 1.4), xytext=(5.2, 2.7), arrowprops=arrow_style)

    # -----------------------------
    # 4. Logits & Losses
    # -----------------------------
    
    # Teacher Logits
    draw_box(8.8, 3.7, 1.2, 0.6, "$f_T(x_{all})$", "white")
    # Student Logits
    draw_box(8.8, 1.1, 1.2, 0.6, "$f_S(x_{all})$", "white")
    
    ax.annotate("", xy=(8.8, 4.0), xytext=(8.0, 4.0), arrowprops=arrow_style)
    ax.annotate("", xy=(8.8, 1.4), xytext=(8.0, 1.4), arrowprops=arrow_style)

    # --- Losses ---
    
    # KD Loss
    circle_kd = patches.Circle((9.4, 2.7), 0.5, fc=color_loss_kd, ec='orange', lw=2, linestyle='--')
    ax.add_patch(circle_kd)
    ax.text(9.4, 2.7, "$\mathcal{L}_{KD}$\n(Soft)", ha='center', va='center', fontsize=10, fontweight='bold', color='#D84315')
    
    ax.annotate("", xy=(9.4, 3.2), xytext=(9.4, 3.7), arrowprops=dict(arrowstyle="-", ls="--", color='orange'))
    ax.annotate("", xy=(9.4, 2.2), xytext=(9.4, 1.7), arrowprops=dict(arrowstyle="->", ls="--", color='orange'))

    # JS Loss (位置上移了一点，画布下移了一点，确保不被切)
    # 圆心在 y=0.1, 半径0.45, 最低点 -0.35 (现在画布到 -0.8，肯定安全)
    circle_js = patches.Circle((9.4, 0.1), 0.45, fc=color_loss_js, ec='#0288D1', lw=2)
    ax.add_patch(circle_js)
    ax.text(9.4, 0.1, "$\mathcal{L}_{JS}$\n(Consist)", ha='center', va='center', fontsize=9, fontweight='bold', color='#0277BD')
    
    ax.annotate("", xy=(9.4, 0.55), xytext=(9.4, 1.1), arrowprops=dict(arrowstyle="<-", color='#0288D1', lw=1.5))

    # CE Loss
    draw_box(11.0, 1.1, 0.8, 0.6, "$y$", "white")
    
    circle_ce = patches.Circle((10.5, 1.4), 0.3, fc=color_loss_ce, ec='green', lw=2)
    ax.add_patch(circle_ce)
    ax.text(10.5, 1.4, "$\mathcal{L}_{CE}$", ha='center', va='center', fontsize=9, fontweight='bold', color='green')

    ax.annotate("", xy=(10.2, 1.4), xytext=(10.0, 1.4), arrowprops=dict(arrowstyle="->", color='green'))
    ax.annotate("", xy=(10.8, 1.4), xytext=(11.0, 1.4), arrowprops=dict(arrowstyle="->", color='green'))

    plt.tight_layout()
    plt.savefig('outputs/method_overview_fixed.pdf')
    print("Fixed Figure 2 saved as method_overview_fixed.pdf")

if __name__ == "__main__":
    draw_figure_2_fixed()