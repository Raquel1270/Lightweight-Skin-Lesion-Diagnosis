import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 数据准备 (保持不变)
# ==========================================
x_epoch = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
gamma_converge = [1.0, 0.9356, 0.8324, 0.7585, 0.7055, 0.6715, 0.6564, 0.6413, 0.6307, 0.6178, 0.6073, 0.6006, 0.5916]
temp_converge = [5.0, 4.9468, 4.9093, 4.8682, 4.8254, 4.7974, 4.7810, 4.7728, 4.7712, 4.7706, 4.7760, 4.7729, 4.7727]
x_gamma = [0.6564, 0.6413, 0.6307, 0.6178, 0.6073, 0.6006, 0.5916]
y_gamma = [79.49, 82.37, 84.67, 86.63, 89.41, 91.62, 93.72]
x_tau = [4.7732, 4.7728, 4.7722, 4.7726, 4.7730, 4.7729, 4.7727]
y_tau = [81.49, 82.37, 84.67, 86.63, 89.41, 91.62, 93.72]
x_shot = [1, 5]
y_baseline = [82.05, 87.35]
y_ours     = [82.11, 93.72]

# ==========================================
# 2. 绘图全局设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 1.0

# ===================== 关键调整 =====================
# 创建画布：扩大整体宽度，为右侧图例预留空白
fig = plt.figure(figsize=(15, 10), dpi=300)  # 宽度从12→15，新增3单位给右侧图例
# 调整子图布局：4个子图仅占左侧75%区域，右侧25%留空放图例
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.5)
ax1 = fig.add_subplot(gs[0, 0])   # 左上子图（占第1列）
ax2 = fig.add_subplot(gs[0, 1])   # 右上子图（占第2列）
ax3 = fig.add_subplot(gs[1, 0])   # 左下子图（占第1列）
ax4 = fig.add_subplot(gs[1, 1])   # 右下子图（占第2列）
# 右侧第3列完全留空，专门放图例

# 颜色/样式定义（不变）
color_gamma = '#1f77b4'
color_temp = '#ff7f0e'
color_baseline = '#2ca02c'
color_ours = '#d62728'
line_width = 2.0
marker_size = 7

# ==========================================
# 3. 子图1 (左上) - 完全不变
# ==========================================
ax1_twin = ax1.twinx()
ax1.plot(x_epoch, gamma_converge, marker='o', color=color_gamma, linewidth=line_width, markersize=marker_size)
ax1_twin.plot(x_epoch, temp_converge, marker='s', color=color_temp, linewidth=line_width, markersize=marker_size)
gamma_opt = 0.5916
temp_opt = 4.7727
ax1.axhline(y=gamma_opt, color=color_gamma, linestyle='--', alpha=0.7)
ax1_twin.axhline(y=temp_opt, color=color_temp, linestyle='--', alpha=0.7)
ax1.annotate(f'$\gamma_{{opt}}={gamma_opt}$', xy=(60, gamma_opt), xytext=(40, 0.75),
             arrowprops=dict(facecolor=color_gamma, shrink=0.05, width=1, headwidth=6),
             color=color_gamma, fontweight='bold', fontsize=9)
ax1_twin.annotate(f'$\\tau_{{opt}}={temp_opt}$', xy=(60, temp_opt), xytext=(40, 4.9),
                  arrowprops=dict(facecolor=color_temp, shrink=0.05, width=1, headwidth=6),
                  color=color_temp, fontweight='bold', fontsize=9)
ax1.set_title('(a) Adaptive Parameter Convergence', fontweight='bold', pad=12)
ax1.set_xlabel('Training Epoch')
ax1.set_ylabel('$\gamma$ Value', color=color_gamma)
ax1_twin.set_ylabel('$\\tau$ Value', color=color_temp)
ax1.tick_params(axis='y', labelcolor=color_gamma)
ax1_twin.tick_params(axis='y', labelcolor=color_temp)
ax1.set_xticks(np.arange(0, 61, 10))
ax1.set_ylim(0.55, 1.05)
ax1_twin.set_ylim(4.7, 5.05)
ax1.grid(True, linestyle='--', alpha=0.5)

# ==========================================
# 4. 子图2 (右上) - 完全不变
# ==========================================
ax2.plot(y_gamma, x_gamma, marker='o', color=color_gamma, linewidth=line_width, markersize=marker_size)
max_y_gamma = max(y_gamma)
max_x_gamma = x_gamma[y_gamma.index(max_y_gamma)]
ax2.annotate(f'Optimum: {max_x_gamma}\nAcc: {max_y_gamma}%', xy=(max_y_gamma, max_x_gamma),
              xytext=(max_y_gamma-5, max_x_gamma+0.05),
              arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
              ha='center', fontsize=9)
ax2.set_title('(b) Balanced Accuracy and $\gamma$', fontweight='bold', pad=12)
ax2.set_xlabel('Balanced Accuracy (%)')
ax2.set_ylabel('Adaptive $\gamma$ Value')
ax2.set_xlim(78, 95)
ax2.set_ylim(0.58, 0.67)
ax2.grid(True, linestyle='--', alpha=0.5)

# ==========================================
# 5. 子图3 (左下) - 完全不变
# ==========================================
ax3.plot(y_tau, x_tau, marker='s', color=color_temp, linewidth=line_width, markersize=marker_size)
max_y_tau = max(y_tau)
max_x_tau = x_tau[y_tau.index(max_y_tau)]
ax3.annotate(f'Optimum: {max_x_tau:.4f}\nAcc: {max_y_tau}%', xy=(max_y_tau, max_x_tau),
             xytext=(max_y_tau-5, max_x_tau+0.005),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
             ha='center', fontsize=9)
ax3.set_title('(c) Balanced Accuracy and $\\tau$', fontweight='bold', pad=12)
ax3.set_xlabel('Balanced Accuracy (%)')
ax3.set_ylabel('Adaptive $\\tau$ Value')
ax3.set_xlim(78, 95)
ax3.set_ylim(4.77, 4.785)
ax3.grid(True, linestyle='--', alpha=0.5)

# ==========================================
# 6. 子图4 (右下) - 完全不变
# ==========================================
ax4.plot(x_shot, y_baseline, marker='^', color=color_baseline, linestyle='--',
         linewidth=line_width, markersize=marker_size)
ax4.plot(x_shot, y_ours, marker='*', color=color_ours, linewidth=line_width,
         markersize=marker_size+2)
ax4.fill_between(x_shot, y_baseline, y_ours, color=color_ours, alpha=0.1)
avg_improve = np.mean(np.array(y_ours) - np.array(y_baseline))
ax4.text(3, 86, f'Avg. Improvement: {avg_improve:.2f}%', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), ha='center')
ax4.set_title('(d) Performance and Shot Count', fontweight='bold', pad=12)
ax4.set_xlabel('Number of Support Shots ($K$)')
ax4.set_ylabel('Balanced Accuracy (%)')
ax4.set_xticks(x_shot)
ax4.set_ylim(80, 95)
ax4.grid(True, linestyle='--', alpha=0.5)

# ==========================================
# 7. 图例：放在整张图右侧空白区（完全独立，不碰任何子图）
# ==========================================
legend_elements = [
    plt.Line2D([0], [0], color=color_gamma, marker='o', linewidth=line_width, markersize=marker_size, label='$\gamma$ (Feature Rectification)'),
    plt.Line2D([0], [0], color=color_temp, marker='s', linewidth=line_width, markersize=marker_size, label='$\\tau$ (Temperature)'),
    plt.Line2D([0], [0], color=color_baseline, marker='^', linestyle='--', linewidth=line_width, markersize=marker_size, label='Baseline (MobileViT)'),
    plt.Line2D([0], [0], color=color_ours, marker='*', linewidth=line_width, markersize=marker_size+2, label='Ours (Adaptive Params)')
]
# 图例放在右侧空白区（第3列），垂直居中
fig.legend(handles=legend_elements,
           loc='center',                # 垂直居中
           bbox_to_anchor=(0.75, 0.3),  # 右侧空白区核心位置（x=0.85是新增的空白区，y=0.5垂直居中）
           fontsize=10,
           frameon=True,
           edgecolor='black',
           fancybox=False,
           framealpha=1.0,
           ncol=1)

# ==========================================
# 8. 保存（包含右侧图例区）
# ==========================================
plt.savefig('adaptive_parameter_final.pdf', bbox_inches='tight', dpi=300)
plt.savefig('adaptive_parameter_final.png', bbox_inches='tight', dpi=300)
plt.show()