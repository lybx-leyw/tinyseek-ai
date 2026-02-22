import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
"""
plot脚本：提取文件数据，绘画可视化折线图
"""

"""
在我的训练日志中，所有包含训练过程的数据都含多个数字
我们想绘制index-batch的图像，并在Conclusion中详细标注batch大小
"""

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def pause(path, index):
    x_plot = []
    y_plot = []
    with open(path, "r", encoding='utf-8') as file:
        x_index = 0
        for _, line in enumerate(file, 1):
            numbers = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', line.strip())
            if len(numbers) < 6:
                continue
            else:
                x_index += 1
                value = float(numbers[index])
                x_plot.append(x_index)
                y_plot.append(value)
    return x_plot, y_plot

def draw(x_plot, y_plot, save_path, label, use_log_fit=False):
    plt.figure(figsize=(18, 10), dpi=100, facecolor='white')
    plt.scatter(x_plot, y_plot, c='lightblue', s=5, alpha=0.3, 
               label='原始数据点', zorder=1)
    plt.plot(x_plot, y_plot, 'b-', linewidth=0.5, alpha=0.2, zorder=2)
    
    if len(x_plot) > 10:
        window = max(10, len(x_plot) // 30)  
        y_trend = pd.Series(y_plot).rolling(window=window, center=True, min_periods=1).mean()
        
        plt.plot(x_plot, y_trend, 'b-', linewidth=3, label=f'{label} (趋势线)', 
                alpha=0.9, zorder=3)
        
        from scipy.signal import argrelextrema
        if len(y_trend) > 10:
            n = max(3, len(y_trend) // 50) 
            local_max = argrelextrema(y_trend.values, np.greater, order=n)[0]
            local_min = argrelextrema(y_trend.values, np.less, order=n)[0]
            plt.scatter([x_plot[i] for i in local_max], 
                       [y_trend.iloc[i] for i in local_max], 
                       c='red', s=50, marker='^', alpha=0.8, zorder=4,
                       label='局部峰值')
            plt.scatter([x_plot[i] for i in local_min], 
                       [y_trend.iloc[i] for i in local_min], 
                       c='green', s=50, marker='v', alpha=0.8, zorder=4,
                       label='局部谷值')
    
    if use_log_fit and len(x_plot) > 3:
        try:
            from scipy.optimize import curve_fit
            
            def log_func(x, a, b):
                return a * np.log(x) + b
            
            popt, pcov = curve_fit(log_func, x_plot, y_plot)
            a, b = popt
            
            residuals = y_plot - log_func(np.array(x_plot), a, b)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((np.array(y_plot) - np.mean(y_plot))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            x_fit = np.linspace(min(x_plot), max(x_plot), 200)
            y_fit = log_func(x_fit, a, b)
            
            plt.plot(x_fit, y_fit, 'r--', linewidth=2.5, 
                    label=f'对数拟合 (R²={r_squared:.3f})', 
                    alpha=0.9, zorder=5)

            equation = f'y = {a:.2e}·ln(x) + {b:.2e}'
            plt.text(0.02, 0.98, equation, transform=plt.gca().transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                    
        except Exception as e:
            print(f"对数拟合失败: {e}")
    
    # 设置图表属性
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=8))
    plt.title(f"{label} - 趋势分析 (总数据点: {len(x_plot)})", 
              fontsize=12, fontweight='bold')
    plt.xlabel('batch', fontsize=10)
    plt.ylabel(label, fontsize=10)
    plt.grid(True, alpha=0.2, linestyle='--', color='gray')
    plt.legend(loc='best', fontsize=9)
    
    # 保存图表
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()

def draw_plt(process_name, log_name, use_log_fit=True):
    path = f"{log_name}.txt"
    x_plot, y_plot = pause(path=path, index=6)
    draw(x_plot, y_plot, f"docs\\figures\\loss_{process_name}.png", 'loss', use_log_fit=use_log_fit)
    
    x_plot, y_plot = pause(path=path, index=8)
    draw(x_plot, y_plot, f"docs\\figures\\ppx1_{process_name}.png", 'ppx1', use_log_fit=use_log_fit)
    
    x_plot, y_plot = pause(path=path, index=10)
    draw(x_plot, y_plot, f"docs\\figures\\ppx2_{process_name}.png", 'ppx2', use_log_fit=use_log_fit)

    x_plot, y_plot = pause(path=path, index=11)
    draw(x_plot, y_plot, f"docs\\figures\\lr_{process_name}.png", 'lr', use_log_fit=False)
    
    print(f"图表绘制完成！")
    print(f"- 生成文件: loss_{process_name}.png, ppx1_{process_name}.png, "
          f"ppx2_{process_name}.png, lr_{process_name}.png" )