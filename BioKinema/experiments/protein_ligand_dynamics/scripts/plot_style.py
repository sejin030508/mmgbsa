import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
import os
import matplotlib.font_manager as fm


# 字体文件夹路径
font_dir = "/cto_studio/xtalpi_lab/fengbin/Helvetica" 

# 批量添加所有字体
font_files = [
    'Helvetica.ttf',
    'Helvetica-Bold.ttf',
    'Helvetica-Oblique.ttf',
    'Helvetica-BoldOblique.ttf',
    'helvetica-light-587ebe5a59211.ttf',
]

for font_file in font_files:
    try:
        fm.fontManager.addfont(os.path.join(font_dir, font_file))
    except FileNotFoundError:
        print(f"Warning: Font file not found: {font_file}")

# --- 设置默认字体和全局黑色样式 ---
plt.rcParams['font.family'] = 'Helvetica'
font_size = 11
plt.rcParams['font.size'] = font_size
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1

# 强制所有元素为黑色
black_color = '#000000'
plt.rcParams['text.color'] = black_color
plt.rcParams['axes.labelcolor'] = black_color
plt.rcParams['xtick.color'] = black_color
plt.rcParams['ytick.color'] = black_color
plt.rcParams['axes.edgecolor'] = black_color
plt.rcParams['axes.titlecolor'] = black_color


import math
import pandas as pd
import numpy as np

def get_optimal_ticks(y_max, y_min=0):
    """
    计算一组刻度，满足：
    1. 范围覆盖 [y_min, y_max]
    2. 刻度数量在 4-6 个之间 (目标是 5 个)
    3. 步长是“漂亮”的数字 (1, 2, 2.5, 5 的倍数)
    4. 支持负数、小数和大整数
    """
    # 1. 基础检查与数据清洗
    if pd.isna(y_max) or pd.isna(y_min):
        return [0, 1]
    
    # 确保 y_max >= y_min
    if y_max < y_min:
        y_max, y_min = y_min, y_max
        
    # 如果最大值等于最小值，强制扩展一个小区间
    if y_max == y_min:
        if y_max == 0:
            return [0, 1]
        else:
            # 扩展 10% 或 至少 0.1
            offset = abs(y_max) * 0.1 if abs(y_max) > 0 else 1.0
            y_min -= offset / 2
            y_max += offset / 2

    # 2. 计算原始范围和理想步长
    data_range = y_max - y_min
    target_ticks = 4  # 我们期望大约 5 个刻度
    raw_step = data_range / (target_ticks - 1)

    # 3. 计算数量级 (magnitude)
    # 例如: raw_step = 0.045 -> mag = 0.01; raw_step = 450 -> mag = 100
    mag = 10 ** math.floor(math.log10(raw_step))
    
    # 4. 归一化步长 (将步长映射到 [1, 10) 区间)
    normalized_step = raw_step / mag
    
    # 5. 寻找最接近的“漂亮”步长
    # 候选倍数：1, 2, 2.5, 5, 10
    possible_steps = [1, 2, 2.5, 5, 10]
    
    # 找到离 normalized_step 最近的那个漂亮倍数
    best_norm_step = min(possible_steps, key=lambda x: abs(x - normalized_step))
    
    # 6. 还原真实步长
    step = best_norm_step * mag
    
    # 7. 根据步长重新计算“漂亮”的下界和上界
    # 使用 epsilon 防止浮点数精度问题 (例如 0.3 / 0.1 变成 2.9999)
    epsilon = step * 1e-6
    
    lower_bound = math.floor((y_min + epsilon) / step) * step
    upper_bound = math.ceil((y_max - epsilon) / step) * step
    
    # 8. 生成刻度列表
    # 这里的做法是先算出个数，再生成，避免 while 循环带来的风险
    # 即使数据极大，因为 step 是根据 range 算出来的，ticks 数量永远只会在 3-7 个左右
    n_intervals = round((upper_bound - lower_bound) / step)
    
    ticks = []
    for i in range(n_intervals + 1):
        val = lower_bound + i * step
        # 处理浮点数精度 (例如 0.300000000004 -> 0.3)
        # 根据 step 的小数位数来决定保留几位
        if step < 1:
            decimals = abs(math.floor(math.log10(step))) + 1 
            val = round(val, decimals)
        else:
            # 如果是整数步长，稍微保留一点精度防止 .00001 误差，或者直接取整
            val = round(val, 10) 
            if val.is_integer():
                val = int(val)
        ticks.append(val)

    return ticks
