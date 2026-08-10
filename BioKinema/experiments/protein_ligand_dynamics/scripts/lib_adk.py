# -*- coding: utf-8 -*-
# Auto-extracted VERBATIM from the manuscript notebook(s), with only the
# final figure parameters baked in (documented inline). Do not hand-edit the
# analysis logic here -- it is the exact code that produced the paper figures.
# Source: fig4b_induced_fit.ipynb (cells 0-3)



#!/usr/bin/env python3
"""
计算CIF文件的RMSD和RMSF
使用TM-align进行结构比对
"""

import os
import re
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict
from Bio.PDB import MMCIFParser
from Bio.PDB.Structure import Structure
import tmtools
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

warnings.filterwarnings('ignore')


# 三字母到单字母氨基酸转换
THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'MSE': 'M', 'SEC': 'C', 'PYL': 'K', 'UNK': 'X'
}


def parse_filename(filename: str) -> Tuple[int, int]:
    """
    从文件名中提取sample和frame编号
    例如: 4AKE_A_R1_0_0_s0_f0_wounresol.cif -> (0, 0)
    """
    pattern = r's(\d+)_f(\d+)'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        raise ValueError(f"无法从文件名 {filename} 中解析sample和frame编号")


def get_ca_info(structure: Structure) -> Tuple[np.ndarray, str]:
    """
    从结构中提取CA原子坐标和序列
    返回: (coords, sequence_string)
    """
    coords = []
    sequence = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ' and 'CA' in residue:
                    coords.append(residue['CA'].get_coord())
                    resname = residue.get_resname()
                    sequence.append(THREE_TO_ONE.get(resname, 'X'))
        break  # 只取第一个model
    
    return np.array(coords), ''.join(sequence)


def load_structure(cif_path: str, parser: MMCIFParser) -> Structure:
    """加载CIF结构"""
    structure_id = os.path.basename(cif_path).replace('.cif', '')
    return parser.get_structure(structure_id, cif_path)


def calculate_rmsd_tmalign(coords1: np.ndarray, seq1: str,
                           coords2: np.ndarray, seq2: str) -> float:
    """使用TM-align计算RMSD"""
    if len(coords1) == 0 or len(coords2) == 0:
        return float('inf')
    
    result = tmtools.tm_align(coords1, coords2, seq1, seq2)
    return result.rmsd


def process_single_file(args: Tuple) -> Tuple[int, int, float, float, np.ndarray, str]:
    """
    处理单个CIF文件，计算与两个参考构象的RMSD
    """
    filepath, sample_id, frame_id, coords_a, seq_a, coords_b, seq_b = args
    
    try:
        parser = MMCIFParser(QUIET=True)
        struct = load_structure(filepath, parser)
        coords, seq = get_ca_info(struct)
        
        rmsd_a = calculate_rmsd_tmalign(coords, seq, coords_a, seq_a)
        rmsd_b = calculate_rmsd_tmalign(coords, seq, coords_b, seq_b)
        
        return sample_id, frame_id, rmsd_a, rmsd_b, coords, seq
        
    except Exception as e:
        print(f"处理 {filepath} 出错: {e}")
        return sample_id, frame_id, float('nan'), float('nan'), None, None


def align_single_frame(args: Tuple) -> np.ndarray:
    """
    将单个帧叠合到参考帧
    """
    idx, coords, seq, ref_coords, ref_seq, n_residues = args
    
    if len(coords) != n_residues:
        return None
    
    try:
        result = tmtools.tm_align(coords, ref_coords, seq, ref_seq)
        transformed = coords @ result.u.T + result.t
        return transformed
    except Exception:
        return None


def main(
    conf_a_path: str,
    conf_b_path: str,
    folder_path: str,
    n_workers: int = None
) -> Tuple[List[List[float]], List[List[float]], List[float]]:
    """
    主函数
    
    参数:
        conf_a_path: 构象A的CIF文件路径
        conf_b_path: 构象B的CIF文件路径
        folder_path: 包含CIF文件的文件夹路径
        n_workers: 并行进程数，默认为CPU核心数
    
    返回:
        rmsd_a: 与构象A的RMSD [[s0_f0, s0_f1, ...], [s1_f0, s1_f1,], ...]
        rmsd_b: 与构象B的RMSD [[s0_f0, s0_f1, ...], [s1_f0, s1_f1,], ...]
        rmsf: CA原子的RMSF列表
    """
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    
    parser = MMCIFParser(QUIET=True)
    
    # 加载参考构象
    print("加载参考构象...")
    struct_a = load_structure(conf_a_path, parser)
    coords_a, seq_a = get_ca_info(struct_a)
    print(f"构象A: {len(coords_a)} 个CA原子")
    
    struct_b = load_structure(conf_b_path, parser)
    coords_b, seq_b = get_ca_info(struct_b)
    print(f"构象B: {len(coords_b)} 个CA原子")
    
    # 扫描文件夹
    file_mapping = defaultdict(dict)  # {sample_id: {frame_id: filepath}}
    cif_files = [f for f in os.listdir(folder_path) if f.endswith('.cif')]
    print(f"找到 {len(cif_files)} 个CIF文件")
    
    for filename in cif_files:
        try:
            sample_id, frame_id = parse_filename(filename)
            file_mapping[sample_id][frame_id] = os.path.join(folder_path, filename)
        except ValueError as e:
            print(f"跳过: {e}")
    
    all_samples = sorted(file_mapping.keys())
    
    # 准备并行任务
    tasks = []
    for sample_id in all_samples:
        for frame_id in sorted(file_mapping[sample_id].keys()):
            filepath = file_mapping[sample_id][frame_id]
            tasks.append((filepath, sample_id, frame_id, coords_a, seq_a, coords_b, seq_b))
    
    total = len(tasks)
    
    # 并行计算RMSD
    rmsd_a_results = defaultdict(dict)
    rmsd_b_results = defaultdict(dict)
    all_coords_for_rmsf = []
    all_seqs_for_rmsf = []
    
    print(f"使用 {n_workers} 个进程并行计算RMSD...")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks}
        processed = 0
        
        for future in as_completed(futures):
            processed += 1
            if processed % 50 == 0:
                print(f"进度: {processed}/{total}")
            
            sample_id, frame_id, rmsd_a, rmsd_b, coords, seq = future.result()
            rmsd_a_results[sample_id][frame_id] = rmsd_a
            rmsd_b_results[sample_id][frame_id] = rmsd_b
            
            if coords is not None and frame_id >= 400:  # RMSF over 4-5us window only (501 frames @10ns)
                all_coords_for_rmsf.append(coords)
                all_seqs_for_rmsf.append(seq)
    
    # 转换RMSD结果为列表格式
    rmsd_a_list = []
    rmsd_b_list = []
    
    for sample_id in all_samples:
        frames_a = rmsd_a_results[sample_id]
        frames_b = rmsd_b_results[sample_id]
        max_frame = max(frames_a.keys()) if frames_a else -1
        
        rmsd_a_list.append([frames_a.get(f, float('nan')) for f in range(max_frame + 1)])
        rmsd_b_list.append([frames_b.get(f, float('nan')) for f in range(max_frame + 1)])
    
    # 并行计算RMSF
    print("计算RMSF...")
    rmsf = calculate_rmsf_parallel(all_coords_for_rmsf, all_seqs_for_rmsf, n_workers)
    
    print(f"\n完成! Sample数: {len(rmsd_a_list)}, RMSF残基数: {len(rmsf)}")
    
    return rmsd_a_list, rmsd_b_list, rmsf


def calculate_rmsf_parallel(all_coords: List[np.ndarray], all_seqs: List[str], 
                            n_workers: int = None) -> List[float]:
    """
    并行计算RMSF
    将所有帧叠合到第一帧后计算每个残基的波动
    """
    if len(all_coords) < 2:
        return []
    
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    
    # 使用第一帧作为参考
    ref_coords = all_coords[0]
    ref_seq = all_seqs[0]
    n_residues = len(ref_coords)
    
    # 准备并行任务（跳过后半部分，因为它是参考帧）
    tasks = [
        (i, coords, seq, ref_coords, ref_seq, n_residues)
        for i, (coords, seq) in enumerate(zip(all_coords[1:], all_seqs[1:]))
    ]
    
    # 并行叠合所有帧到参考帧
    aligned_coords = [ref_coords]
    
    print(f"使用 {n_workers} 个进程并行叠合 {len(tasks)} 帧...")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(align_single_frame, task): task[0] for task in tasks}
        processed = 0
        total = len(tasks)
        
        for future in as_completed(futures):
            processed += 1
            if processed % 50 == 0:
                print(f"叠合进度: {processed}/{total}")
            
            result = future.result()
            if result is not None:
                aligned_coords.append(result)
    
    if len(aligned_coords) < 2:
        return []
    
    # 计算RMSF
    coords_matrix = np.array(aligned_coords)  # (n_frames, n_residues, 3)
    mean_coords = np.mean(coords_matrix, axis=0)
    
    deviations = coords_matrix - mean_coords
    squared_dev = np.sum(deviations ** 2, axis=2)
    rmsf = np.sqrt(np.mean(squared_dev, axis=0))
    
    return rmsf.tolist()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
sns.set_palette("mako_r")
# 注册 seaborn colormap 到 matplotlib
import matplotlib.pyplot as plt

global_set_title = False
global_set_xlabel = True


def get_density(v):
    cov = np.transpose(v)
    density = gaussian_kde(cov)(cov)
    return density


def make_subplot(ax, data, title, ax0=None):
    try:
        if global_set_title:
            ax.set_title(title)
        else:
            ax.set_title("")
        ax.scatter(data[:,0], data[:, 1], s=10, alpha=0.7, c=get_density(data), cmap="mako_r", vmin=-0.05, vmax=1.0)
    except Exception as e:
        print(f"Plotting failed: {e}")
        
    if ax0 is not None:
        ax.set_xlim(ax0.get_xlim()[0], ax0.get_xlim()[1])
        ax.set_ylim(ax0.get_ylim()[0], ax0.get_ylim()[1])


def prepare_rmsd_scatter_data(rmsd_a_list: list, rmsd_b_list: list) -> np.ndarray:
    """
    将main函数返回的rmsd_list转换为散点图数据
    
    参数:
        rmsd_a_list: [[s0_f0, s0_f1, ...], [s1_f0, s1_f1,], ...]
        rmsd_b_list: [[s0_f0, s0_f1, ...], [s1_f0, s1_f1,], ...]
    
    返回:
        np.ndarray of shape (N, 2), 每行是 [rmsd_a, rmsd_b]
    """
    points = []
    for sample_a, sample_b in zip(rmsd_a_list, rmsd_b_list):
        for rmsd_a, rmsd_b in zip(sample_a, sample_b):
            if not np.isnan(rmsd_a) and not np.isnan(rmsd_b):
                points.append([rmsd_a, rmsd_b])
    return np.array(points)


def plot_rmsd_scatter(
    rmsd_a_list: list,
    rmsd_b_list: list,
    name: str = "Protein",
    output_file: str = "./rmsd_scatter.pdf",
    xlabel: str = "RMSD to Conf A (Å)",
    ylabel: str = "RMSD to Conf B (Å)",
):
    """
    绘制RMSD散点图
    
    参数:
        rmsd_a_list: main函数返回的rmsd_a
        rmsd_b_list: main函数返回的rmsd_b
        name: 蛋白名称
        output_file: 输出文件路径
    """
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
    
    # 准备数据
    data = prepare_rmsd_scatter_data(rmsd_a_list, rmsd_b_list)
    print(f"Plotting RMSD scatter with {len(data)} points")
    
    # 绘制散点图
    make_subplot(ax, data, name, ax0=None)
    
    # 设置标签
    ax.set_ylabel(ylabel)
    if global_set_xlabel:
        ax.set_xlabel(xlabel)

    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.yticks([0, 1, 2, 3, 4, 5])
    plt.xticks([0, 1, 2, 3, 4, 5])

    fig.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图片已保存为: {output_file}")
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import os


def prepare_rmsd_for_plot(
    rmsd_list: list,
    label: str
) -> tuple:
    """
    将main函数返回的rmsd_list转换为plot_rmsd需要的格式
    
    参数:
        rmsd_list: [[s0_f0, s0_f1, ...], [s1_f0, s1_f1,], ...] 来自main函数
        label: 数据标签
    
    返回:
        (label, frame_indices, rmsd_mean, rmsd_std)
    """
    if not rmsd_list or not rmsd_list[0]:
        return (label, [], [], [])
    
    # 转换为numpy数组，处理长度不一致的情况
    max_frames = max(len(sample) for sample in rmsd_list)
    
    # 填充为相同长度
    padded = []
    for sample in rmsd_list:
        if len(sample) < max_frames:
            sample = list(sample) + [float('nan')] * (max_frames - len(sample))
        padded.append(sample)
    
    rmsd_array = np.array(padded)  # (n_samples, n_frames)
    
    # 计算均值和标准差（忽略nan）
    rmsd_min = np.min(rmsd_array, axis=0)[1:]
    rmsd_max = np.max(rmsd_array, axis=0)[1:]
    frame_indices = np.arange(len(rmsd_min))
    
    return (label, frame_indices, rmsd_min, rmsd_max)


def plot_rmsd(
    data_list,
    output_file="./plot/figures_main/rmsd_separate_regions.pdf",
    xlabel="Frame",
    ylabel="Å",
    title=None,
    show_std=True
):
    """
    绘制RMSD曲线图（带标准差阴影）。
    
    参数:
        data_list: list of (folder_label, frame_indices, rmsd_mean, rmsd_std)
        output_file: str, 输出文件路径
        xlabel: str, X轴标签
        ylabel: str, Y轴标签
        title: str or None, 图标题
        show_std: bool, 是否显示标准差阴影
    """
    import matplotlib.font_manager as fm
    
    # 尝试加载 Helvetica 字体
    font_dir = "/cto_studio/xtalpi_lab/fengbin/Helvetica"
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
            pass

    fig, ax = plt.subplots(figsize=(5, 1.55))
    
    # 设置默认字体和样式
    plt.rcParams['font.family'] = 'Helvetica'
    font_size = 11
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.width'] = 1

    black_color = '#000000'
    plt.rcParams['text.color'] = black_color
    plt.rcParams['axes.labelcolor'] = black_color
    plt.rcParams['xtick.color'] = black_color
    plt.rcParams['ytick.color'] = black_color
    plt.rcParams['axes.edgecolor'] = black_color
    
    if not data_list:
        print("没有有效的数据可供绘图。")
        return

    color_list = ["#356a9f", "#ac3d48", "#00A087", "#E64B35", "#4DBBD5", "#7E6148"]
    
    for idx, item in enumerate(data_list):
        folder_label, frame_indices, rmsd_min, rmsd_max = item
        color = color_list[idx % len(color_list)]
        
        # 绘制均值曲线
        ax.plot(frame_indices, (rmsd_min + rmsd_max) / 2, label=folder_label, linewidth=2, alpha=1, color=color)
        
        # 绘制标准差阴影
        if show_std and rmsd_max is not None:
            ax.fill_between(
                frame_indices,
                rmsd_min,
                rmsd_max,
                alpha=0.2,
                color=color
            )
    
    # ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    plt.xlim(0, 500)
    plt.ylim(0, 6)
    plt.yticks([0, 3, 6], [" 0", " 3", " 6"])
    plt.xticks([i * 50 for i in range(11)], [f"{i * 0.5:.1f}" for i in range(11)])  # 5us axis
    fig.legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=font_size)
    plt.tight_layout()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"\n绘图完成！图片已保存为: {output_file}")
    plt.show()


def plot_rmsd_from_main(
    rmsd_a_list: list,
    rmsd_b_list: list,
    label_a: str = "RMSD to Conf A",
    label_b: str = "RMSD to Conf B",
    output_file: str = "./rmsd_plot.pdf",
    **kwargs
):
    """
    直接使用main函数的输出绘制RMSD图
    
    参数:
        rmsd_a_list: main函数返回的rmsd_a
        rmsd_b_list: main函数返回的rmsd_b
        label_a: 构象A的图例标签
        label_b: 构象B的图例标签
        output_file: 输出文件路径
        **kwargs: 传递给plot_rmsd的其他参数
    """
    data_list = [
        prepare_rmsd_for_plot(rmsd_a_list, label_a),
        prepare_rmsd_for_plot(rmsd_b_list, label_b),
    ]
    
    plot_rmsd(data_list, output_file=output_file, **kwargs)

    

def make_rmsf_plot(rmsf_list, labels, output_file="./plot/figures_main/fig4_rmsf.pdf"):
    """
    根据RMSF数据列表绘制RMSF图。
    
    参数:
        rmsf_list: RMSF数据列表，每个元素是一个残基RMSF值的列表
        labels: 每条曲线的标签列表
        output_file: 输出文件路径
    """
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import numpy as np
    import os
    
    font_dir = "/cto_studio/xtalpi_lab/fengbin/Helvetica" 

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

    fig, ax = plt.subplots(figsize=(3.5, 1.55))
    plt.rcParams['font.family'] = 'Helvetica'
    font_size = 11
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.width'] = 1

    black_color = '#000000'
    plt.rcParams['text.color'] = black_color
    plt.rcParams['axes.labelcolor'] = black_color
    plt.rcParams['xtick.color'] = black_color
    plt.rcParams['ytick.color'] = black_color
    plt.rcParams['axes.edgecolor'] = black_color
    plt.rcParams['axes.titlecolor'] = black_color
    
    if not rmsf_list:
        print("没有有效的数据可供绘图。")
        return

    color_list = [
        "#356a9f", "#ac3d48", "#00A087", 
    ]

    rmsf_vals_old = None

    for idx, (rmsf_vals, label) in enumerate(zip(rmsf_list, labels)):
        rmsf_vals = np.array(rmsf_vals)
        res_ids = np.arange(1, len(rmsf_vals) + 1)
        
        plt.plot(res_ids, rmsf_vals, label=label, linewidth=2, alpha=1, 
                 color=color_list[idx % len(color_list)])

        if rmsf_vals_old is not None and len(rmsf_vals) == len(rmsf_vals_old):
            delta_rmsf = rmsf_vals - rmsf_vals_old
            plt.plot(res_ids, delta_rmsf, label="ΔRMSF", linewidth=2, alpha=1, 
                     color=color_list[(idx + 1) % len(color_list)])
            print(f"RMSF ({label}):", rmsf_vals.tolist())
            print(f"RMSF (prev):", rmsf_vals_old.tolist())
            print(f"ΔRMSF:", delta_rmsf.tolist())

        rmsf_vals_old = rmsf_vals
        
    # plt.xlabel('Residue ID')
    plt.ylabel('Å')
    plt.ylim(-2, 10)
    fig.legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.1), 
               ncol=3, fontsize=font_size)
    plt.tight_layout()

    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"\n绘图完成！图片已保存为: {output_file}")
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 假设你已经从main函数获得了rmsf数据
    # rmsd_a, rmsd_b, rmsf = main(conf_a_path, conf_b_path, folder_path)
    
    # 绘制单条RMSF
    # make_rmsf_plot([rmsf], ["Sample"], output_file="rmsf.pdf")
    
    # 绘制多条RMSF并计算差值
    # make_rmsf_plot([rmsf1, rmsf2], ["Holo", "Apo"], output_file="rmsf_compare.pdf")
    pass