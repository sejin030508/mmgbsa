# -*- coding: utf-8 -*-
# Auto-extracted VERBATIM from the manuscript notebook(s), with only the
# final figure parameters baked in (documented inline). Do not hand-edit the
# analysis logic here -- it is the exact code that produced the paper figures.
# Source: fig2a_physical_stability.ipynb (cell 0)

import re
import gemmi
import numpy as np
import pandas as pd
import os
import glob
from plot_style import font_size, get_optimal_ticks  # shared style (font_size, nice tick helper)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import rdDetermineBonds
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('__file__')), '..', 'shared'))
from plot_style import *

# ==========================================
# 1. 核心计算函数 (Gemmi + RDKit) - 保持不变
# ==========================================

def get_ligand_data_with_gemmi(file_path, ligand_name="LIG"):
    """使用 Gemmi 读取 CIF/PDB，提取指定配体的原子序数和坐标。"""
    try:
        st = gemmi.read_structure(file_path)
        atomic_nums = []
        coords = []
        for model in st:
            for chain in model:
                for residue in chain:
                    if residue.name == ligand_name:
                        for atom in residue:
                            # 过滤氢原子
                            if atom.element.name == "H": continue
                            atomic_nums.append(atom.element.atomic_number)
                            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
        if not coords: return None, None
        return atomic_nums, np.array(coords)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

def create_mol_from_data(atomic_nums, coords):
    """创建一个没有键连接的 RDKit RWMol 对象，并赋予坐标。"""
    mol = Chem.RWMol()
    conf = Chem.Conformer(len(atomic_nums))
    for idx, (z, pos) in enumerate(zip(atomic_nums, coords)):
        atom = Chem.Atom(int(z))
        mol.AddAtom(atom)
        from rdkit.Geometry import Point3D
        conf.SetAtomPosition(idx, Point3D(float(pos[0]), float(pos[1]), float(pos[2])))
    mol.AddConformer(conf)
    return mol


def get_gt_mol(pdb_id):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    pdb_id = pdb_id.lower()
    sdf_path = Path(f"/cto_studio/xtalpi_lab/Datasets/pdbbind/PDBbind_v2020/v2020-other-PL/{pdb_id}/{pdb_id}_ligand.sdf")
    if not sdf_path.exists():
        sdf_path = Path(f"/cto_studio/xtalpi_lab/Datasets/pdbbind/PDBbind_v2020/refined-set/{pdb_id}/{pdb_id}_ligand.sdf")
    
    # 读取SDF文件(这个文件本身就包含完整的键信息和3D坐标)
    try:
        mol_sdf = Chem.SDMolSupplier(str(sdf_path), sanitize=False, strictParsing=False)[0]
        mol_sdf = Chem.RemoveHs(mol_sdf)
    except:
        mol2_path = Path(f"/cto_studio/xtalpi_lab/Datasets/pdbbind/PDBbind_v2020/v2020-other-PL/{pdb_id}/{pdb_id}_ligand.mol2")
        if not mol2_path.exists():
            mol2_path = Path(f"/cto_studio/xtalpi_lab/Datasets/pdbbind/PDBbind_v2020/refined-set/{pdb_id}/{pdb_id}_ligand.mol2")
        mol_sdf = Chem.MolFromMol2File(str(mol2_path), sanitize=False)
        mol_sdf = Chem.RemoveHs(mol_sdf)
    return mol_sdf

def process_single_system(system_dir, ligand_name="LIG"):
    """
    处理单个体系文件夹：
    1. 找到 predictions 文件夹
    2. 排序 CIF 文件 (第1个是 GT)
    3. 计算所有误差
    返回: 包含误差数据的 list of dicts
    """
    pred_dir = os.path.join(system_dir, "predictions")
    if not os.path.exists(pred_dir):
        return []

    # 获取所有 cif 文件并排序
    cif_files = sorted(glob.glob(os.path.join(pred_dir, "*.cif")),
                       key=lambda p: int(re.search(r"_f(\\d+)_", p).group(1)) if re.search(r"_f(\\d+)_", p) else 0)  # numeric frame sort
    if len(cif_files) < 2:
        return [] # 至少需要 GT 和 1 个预测文件

    gt_file = cif_files[0]
    pred_files = cif_files[1:]
    system_name = os.path.basename(system_dir)
    
    # --- 1. 处理 GT ---
    pdbid = os.path.basename(gt_file).split(".")[0]
    gt_mol = get_gt_mol(pdbid.split("_")[0])
    
    
    # 预计算 GT 的键长和键角
    bond_indices = []
    gt_bond_values = []
    for bond in gt_mol.GetBonds():
        idx_a, idx_b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_indices.append((idx_a, idx_b))
        gt_bond_values.append(rdMolTransforms.GetBondLength(gt_mol.GetConformer(), idx_a, idx_b))

    angle_indices = []
    gt_angle_values = []
    for atom in gt_mol.GetAtoms():
        center_idx = atom.GetIdx()
        neighbors = [x.GetIdx() for x in atom.GetNeighbors()]
        if len(neighbors) < 2: continue
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                idx_a, idx_c = neighbors[i], neighbors[j]
                angle_indices.append((idx_a, center_idx, idx_c))
                # RDKit 返回度数 (degrees)
                angle_deg = rdMolTransforms.GetAngleDeg(gt_mol.GetConformer(), idx_a, center_idx, idx_c)
                gt_angle_values.append(np.radians(angle_deg))

    results = []

    # --- 2. 遍历预测文件 ---
    for frame_idx, fpath in enumerate(pred_files):
        # frame_idx = 0 对应第一个预测文件
        _, pred_coords = get_ligand_data_with_gemmi(fpath, ligand_name)
        
        if pred_coords is None or len(pred_coords) != gt_mol.GetNumAtoms():
            continue

        # 克隆拓扑，更新坐标
        pred_mol = Chem.Mol(gt_mol)
        conf = pred_mol.GetConformer()
        from rdkit.Geometry import Point3D
        for i, pos in enumerate(pred_coords):
            conf.SetAtomPosition(i, Point3D(float(pos[0]), float(pos[1]), float(pos[2])))

        # 计算键长误差 (Å)
        errs = []
        for k, (u, v) in enumerate(bond_indices):
            pred_len = rdMolTransforms.GetBondLength(conf, u, v)
            err = abs(pred_len - gt_bond_values[k])
            errs.append(err)
        
        if errs:
            results.append({
                "System": system_name,
                "Frame Index": frame_idx,
                "Error Type": "Bond Length Error-Å",
                "Error Value": np.mean(errs)
            })

        # 计算键角误差 (Radians)
        errs = []
        for k, (a, center, c) in enumerate(angle_indices):
            pred_angle_deg = rdMolTransforms.GetAngleDeg(conf, a, center, c)
            pred_angle_rad = np.radians(pred_angle_deg)
            err = abs(pred_angle_rad - gt_angle_values[k])
            errs.append(err)
        
        if errs:
            results.append({
                "System": system_name,
                "Frame Index": frame_idx,
                "Error Type": "Bond Angle Error-radians",
                "Error Value": np.mean(errs)
            })
            
    return results


def plot_metric(df, metric_name, filename_suffix, order, palette, output_dir, md_value=None, y_min=None, y_max=None, title=None, ylabel=None):
    # 筛选数据
    subset_df = df[df['Error Type'] == metric_name]
    
    if subset_df.empty:
        print(f"警告: {metric_name} 没有数据，跳过绘图。")
        return

    plt.figure(figsize=(5, 2.2)) # 单张图的大小
    
    # 绘图
    ax = sns.violinplot(
        data=subset_df,
        x='Time Segment',
        y='Error Value',
        order=order,
        palette=palette, # 使用显式传递的颜色列表
        linewidth=0.8, # 小提琴轮廓线宽
        linecolor='black' # 小提琴轮廓颜色黑色
    )
    
    # 设置标签
    ax.set_xlabel("")
    if ylabel is None:
        ax.set_ylabel(metric_name.split("-")[-1].strip(), color='black')
    else:
        ax.set_ylabel(ylabel, color='black')
    if title is None:
        title = metric_name.split("-")[0].strip()
    ax.set_title(title, fontsize=font_size, pad=10, color='black')
    
    y_max_data = max(subset_df['Error Value'])
    y_min_data = min(subset_df['Error Value'])
    print(y_max_data, y_min_data)
    # ticks follow the requested y-range (y_min/y_max) when given, so an explicit
    # ylim (e.g. 0-0.2) gets sensible ticks instead of ticks spanning the raw outliers.
    ticks = get_optimal_ticks(y_max if y_max is not None else y_max_data,
                              y_min if y_min is not None else y_min_data)
    print(ticks)
    top_tick = ticks[-1] if y_max is None else y_max
    bottom_tick = ticks[0] if y_min is None else y_min
    
    # 2. 强制设置刻度和范围
    ax.set_yticks(ticks)
    ax.set_ylim(bottom_tick, top_tick)

    # 添加红色虚线表示 md_value
    if md_value is not None:
        ax.axhline(y=md_value, color='red', linestyle='--', linewidth=1.0, label=f'MD: {md_value:.2f}')
    
    # 3. 确保坐标轴颜色为黑色
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    # 删除所有 X 轴刻度和标签 (但保留轴线颜色为黑)
    ax.set_xticks([]) 
    
    # 强制去除网格
    ax.grid(False)
    
    # 去除顶部和右侧的边框
    sns.despine(left=False, bottom=False, right=True, top=True)

    plt.tight_layout()
    
    # 保存
    save_name = f'misato_{filename_suffix}.pdf'
        
    save_path = os.path.join(output_dir, save_name)
    print(f"保存图像至: {save_path}")
    plt.savefig(save_path)
    plt.show()
    plt.close()
