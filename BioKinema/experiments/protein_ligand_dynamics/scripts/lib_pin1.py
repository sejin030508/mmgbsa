# -*- coding: utf-8 -*-
# Auto-extracted VERBATIM from the manuscript notebook(s), with only the
# final figure parameters baked in (documented inline). Do not hand-edit the
# analysis logic here -- it is the exact code that produced the paper figures.
# Source: fig4c_allosteric.ipynb (cell 0)

import os
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import MMCIFParser, Superimposer
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

def get_ca_atoms_by_chain(structure):
    """
    从结构中按链提取所有 CA 原子。
    返回: OrderedDict {chain_id: [CA atoms]}
    """
    ca_by_chain = OrderedDict()
    for model in structure:
        for chain in model:
            chain_id = chain.id
            if chain_id not in ca_by_chain:
                ca_by_chain[chain_id] = []
            for i, residue in enumerate(chain):
                if 'CA' in residue:
                    ca_by_chain[chain_id].append(residue['CA'])
    return ca_by_chain


def get_ca_atoms_by_chain_with_resid(structure):
    """
    从结构中按链提取所有 CA 原子及其残基编号。
    返回: OrderedDict {chain_id: [(res_id, CA atom), ...]}
    """
    ca_by_chain = OrderedDict()
    for model in structure:
        for chain in model:
            chain_id = chain.id
            if chain_id not in ca_by_chain:
                ca_by_chain[chain_id] = []
            for residue in chain:
                if 'CA' in residue:
                    res_id = residue.id[1]  # 残基编号
                    ca_by_chain[chain_id].append((res_id, residue['CA']))
    return ca_by_chain


def _parse_cif_task(args):
    """单个文件读取任务（用于并行处理）"""
    filename, folder_path = args
    parser = MMCIFParser(QUIET=True)
    file_path = os.path.join(folder_path, filename)
    try:
        structure = parser.get_structure(filename, file_path)
        return get_ca_atoms_by_chain(structure)
    except Exception as e:
        print(f"  读取 {filename} 失败: {e}")
        return None


def _parse_cif_task_with_resid(args):
    """单个文件读取任务，带残基编号（用于并行处理）"""
    filename, folder_path = args
    parser = MMCIFParser(QUIET=True)
    file_path = os.path.join(folder_path, filename)
    try:
        structure = parser.get_structure(filename, file_path)
        return get_ca_atoms_by_chain_with_resid(structure)
    except Exception as e:
        print(f"  读取 {filename} 失败: {e}")
        return None


def calculate_rmsd_with_separate_regions(
    folder_path,
    align_residue_range,
    calc_residue_range,
    chain_id=None,
    min_residues=10
):
    """
    在指定的对齐区域上叠合结构，然后在另一个区域上计算相对于第0帧的RMSD。
    支持多个sample，对每个sample分别计算RMSD后取平均。
    
    参数:
        folder_path: str, CIF文件所在文件夹路径
        align_residue_range: tuple (start, end), 用于对齐的残基范围（包含两端）
        calc_residue_range: tuple (start, end), 用于计算RMSD的残基范围（包含两端）
        chain_id: str or None, 指定要处理的链ID，None表示处理所有蛋白质链
        min_residues: int, 链的最小残基数，少于此数的链被忽略
    
    返回:
        frame_indices: list, 帧索引列表
        rmsd_mean: np.array, 各sample的RMSD平均值
        rmsd_std: np.array, 各sample的RMSD标准差
        或 None（处理失败时）
    """
    import re
    
    folder_name = folder_path.split("/")[-2] if "/" in folder_path else folder_path
    
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹不存在 - {folder_path}")
        return None

    cif_files = [f for f in os.listdir(folder_path) if f.endswith('.cif')]
    
    if len(cif_files) < 2:
        print(f"跳过 {folder_name}: 至少需要两个 CIF 文件。")
        return None

    # 解析文件名，按sample分组
    # 文件名格式: xxxx_s{sample}_f{frame}.cif
    sample_files = {}  # {sample_id: [(frame_id, filename), ...]}
    
    pattern = re.compile(r'_s(\d+)_f(\d+)(?:_[^.]*)?\.cif$', re.IGNORECASE)
    
    for f in cif_files:
        match = pattern.search(f)
        if match:
            sample_id = int(match.group(1))
            frame_id = int(match.group(2))
            if sample_id not in sample_files:
                sample_files[sample_id] = []
            sample_files[sample_id].append((frame_id, f))
        else:
            print(f"  警告: 文件名 {f} 不符合 xxxx_s{{sample}}_f{{frame}}.cif 格式，已跳过")
    
    if not sample_files:
        print(f"跳过 {folder_name}: 没有找到符合格式的文件。")
        return None
    
    # 对每个sample内的文件按frame排序
    for sample_id in sample_files:
        sample_files[sample_id] = sorted(sample_files[sample_id], key=lambda x: x[0])
    
    print(f"正在处理文件夹: {folder_name}")
    print(f"  发现 {len(sample_files)} 个 sample: {sorted(sample_files.keys())}")
    print(f"  对齐区域: 残基 {align_residue_range[0]} - {align_residue_range[1]}")
    print(f"  计算区域: 残基 {calc_residue_range[0]} - {calc_residue_range[1]}")

    # 辅助函数：根据残基范围筛选原子
    def filter_atoms_by_range(atoms_with_resid, res_range):
        """从(res_id, atom)列表中筛选指定范围内的原子"""
        start, end = res_range
        filtered = [(rid, atom) for rid, atom in atoms_with_resid if start <= rid <= end]
        return filtered

    # 存储每个sample的RMSD结果
    all_sample_rmsd = {}  # {sample_id: [rmsd_values]}
    
    for sample_id in sorted(sample_files.keys()):
        frames = sample_files[sample_id]
        print(f"  处理 sample {sample_id} ({len(frames)} 帧)...")
        
        if len(frames) < 2:
            print(f"    跳过: 帧数少于2")
            continue
        
        # 并行读取该sample的所有结构
        task_args = [(f, folder_path) for _, f in frames]
        structures_by_chain = []
        
        with ProcessPoolExecutor() as executor:
            results = executor.map(_parse_cif_task_with_resid, task_args)
            for res in results:
                if res:
                    structures_by_chain.append(res)
                else:
                    structures_by_chain.append(None)
        
        # 检查是否有失败的读取
        valid_indices = [i for i, s in enumerate(structures_by_chain) if s is not None]
        if len(valid_indices) < 2:
            print(f"    跳过: 有效结构少于2")
            continue
        
        # 使用第一个有效结构作为参考
        ref_idx = valid_indices[0]
        ref_structure = structures_by_chain[ref_idx]
        
        # 确定要处理的链
        if chain_id is not None:
            if chain_id not in ref_structure:
                print(f"    跳过: 链 {chain_id} 不存在")
                continue
            chain_ids = [chain_id]
        else:
            chain_ids = [cid for cid, atoms in ref_structure.items() 
                        if len(atoms) >= min_residues]
        
        if not chain_ids:
            print(f"    跳过: 没有有效的蛋白质链")
            continue
        
        # 提取参考结构的对齐和计算区域原子
        ref_align_atoms = []
        ref_calc_atoms = []
        ref_align_count = {}
        ref_calc_count = {}
        
        for cid in chain_ids:
            align_atoms = filter_atoms_by_range(ref_structure[cid], align_residue_range)
            calc_atoms = filter_atoms_by_range(ref_structure[cid], calc_residue_range)
            ref_align_count[cid] = len(align_atoms)
            ref_calc_count[cid] = len(calc_atoms)
            ref_align_atoms.extend([atom for _, atom in align_atoms])
            ref_calc_atoms.extend([atom for _, atom in calc_atoms])
        
        if len(ref_align_atoms) == 0:
            print(f"    跳过: 对齐区域没有原子")
            continue
        if len(ref_calc_atoms) == 0:
            print(f"    跳过: 计算区域没有原子")
            continue
        
        ref_calc_coords = np.array([atom.get_coord() for atom in ref_calc_atoms])
        
        # 计算每帧的RMSD
        sup = Superimposer()
        rmsd_values = []
        valid_frame_indices = []
        
        for i, struct in enumerate(structures_by_chain):
            if struct is None:
                continue
            
            # 检查该结构是否与参考结构兼容
            valid = True
            moving_align_atoms = []
            moving_calc_atoms = []
            
            for cid in chain_ids:
                if cid not in struct:
                    valid = False
                    break
                align_atoms = filter_atoms_by_range(struct[cid], align_residue_range)
                calc_atoms = filter_atoms_by_range(struct[cid], calc_residue_range)
                if len(align_atoms) != ref_align_count[cid] or len(calc_atoms) != ref_calc_count[cid]:
                    valid = False
                    break
                moving_align_atoms.extend([atom for _, atom in align_atoms])
                moving_calc_atoms.extend([atom for _, atom in calc_atoms])
            
            if not valid:
                continue
            
            if i == ref_idx:
                # 参考帧，RMSD为0
                rmsd_values.append(0.0)
            else:
                # 基于对齐区域进行叠合
                sup.set_atoms(ref_align_atoms, moving_align_atoms)
                sup.apply(moving_calc_atoms)
                
                # 计算RMSD
                moving_calc_coords = np.array([atom.get_coord() for atom in moving_calc_atoms])
                diff = moving_calc_coords - ref_calc_coords
                rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
                rmsd_values.append(rmsd)
            
            valid_frame_indices.append(frames[i][0])  # 使用原始frame_id
        
        if len(rmsd_values) >= 2:
            all_sample_rmsd[sample_id] = {
                'frame_indices': valid_frame_indices,
                'rmsd': np.array(rmsd_values)
            }
            print(f"    完成: {len(rmsd_values)} 帧, RMSD范围: {min(rmsd_values):.3f} - {max(rmsd_values):.3f} Å")
    
    if not all_sample_rmsd:
        print(f"跳过 {folder_name}: 没有有效的sample数据。")
        return None
    
    # 对齐所有sample的帧索引并计算平均RMSD
    # 找到所有sample共有的帧索引
    all_frame_sets = [set(data['frame_indices']) for data in all_sample_rmsd.values()]
    common_frames = sorted(set.intersection(*all_frame_sets))
    
    if len(common_frames) < 2:
        # 如果没有足够的共同帧，使用最长的sample或者按位置对齐
        print(f"  警告: 各sample的帧不完全一致，按帧位置对齐取平均")
        
        # 找到最小的帧数
        min_frames = min(len(data['rmsd']) for data in all_sample_rmsd.values())
        
        # 截取每个sample的前min_frames帧
        rmsd_matrix = []
        for sample_id, data in all_sample_rmsd.items():
            rmsd_matrix.append(data['rmsd'][:min_frames])
        
        rmsd_matrix = np.array(rmsd_matrix)
        rmsd_mean = np.mean(rmsd_matrix, axis=0)
        rmsd_std = np.std(rmsd_matrix, axis=0)
        frame_indices = list(range(min_frames))
    else:
        # 使用共同的帧索引
        rmsd_matrix = []
        for sample_id, data in all_sample_rmsd.items():
            frame_to_rmsd = dict(zip(data['frame_indices'], data['rmsd']))
            sample_rmsd = [frame_to_rmsd[f] for f in common_frames]
            rmsd_matrix.append(sample_rmsd)
        
        rmsd_matrix = np.array(rmsd_matrix)
        rmsd_mean = np.mean(rmsd_matrix, axis=0)
        rmsd_std = np.std(rmsd_matrix, axis=0)
        frame_indices = common_frames
    
    print(f"  汇总: {len(all_sample_rmsd)} 个sample, {len(frame_indices)} 帧")
    print(f"  平均RMSD范围: {rmsd_mean.min():.3f} - {rmsd_mean.max():.3f} Å")
    print(f"  平均RMSD均值: {rmsd_mean.mean():.3f} ± {rmsd_std.mean():.3f} Å")
    
    return frame_indices, rmsd_mean, rmsd_std


def calculate_rmsd_data(
    folder_list,
    align_residue_range,
    calc_residue_range,
    chain_id=None,
    min_residues=10
):
    """
    批量处理多个文件夹，计算RMSD数据。
    
    参数:
        folder_list: list, 文件夹路径列表
        align_residue_range: tuple, 对齐区域残基范围
        calc_residue_range: tuple, 计算区域残基范围
        chain_id: str or None, 指定链ID
        min_residues: int, 最小残基数
    
    返回:
        data_list: list of (folder_label, frame_indices, rmsd_mean, rmsd_std)
    """
    data_list = []
    
    for folder in folder_list:
        result = calculate_rmsd_with_separate_regions(
            folder,
            align_residue_range,
            calc_residue_range,
            chain_id=chain_id,
            min_residues=min_residues
        )
        
        if result:
            frame_indices, rmsd_mean, rmsd_std = result
            folder_label = folder.split("/")[-2] if "/" in folder else folder
            data_list.append((folder_label, frame_indices, rmsd_mean, rmsd_std))
        else:
            print(f"警告: 无法获取 {folder} 的RMSD数据。")
    
    return data_list


def plot_rmsd(
    data_list,
    output_file="./plot/figures_main/rmsd_separate_regions.pdf",
    xlabel="Frame",
    ylabel="RMSD (Å)",
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
        folder_label, frame_indices, rmsd_mean, rmsd_std = item
        color = color_list[idx % len(color_list)]
        
        # 绘制均值曲线
        ax.plot(frame_indices, rmsd_mean, label=folder_label, linewidth=2, alpha=1, color=color)
        
        # 绘制标准差阴影
        if show_std and rmsd_std is not None:
            ax.fill_between(
                frame_indices,
                rmsd_mean - rmsd_std,
                rmsd_mean + rmsd_std,
                alpha=0.2,
                color=color
            )
    
    # ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    plt.xlim(-2, 152)
    # x ticks relabeled frames -> us (151 frames @10ns = 1.5us), no "us" xlabel
    plt.xticks([0, 30, 60, 90, 120, 150], ["0.0", "0.3", "0.6", "0.9", "1.2", "1.5"])
    plt.ylim(0, 6)
    plt.yticks([0, 3, 6], [" 0", " 3", " 6"])
    plt.tight_layout()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"\n绘图完成！图片已保存为: {output_file}")
    plt.show()


# ============== 原有函数保持不变 ==============

def calculate_rmsf_data(folder_path, min_residues=10):
    """
    处理单个文件夹：读取 CIF -> 按链对齐结构 -> 计算 RMSF。
    """
    folder_name = folder_path.split("/")[-2]
    
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹不存在 - {folder_path}")
        return None

    cif_files = [f for f in os.listdir(folder_path) if f.endswith('.cif')]
    
    try:
        cif_files = sorted(cif_files, key=lambda x: int(x.split('_')[-2][1:]))
        cif_files = cif_files[:]
    except (ValueError, IndexError):
        cif_files = sorted(cif_files)

    if len(cif_files) < 2:
        print(f"跳过 {folder_name}: 至少需要两个 CIF 文件才能计算 RMSF。")
        return None

    print(f"正在处理文件夹: {folder_name} ({len(cif_files)} 个文件)...")

    all_structures_by_chain = []
    task_args = [(f, folder_path) for f in cif_files]
    
    with ProcessPoolExecutor() as executor:
        results = executor.map(_parse_cif_task, task_args)
        for res in results:
            if res:
                all_structures_by_chain.append(res)

    if not all_structures_by_chain:
        return None

    ref_chains = all_structures_by_chain[0]
    chain_ids = []
    for cid, atoms in ref_chains.items():
        if len(atoms) >= min_residues:
            chain_ids.append(cid)
        else:
            print(f"  忽略链 {cid}: 只有 {len(atoms)} 个残基（可能是小分子）")
    
    if not chain_ids:
        print(f"跳过 {folder_name}: 没有包含足够 CA 原子的蛋白质链。")
        return None
    
    print(f"  蛋白质链: {chain_ids}, 每链残基数: {[len(ref_chains[cid]) for cid in chain_ids]}")

    valid_structures = []
    for struct_by_chain in all_structures_by_chain:
        valid = True
        for chain_id in chain_ids:
            if chain_id not in struct_by_chain:
                valid = False
                break
            if len(struct_by_chain[chain_id]) != len(ref_chains[chain_id]):
                valid = False
                break
        if valid:
            valid_structures.append(struct_by_chain)
    
    if len(valid_structures) < 2:
        print(f"跳过 {folder_name}: 有效结构少于 2 个。")
        return None

    print(f"  有效结构数: {len(valid_structures)}")

    sup = Superimposer()
    all_chain_aligned_coords = OrderedDict()
    
    for chain_id in chain_ids:
        n_residues = len(ref_chains[chain_id])
        n_structs = len(valid_structures)
        aligned_coords = np.zeros((n_structs, n_residues, 3))
        
        reference_atoms = valid_structures[0][chain_id]
        for j, atom in enumerate(reference_atoms):
            aligned_coords[0, j, :] = atom.get_coord()
        
        for i in range(1, n_structs):
            moving_atoms = valid_structures[i][chain_id]
            sup.set_atoms(reference_atoms, moving_atoms)
            sup.apply(moving_atoms)
            for j, atom in enumerate(moving_atoms):
                aligned_coords[i, j, :] = atom.get_coord()
        
        all_chain_aligned_coords[chain_id] = aligned_coords

    combined_coords = np.concatenate(
        [all_chain_aligned_coords[cid] for cid in chain_ids], 
        axis=1
    )
    
    mean_coords = np.mean(combined_coords, axis=0)
    sq_diff = (combined_coords - mean_coords) ** 2
    msf = np.mean(np.sum(sq_diff, axis=2), axis=0)
    rmsf = np.sqrt(msf)

    residue_ids = []
    offset = 0
    prev_max_id = 0
    
    for chain_id in chain_ids:
        reference_atoms = valid_structures[0][chain_id]
        for atom in reference_atoms:
            res_num = atom.get_parent().id[1]
            new_id = res_num + offset
            residue_ids.append(new_id)
            prev_max_id = max(prev_max_id, new_id)
        offset = prev_max_id

    print(f"  总残基数: {len(residue_ids)}, RMSF 长度: {len(rmsf)}")
    
    return residue_ids, rmsf


def read_data(folder_list, min_residues=10):
    """
    读取多个文件夹的 RMSF 数据。
    """
    data_list = []
    
    for folder in folder_list:
        result = calculate_rmsf_data(folder, min_residues=min_residues)
        
        if result:
            res_ids, rmsf_vals = result
            folder_label = folder.split("/")[-2]
            
            if "8STG" in folder_label:
                res_ids = [x - 3 for x in res_ids]
            print(f"  残基 ID 范围: {min(res_ids)} - {max(res_ids)}")
            
            data_list.append((folder_label, res_ids, rmsf_vals))
        else:
            print(f"警告: 无法获取 {folder} 的数据。")
    
    return data_list


def make_plot(data_list, output_file="./figures_main/fig4_rmsf.pdf"):
    """
    根据数据列表绘制 RMSF 图。
    """
    import matplotlib.font_manager as fm
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

    fig, ax = plt.subplots(figsize=(5, 1.7))
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
    
    if not data_list:
        print("没有有效的数据可供绘图。")
        return

    rmsf_vals_old = None
    label_dict = {
        "FGFR_S_R1_0_0": "FGFR1_cpd6",
        "FGFR_U_R1_0_0": "FGFR1_apo",
        "FGFR_S_R2_0_0": "FGFR2_cpd6",
        "FGFR_U_R2_0_0": "FGFR2_apo",
        "3TDB_S_R1_0_0": "Holo",
        "3TDB_U_R1_0_0": "Apo",
    }
    color_list = [
        "#356a9f", "#ac3d48","#00A087", 
    ]
    idx = 0

    for folder_label, res_ids, rmsf_vals in data_list:
        plt.plot(res_ids, rmsf_vals, label=label_dict.get(folder_label, folder_label), linewidth=2, alpha=1, color=color_list[idx])
        idx += 1

        if rmsf_vals_old is not None:
            plt.plot(res_ids, rmsf_vals - rmsf_vals_old, label="ΔRMSF", linewidth=2, alpha=1, color=color_list[idx])
            idx += 1
            print(rmsf_vals.tolist())
            print(rmsf_vals_old.tolist())
            print((rmsf_vals-rmsf_vals_old).tolist())

        rmsf_vals_old = rmsf_vals
        
    plt.ylabel('Å')
    plt.ylim(-2, 10)
    fig.legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize=font_size)
    plt.tight_layout()

    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"\n绘图完成！图片已保存为: {output_file}")
    plt.show()


def plot_multiple_folders(folder_list, min_residues=10):
    """
    接收文件夹路径列表，计算每个文件夹的 RMSF 并绘制在同一张图上。
    """
    data_list = read_data(folder_list, min_residues=min_residues)
    make_plot(data_list)
