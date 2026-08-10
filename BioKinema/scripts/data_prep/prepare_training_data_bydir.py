# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare processed training data from mmCIF files. Modified from Protenix's data pipeline. """

import argparse
import csv
from pathlib import Path
from typing import Optional, List, Dict

import mdtraj, os, tempfile

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from protenix.data.data_pipeline import DataPipeline
from protenix.utils.file_io import dump_gzip_pickle
from protenix.utils.file_io import dump_gzip_pickle, load_gzip_pickle
import json


def load_annotation_mapping(input_csv: Path) -> Dict:
    """
    从CSV文件中加载注释映射。

    Args:
        input_csv (Path): 输入的CSV文件路径。

    Returns:
        Dict: 一个映射，键是文件的绝对路径，值是包含注释的字典。
    """
    df = pd.read_csv(input_csv)
    mapping = {}
    for i, row in df.iterrows():
        # 确保键是绝对路径，以便进行一致的匹配
        mapping[os.path.abspath(row["name"])] = {
            k: float(v) for k, v in row.items() if k != "name"
        }
    return mapping


def gen_a_bioassembly_data(
    mmcif: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    dataset: str = "Atlas",
    annotation_mapping: Optional[dict] = None,
) -> Optional[List[Dict]]:
    """
    从单个mmCIF文件生成生物组装数据，并将其保存到指定的输出目录。

    Args:
        mmcif (Path): mmCIF文件的路径。
        bioassembly_output_dir (Path): 保存生物组装数据的目录。
        cluster_file (Optional[Path]): 聚类文件的路径（如果可用）。
        dataset (str, optional): 数据集名称，默认为 "Atlas"。
        annotation_mapping (Optional[dict], optional): 用于 'OpenMM' 数据集的注释映射。默认为 None。

    Returns:
        Optional[List[Dict]]: 如果数据成功生成，则返回样本索引列表；否则返回 None。
    """
    kwargs = {}
    # 如果数据集是OpenMM，则从注释映射中加载额外参数
    if dataset == "OpenMM":
        if annotation_mapping is None:
            raise ValueError("Annotation mapping is required for OpenMM dataset but was not provided.")
        # 使用文件的绝对路径作为键来查找注释
        abs_mmcif_path = str(mmcif.resolve())
        if abs_mmcif_path in annotation_mapping:
            kwargs = annotation_mapping[abs_mmcif_path]
        else:
            print(f"Warning: No annotation found for {mmcif} in the provided mapping file.")
            return None

    sample_indices_list, bioassembly_dict = DataPipeline.get_data_from_mmcif(
        mmcif, cluster_file, dataset, **kwargs
    )

    if sample_indices_list and bioassembly_dict:
        pdb_id = bioassembly_dict["pdb_id"]
        if not pdb_id:
            pdb_id = mmcif.stem.replace('.cif', '') # 清理文件名以获得ID
        # 保存处理后的数据到pkl.gz文件
        bioassembly_output_dir.mkdir(parents=True, exist_ok=True)
        dump_gzip_pickle(bioassembly_dict, bioassembly_output_dir / f"{pdb_id}.pkl.gz")
        return sample_indices_list
    else:
        print(f"Failed to generate data for {mmcif}")
        return None

def process_single_folder(
    folder_path: Path,
    output_csv_dir: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    dataset: str,
    annotation_mapping: Optional[Dict] = None,
):
    """
    处理单个文件夹内的所有mmCIF文件，并将结果保存到一个单独的CSV文件中。
    此函数设计为在并行进程中运行。

    Args:
        folder_path (Path): 要处理的子文件夹的路径。
        output_csv_dir (Path): 用于存放生成的CSV文件的目录。
        bioassembly_output_dir (Path): 用于存放所有生物组装输出（.pkl.gz文件）的目录。
        cluster_file (Optional[Path]): 聚类文件的路径。
        dataset (str): 数据集名称。
        annotation_mapping (Optional[Dict], optional): 全局注释映射。默认为 None。
    """
    # 为此文件夹定义输出CSV文件的路径
    output_csv_path = output_csv_dir / f"{folder_path.name}.csv"
    if output_csv_path.exists():
        print(f"✓ Folder {folder_path.name} already exists. Results saved to {output_csv_path}")
        return
    
    # 查找此文件夹中的所有mmCIF文件（包括.cif和.cif.gz）
    mmcif_list = list(folder_path.glob("*.cif")) + list(folder_path.glob("*.cif.gz"))
    
    if not mmcif_list:
        temp_path = folder_path / "temp_fixed"
        mmcif_list = list(temp_path.glob("*.cif")) + list(temp_path.glob("*.cif.gz"))

    if not mmcif_list:
        print(f"No mmCIF files found in {folder_path} and temp_fixed, skipping.")
        return

    folder_results = []
    # 按照要求，在文件夹内部串行处理文件
    print(f"Processing {len(mmcif_list)} files in folder {folder_path.name}...")
    for mmcif in mmcif_list:
        try:
            sample_indices_list = gen_a_bioassembly_data(
                mmcif,
                bioassembly_output_dir / folder_path.name,
                cluster_file,
                dataset,
                annotation_mapping
            )
            if sample_indices_list:
                folder_results.extend(sample_indices_list)
        except Exception as e:
            #if "mol_id" not in str(e):
            #raise e
            print(f"Catch error when processing  {folder_path.name}. error: {e}, skipping")
            #raise e
            return

    # 将此文件夹的所有结果写入其专属的CSV文件
    if folder_results:
        df = pd.DataFrame(folder_results)
        df.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"✓ Finished processing folder {folder_path.name}. Results saved to {output_csv_path}")
    else:
        print(f"No data could be generated for any file in folder {folder_path.name}.")


def _gen_data_from_mmcifs_list(
    mmcif_list: List[Path],
    output_indices_csv: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    dataset: str = "Atlas",
    num_workers: int = 1,
    annotation_mapping: Optional[Dict] = None,
):
    """
    （内部函数）从一个明确的mmCIF文件列表生成训练数据（原始的并行逻辑）。
    用于处理.txt或.csv文件输入的情况。
    """
    all_sample_indices_list = [
        r
        for r in tqdm(
            Parallel(n_jobs=num_workers, return_as="generator_unordered")(
                delayed(gen_a_bioassembly_data)(
                    mmcif, bioassembly_output_dir, cluster_file, dataset, annotation_mapping
                )
                for mmcif in mmcif_list
            ),
            total=len(mmcif_list),
            desc="Processing files in parallel"
        )
    ]

    merged_results = []
    for sample_indices_list in all_sample_indices_list:
        if sample_indices_list:
            merged_results.extend(sample_indices_list)
    
    if merged_results:
        df = pd.DataFrame(merged_results)
        df.to_csv(output_indices_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"File-based processing complete. Results saved to {output_indices_csv}")
    else:
        print("File-based processing complete. No data was generated.")


def run_gen_data(
    input_path: Path,
    output_csv_dir: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    dataset: str = "Atlas",
    num_workers: int = 1,
    error_file: Optional[Path] = None,
):
    """
    从mmCIF文件生成数据并保存输出。
    此函数现在可以处理一个包含许多子文件夹的目录。

    Args:
        input_path (Path): 输入路径。可以是包含多个子文件夹的目录，也可以是列出mmCIF文件路径的.txt或.csv文件。
        output_csv_dir (Path): 用于保存输出CSV文件的目录。
        bioassembly_output_dir (Path): 用于保存生物组装输出的目录。
        cluster_file (Optional[Path]): 聚类文件的路径。
        dataset (str, optional): 数据集名称。默认为 "Atlas"。
        num_workers (int, optional): 使用的工作进程数。默认为 1。
    """
    input_path = Path(input_path)
    bioassembly_output_dir = Path(bioassembly_output_dir)
    output_csv_dir = Path(output_csv_dir)

    # 创建输出目录
    output_csv_dir.mkdir(parents=True, exist_ok=True)
    bioassembly_output_dir.mkdir(parents=True, exist_ok=True)

    annotation_mapping = None

    if input_path.is_dir():
        # 新逻辑：处理一个包含多个子文件夹的目录
        # 获取所有子文件夹的列表
        folder_list = [d for d in input_path.iterdir() if d.is_dir()]
        
        if not folder_list:
            # 如果目录中没有子文件夹，则作为单个文件夹处理（向后兼容）
            print(f"No subdirectories found in {input_path}. Processing it as a single folder.")
            process_single_folder(
                input_path,
                output_csv_dir,
                bioassembly_output_dir,
                cluster_file,
                dataset,
                None
            )
            return

        print(f"Found {len(folder_list)} subdirectories to process in parallel using {num_workers} workers.")
        # 读取错误文件
        error_datas = []
        if error_file is not None:
            with open(error_file, "r") as f:
                error_datas = json.load(f)
        folder_list = [d for d in folder_list if d.name not in set(error_datas)]
        print(f"Found {len(error_datas)} errors in {error_file}, after filter, {len(folder_list)} subdirectories to process in parallel using {num_workers} workers.")

        # 并行处理每个文件夹
        Parallel(n_jobs=num_workers)(
            delayed(process_single_folder)(
                folder_path,
                output_csv_dir,
                bioassembly_output_dir,
                cluster_file,
                dataset,
                annotation_mapping,  # 传递全局注释（如果未来需要）
            )
            for folder_path in tqdm(folder_list, desc="Processing folders")
        )
        print("\nAll folders have been processed.")

    elif input_path.suffix == ".csv":
        # 保留原有逻辑：处理CSV注释文件
        print("Processing a .csv annotation file. Parallelization will be per-file.")
        annotation_mapping = load_annotation_mapping(input_path)
        mmcif_list = [Path(p) for p in annotation_mapping.keys()]
        # 为此情况生成一个统一的CSV输出文件
        output_csv_file = output_csv_dir / f"{input_path.stem}_output.csv"
        _gen_data_from_mmcifs_list(
            mmcif_list,
            output_csv_file,
            bioassembly_output_dir,
            cluster_file,
            dataset,
            num_workers,
            annotation_mapping=annotation_mapping,
        )

    elif input_path.suffix == ".txt":
        # 保留原有逻辑：处理TXT文件列表
        print("Processing a .txt file list. Parallelization will be per-file.")
        with open(input_path) as f:
            mmcif_list = [Path(i.strip()) for i in f.readlines()]
        # 为此情况生成一个统一的CSV输出文件
        output_csv_file = output_csv_dir / f"{input_path.stem}_output.csv"
        _gen_data_from_mmcifs_list(
            mmcif_list,
            output_csv_file,
            bioassembly_output_dir,
            cluster_file,
            dataset,
            num_workers,
        )
    else:
        raise NotImplementedError(f"Unsupported input path: {input_path}. Must be a directory containing subfolders, a .txt file, or a .csv file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process mmCIF files from a directory of subfolders into bioassembly data.")
    parser.add_argument(
        "-i",
        "--input_path",
        type=Path,
        required=True,
        help="Path to the parent directory containing subfolders of mmCIF files, or a .txt/.csv file listing mmCIF file paths.",
    )
    parser.add_argument(
        "-o",
        "--output_csv_dir", # 参数已重命名
        type=Path,
        required=True,
        help="Path to the output directory where CSV files will be saved. For folder processing, one CSV per subfolder will be created here.",
    )
    parser.add_argument(
        "-b",
        "--bio_output_dir",
        type=Path,
        required=True,
        help="Directory where bioassembly outputs (.pkl.gz files) will be saved.",
    )
    parser.add_argument(
        "-c",
        "--cluster_file",
        type=Path,
        default=None,
        help="Path to the cluster txt file, if any.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="Atlas",
        choices=["Atlas", "OpenMM", "WeightedPDB", "MISATO", "MDposit"],
        help="Dataset name (e.g., Atlas, OpenMM).",
    )
    parser.add_argument(
        "-n",
        "--n_cpu",
        type=int,
        default=1,
        help="Number of worker processes to use for parallel processing of folders. Defaults to 1.",
    )
    parser.add_argument(
        "-e",
        "--error_file",
        type=Path,
        default=None,
        help="Path to the error json file, if any.",
    )

    args = parser.parse_args()

    # 确保必需的参数已提供
    if not args.input_path or not args.output_csv_dir or not args.bio_output_dir:
        parser.error("The following arguments are required: -i/--input_path, -o/--output_csv_dir, -b/--bio_output_dir")

    run_gen_data(
        input_path=args.input_path,
        output_csv_dir=args.output_csv_dir, # 使用更新后的参数名
        bioassembly_output_dir=args.bio_output_dir,
        cluster_file=args.cluster_file,
        dataset=args.dataset,
        num_workers=args.n_cpu,
        error_file=args.error_file,
    )
