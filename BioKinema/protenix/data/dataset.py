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

import json
import os
import pickle
import random
import traceback
from copy import deepcopy
import uuid
from pathlib import Path
from typing import Any, Callable, Optional, Union
import glob
import math
import numpy as np
import pandas as pd
import torch
from biotite.structure.atoms import AtomArray
from ml_collections.config_dict import ConfigDict
from torch.utils.data import Dataset
from tqdm import tqdm
import csv
import tempfile
from protenix.utils.file_io import dump_gzip_pickle

from protenix.data.constants import EvaluationChainInterface
from protenix.data.data_pipeline import DataPipeline
from protenix.data.featurizer import Featurizer
from protenix.data.msa_featurizer import MSAFeaturizer
from protenix.data.tokenizer import TokenArray
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.utils.cropping import CropData
from protenix.utils.file_io import read_indices_csv
from protenix.utils.logger import get_logger
from protenix.utils.torch_utils import dict_to_tensor
from protenix.utils.geometry import angle_3p, random_transform

logger = get_logger(__name__)


class BaseSingleDataset(Dataset):
    """
    dataset for a single data source
    data = self.__item__(idx)
    return a dict of features and labels, the keys and the shape are defined in protenix.data.utils
    """

    def __init__(
        self,
        mmcif_dir: Union[str, Path],
        bioassembly_dict_dir: Optional[Union[str, Path]],
        indices_fpath: Union[str, Path],
        cropping_configs: dict[str, Any],
        msa_featurizer: Optional[MSAFeaturizer] = None,
        template_featurizer: Optional[Any] = None,
        name: str = None,
        downsample: int = -1,
        **kwargs,
    ) -> None:
        super(BaseSingleDataset, self).__init__()

        # Configs
        self.mmcif_dir = mmcif_dir
        self.bioassembly_dict_dir = bioassembly_dict_dir
        self.indices_fpath = indices_fpath
        self.cropping_configs = cropping_configs
        self.name = name
        self.downsample = downsample
        # General dataset configs
        self.ref_pos_augment = kwargs.get("ref_pos_augment", True)
        self.lig_atom_rename = kwargs.get("lig_atom_rename", False)
        self.reassign_continuous_chain_ids = kwargs.get(
            "reassign_continuous_chain_ids", False
        )
        self.shuffle_mols = kwargs.get("shuffle_mols", False)
        self.shuffle_sym_ids = kwargs.get("shuffle_sym_ids", False)

        # Typically used for test sets
        self.find_pocket = kwargs.get("find_pocket", False)
        self.find_all_pockets = kwargs.get("find_all_pockets", False)  # for dev
        self.find_eval_chain_interface = kwargs.get("find_eval_chain_interface", False)
        self.group_by_pdb_id = kwargs.get("group_by_pdb_id", False)  # for test set
        self.sort_by_n_token = kwargs.get("sort_by_n_token", False)

        # Typically used for training set
        self.random_sample_if_failed = kwargs.get("random_sample_if_failed", False)
        self.use_reference_chains_only = kwargs.get("use_reference_chains_only", False)
        self.is_distillation = kwargs.get("is_distillation", False)
        self.dump_embeddings = kwargs.get("dump_embeddings", False)
        # Trajectory-generation training only. The MSM paired-pretrain / single-frame
        # pretrain / pseudo-trajectory modes were removed from this release; these flags
        # are kept (permanently off) so the trajectory code paths stay unchanged.
        self.single_frame_pretrain = False
        self.msm_training = False
        # Lazy reverse-state map cache (used by the kept MSM TICA-basis attach).
        self._msm_state_to_frames = {}

        # Configs for data filters
        self.max_n_token = kwargs.get("max_n_token", -1)
        self.pdb_list = kwargs.get("pdb_list", None)
        if len(self.pdb_list) == 0:
            self.pdb_list = None
        # Used for removing rows in the indices list. Column names and excluded values are specified in this dict.
        self.exclusion_dict = kwargs.get("exclusion", {})
        self.limits = kwargs.get(
            "limits", -1
        )  # Limit number of indices rows, mainly for test

        self.error_dir = kwargs.get("error_dir", None)
        if self.error_dir is not None:
            os.makedirs(self.error_dir, exist_ok=True)

        self.msa_featurizer = msa_featurizer
        self.template_featurizer = template_featurizer

        # Read data
        self.indices_list = self.read_indices_list(indices_fpath)
        # re-index
        self.indices_list = self.indices_list.reset_index(drop=True)
        self.preprocess_indices_list(kwargs)
        self.indices_list['frame_id'] = self.indices_list['frame_id'].astype(int)

        # MSM initialization (for transition/population/autocorrelation losses + pseudo-trajectory)
        self.msm_cache_dir = kwargs.get("msm_cache_dir", None)
        self.msm_configs = kwargs.get("msm_configs", {})
        self.n_coarse_states = kwargs.get("n_coarse_states", 10)  # K used by loss-side artifacts
        # MSM-mode hybrid: K for state-pair sampling (defaults to n_coarse_states so
        # behavior is unchanged unless explicitly set). When != n_coarse_states, sampling
        # uses one K (e.g. 50, finer dynamics) while the population loss uses another
        # (e.g. 10, better KL statistics).
        self.n_sampling_states = kwargs.get("n_sampling_states", self.n_coarse_states)
        # Pseudo-trajectory augmentation (MSM pretrain feature) removed; kept off.
        self.pseudo_traj_ratio = 0.0
        self.pseudo_traj_stop_step = -1
        # If False, systems not in the pre-built cache are silently skipped (has_msm=False)
        # rather than built from scratch from bio files. Set True only for first-time builds.
        self.msm_build_missing = kwargs.get("msm_build_missing", True)
        self.msm_artifacts = {}  # {system_name: MSMArtifacts}
        self.traj_to_system = {}  # {traj_name: system_name}
        self._current_train_step = 0  # updated from trainer

        if self.msm_cache_dir is not None and self.msm_cache_dir != "NONE" and not self.dump_embeddings:
            self._init_msm()
    
    def filter_traj_list(self):
        if self.downsample > 0:
            random.seed(66)
            if self.downsample < len(self.traj_name_list):
                random_index = random.sample(range(len(self.traj_name_list)), self.downsample)
                self.traj_name_list = [self.traj_name_list[i] for i in random_index]

    def set_train_step(self, step: int):
        """Called from trainer to update current training step (for pseudo-traj scheduling)."""
        self._current_train_step = step

    def _get_system_name(self, traj_name: str) -> str:
        """Extract system name from traj_name.
        MSR/BioEmu: 'cath2_1a1wA00|run106_protein' → system = 'cath2_1a1wA00' (before '|')
        Atlas: '6irx_A_R2' → system = '6irx_A' (first 2 underscore segments)
        """
        if "|" in traj_name:
            return traj_name.split("|")[0]
        parts = traj_name.split("_")
        return "_".join(parts[:2])

    def _init_msm(self):
        """Initialize MSM artifacts: build on first run, load from cache subsequently."""
        from protenix.data.msm_builder import MSMCache

        self.msm_cache = MSMCache(self.msm_cache_dir, self.msm_configs)

        # Group traj_names by system
        system_to_trajs = {}
        for traj_name in self.traj_name_list:
            system_name = self._get_system_name(traj_name)
            self.traj_to_system[traj_name] = system_name
            system_to_trajs.setdefault(system_name, []).append(traj_name)
        # Cache the reverse lookup for fast sibling-trajectory access in pretrain mode.
        self.system_to_traj_names = system_to_trajs

        # Check which systems need building
        to_build = {
            sn: trajs for sn, trajs in system_to_trajs.items()
            if not self.msm_cache.has(sn)
        }

        if to_build and self.msm_build_missing:
            logger.info(f"Building MSM for {len(to_build)}/{len(system_to_trajs)} systems "
                        f"(cache: {self.msm_cache_dir})")
            for system_name, traj_names in tqdm(to_build.items(), desc="Building MSM"):
                traj_ca_coords = self._load_all_ca_coords_for_system(traj_names)
                if traj_ca_coords:
                    self.msm_cache.build_and_save(system_name, traj_ca_coords)
        elif to_build:
            logger.info(f"[MSM] Skipping build for {len(to_build)}/{len(system_to_trajs)} systems "
                        f"not in cache (msm_build_missing=False); they will have has_msm=False")
        else:
            logger.info(f"MSM cache fully built for {len(system_to_trajs)} systems")

        # Load all into memory
        for system_name in system_to_trajs:
            artifacts = self.msm_cache.get(system_name)
            if artifacts is not None:
                self._precompute_cluster_log_vars(artifacts)
                self.msm_artifacts[system_name] = artifacts

        logger.info(f"MSM artifacts loaded for {len(self.msm_artifacts)}/{len(system_to_trajs)} systems")

    @staticmethod
    def _precompute_cluster_log_vars(msm) -> None:
        """Precompute empirical per-cluster diagonal log-variance from raw TICA coords + labels.

        For multi-K plain-dict caches: writes `cluster_log_vars [K, NT]` into each `by_k[K]` entry.
        For old MSMArtifacts dataclass: skipped (those already carry `coarse_cluster_log_vars`).
        Variance floor 1e-4 to avoid log(0) for tiny clusters.
        """
        import numpy as _np
        if not isinstance(msm, dict):
            return
        coords_by_traj = msm.get("tica_coords_by_traj")
        by_k = msm.get("by_k")
        if not coords_by_traj or not by_k:
            return
        for _, k_entry in by_k.items():
            if "cluster_log_vars" in k_entry:
                continue
            n_states = k_entry["n_states"]
            centers = k_entry["cluster_centers"]
            nt = centers.shape[1]
            state_labels = k_entry["state_labels"]
            sumsq = _np.zeros((n_states, nt), dtype=_np.float64)
            counts = _np.zeros(n_states, dtype=_np.int64)
            for traj, labels in state_labels.items():
                coords = coords_by_traj.get(traj)
                if coords is None:
                    continue
                n = min(len(coords), len(labels))
                if n == 0:
                    continue
                lbl = _np.asarray(labels[:n])
                xs = _np.asarray(coords[:n])
                valid = lbl >= 0
                if not valid.any():
                    continue
                lbl_v = lbl[valid]
                d = xs[valid] - centers[lbl_v]
                _np.add.at(sumsq, lbl_v, d * d)
                _np.add.at(counts, lbl_v, 1)
            denom = _np.maximum(counts - 1, 1)[:, None].astype(_np.float64)
            var = sumsq / denom
            var = _np.maximum(var, 1e-4)
            k_entry["cluster_log_vars"] = _np.log(var).astype(_np.float32)

    def _load_all_ca_coords_for_system(self, traj_names: list) -> dict:
        """Load all frames' Cα coordinates for MSM building. Called only on first run."""
        traj_ca_coords = {}
        for traj_name in traj_names:
            target_indices = self.indices_list[self.indices_list.traj_name == traj_name]
            target_indices = target_indices.sort_values(by='frame_id')

            coords_list = []
            for idx in target_indices.index:
                try:
                    _, bioassembly_dict, _ = self._get_bioassembly_data(idx)
                    atom_array = bioassembly_dict["atom_array"]
                    ca_mask = atom_array.atom_name == "CA"
                    ca_coords = atom_array.coord[ca_mask]  # [N_ca, 3]
                    coords_list.append(ca_coords)
                except Exception:
                    continue

            if len(coords_list) > 1:
                # Verify consistent N_ca across frames
                n_ca_set = set(c.shape[0] for c in coords_list)
                if len(n_ca_set) == 1:
                    traj_ca_coords[traj_name] = np.stack(coords_list, axis=0)
                else:
                    logger.warning(f"Inconsistent N_ca in {traj_name}: {n_ca_set}, skipping")

        return traj_ca_coords

    def _attach_msm_data_to_sample(self, data, traj_name, selected_index, frame_id_list):
        """Attach MSM artifacts to the data dict for loss computation.

        Supports two artifact formats:
          - MSMArtifacts (old): dataclass with coarse_* fields
          - plain dict (new multi-K): has "by_k" key from build_multiK_msm.py
        """
        import numpy as _np

        system_name = self.traj_to_system.get(traj_name)
        if not system_name or system_name not in self.msm_artifacts:
            data["label_dict"]["has_msm"] = torch.tensor(False)
            return

        msm = self.msm_artifacts[system_name]

        # ── Extract fields — support both formats ───────────────────────────────
        if isinstance(msm, dict):
            # New multi-K plain-dict format: select K closest to n_coarse_states config
            by_k = msm.get("by_k", {})
            if not by_k:
                data["label_dict"]["has_msm"] = torch.tensor(False)
                return
            available_ks = sorted(by_k.keys())
            k_use = min(available_ks, key=lambda k: abs(k - self.n_coarse_states))
            k_entry = by_k[k_use]

            tica_mean         = msm["tica_mean"]
            tica_components   = msm["tica_components"]
            tica_eigenvalues  = msm["tica_eigenvalues"]
            idx_i             = msm["pair_indices_i"]
            idx_j             = msm["pair_indices_j"]
            n_ca_full         = msm.get("n_ca_full", 0)
            lagtime_frames    = msm["lagtime_frames"]
            state_labels_dict = k_entry["state_labels"]   # {traj: ndarray [T]}
            n_states          = k_entry["n_states"]
            cluster_centers   = k_entry["cluster_centers"]
            transition_matrix = k_entry["transition_matrix"]
            stationary_dist   = k_entry["stationary_dist"]
            log_vars_single = k_entry.get("cluster_log_vars")

            # Some caches (e.g. octapeptide K=10 with <10 populated clusters) have
            # cluster_centers shape [n_states_actual, D] but stationary_dist shape [K].
            # This breaks hmm_forward_log (log_pi + log_emit dim mismatch). Skip MSM
            # for these samples — loss gates on has_msm.
            if cluster_centers.shape[0] != stationary_dist.shape[0]:
                data["label_dict"]["has_msm"] = torch.tensor(False)
                return
        else:
            # Old MSMArtifacts dataclass format
            tica_mean         = msm.tica_mean
            tica_components   = msm.tica_components
            tica_eigenvalues  = msm.tica_eigenvalues
            idx_i             = msm.pair_indices_i
            idx_j             = msm.pair_indices_j
            n_ca_full         = getattr(msm, "n_ca_full", 0)
            lagtime_frames    = msm.lagtime_frames
            state_labels_dict = msm.coarse_state_labels
            n_states          = msm.coarse_n_states
            cluster_centers   = msm.coarse_cluster_centers
            transition_matrix = msm.coarse_transition_matrix
            stationary_dist   = msm.coarse_stationary_distribution
            log_vars_single = getattr(msm, "coarse_cluster_log_vars", None)

        if traj_name not in state_labels_dict:
            data["label_dict"]["has_msm"] = torch.tensor(False)
            return

        # Crop consistency check: if spatial cropping dropped CA atoms, pair_indices
        # built from the full protein would reference atoms that no longer exist.
        # n_ca_full=0 means old cache (unknown N_ca); fall through to the in-loss guard.
        if n_ca_full > 0:
            atom_to_tokatom_idx = data["input_feature_dict"].get("atom_to_tokatom_idx")
            is_ligand = data["input_feature_dict"].get("is_ligand")
            if atom_to_tokatom_idx is not None and is_ligand is not None:
                ca_mask = (atom_to_tokatom_idx == 1) & (~is_ligand.bool())
                n_ca_cropped = int(ca_mask.sum().item())
                if n_ca_cropped < n_ca_full:
                    data["label_dict"]["has_msm"] = torch.tensor(False)
                    return

        # Get state labels for selected frames.
        # Use frame_id_list (actual frame numbers from pkl files), NOT selected_index
        # (which is a row offset into the CSV DataFrame and doesn't correspond to
        # the sequential frame indices stored in coarse_state_labels).
        all_labels = state_labels_dict[traj_name]
        try:
            frame_labels = [int(all_labels[fid]) for fid in frame_id_list]
        except (IndexError, KeyError):
            data["label_dict"]["has_msm"] = torch.tensor(False)
            return

        # Guard: if any selected frame has an invalid label (-1, disconnected cluster),
        # the HMM emission for that frame is near-zero for all states, causing extreme
        # NaN/inf in the HMM forward pass. Disable TICA loss for the whole trajectory.
        if any(lbl < 0 for lbl in frame_labels):
            data["label_dict"]["has_msm"] = torch.tensor(False)
            return

        # Empirical population over the SAMPLED training window (not the full trajectory).
        # Full-traj target is ill-posed for Atlas: a 31-frame window covers only ~3% of a
        # 1000-frame trajectory and ~46% of the full traj_pop mass → KL floor ≈ 9.
        # Aligning target with q_avg's actual support (the sampled frames) makes the loss
        # optimizable on Atlas while remaining ~identical on CATH2 (window ≈ full traj).
        _K = n_states
        _window_labels = _np.asarray(frame_labels, dtype=_np.int64)
        _counts = _np.bincount(_window_labels, minlength=_K)[:_K].astype(_np.float32)
        _traj_pop = _counts / _counts.sum()

        data["label_dict"]["has_msm"] = torch.tensor(True)
        data["label_dict"]["msm_traj_name"] = str(traj_name)
        # import logging as _logging; _logging.getLogger(__name__).info(f"[MSM] has_msm=True for {traj_name}, {len(frame_labels)} frames, labels={frame_labels[:3]}...")
        data["label_dict"]["msm_state_labels"] = torch.tensor(frame_labels, dtype=torch.long)
        data["label_dict"]["msm_traj_population"] = torch.from_numpy(_traj_pop)
        data["label_dict"]["msm_tica_mean"] = torch.from_numpy(tica_mean)
        data["label_dict"]["msm_tica_components"] = torch.from_numpy(tica_components)
        data["label_dict"]["msm_tica_eigenvalues"] = torch.from_numpy(tica_eigenvalues)
        data["label_dict"]["msm_cluster_centers"] = torch.from_numpy(cluster_centers)

        # Emission: single diagonal Gaussian per cluster.
        # `log_vars_single` comes from cache-time precompute (multi-K dict caches)
        # or from the legacy `coarse_cluster_log_vars` field (old MSMArtifacts).
        if log_vars_single is not None:
            data["label_dict"]["msm_cluster_log_vars"] = torch.from_numpy(log_vars_single)
        else:
            _K2, _D = cluster_centers.shape
            data["label_dict"]["msm_cluster_log_vars"] = torch.from_numpy(
                _np.zeros((_K2, _D), dtype=_np.float32)
            )

        data["label_dict"]["msm_transition_matrix"] = torch.from_numpy(transition_matrix)
        data["label_dict"]["msm_stationary_distribution"] = torch.from_numpy(stationary_dist)
        data["label_dict"]["msm_pair_indices_i"] = torch.from_numpy(idx_i)
        data["label_dict"]["msm_pair_indices_j"] = torch.from_numpy(idx_j)
        # frame_time_gap_ratio: ratio of actual training-frame spacing to the MSM lag time.
        # = actual_spacing / lagtime_frames
        # This ensures T^(frame_time_gap_ratio) in the HMM matches the physical time gap
        # between consecutive training frames.
        # actual_spacing is inferred from consecutive frame_ids (e.g. [0,10,20] → spacing=10).
        # Physical lag is a magnitude — reverse-time sampling still has positive |Δt|.
        # Without abs(), a reversed frame order produces negative ratio → λ^{-n} > 1 and
        # the ACF target (and T^k) become nonsense.
        if len(frame_id_list) >= 2:
            _actual_spacing = max(1, abs(int(frame_id_list[1] - frame_id_list[0])))
        else:
            _actual_spacing = 1
        data["label_dict"]["msm_frame_time_gap_ratio"] = torch.tensor(
            _actual_spacing / max(lagtime_frames, 1), dtype=torch.float32
        )

    def preprocess_indices_list(self, kwargs):
        # Only compute traj_name/frame_id from pdb_id if the CSV doesn't already have them.
        # MSR/BioEmu CSVs have correct traj_name/frame_id; Atlas CSVs derive them from pdb_id.
        if "traj_name" not in self.indices_list.columns:
            self.indices_list["traj_name"] = self.indices_list["pdb_id"].apply(
                lambda x: "_".join(x.split("_")[:-1])
            )
        if "frame_id" not in self.indices_list.columns:
            self.indices_list["frame_id"] = self.indices_list["pdb_id"].apply(
                lambda x: int(x.split("_")[-1])
            )
        traj_name2num_tokens = pd.Series(self.indices_list['num_tokens'].values, index=self.indices_list['traj_name'].values).to_dict()
        token_crop_size = kwargs.get("token_crop_size", 384)
        print("token_crop_size", token_crop_size)

        
        self.traj_name_list = sorted(list(set(self.indices_list.traj_name)))
        random.seed(66)
        random.shuffle(self.traj_name_list)
        self.traj_name_list = [x for x in self.traj_name_list if '16pk_A' not in x]
        # TODO, filter out traj_name with num_tokens > token_crop_size for now. fix this error later
        if not self.dump_embeddings:
            self.traj_name_list = [x for x in self.traj_name_list if int(traj_name2num_tokens[x]) <= token_crop_size]

        self.precomputed_emb_dir = kwargs.get("precomputed_emb_dir", None)
        self.precomputed_emb_dir = Path(self.precomputed_emb_dir)
        if not self.dump_embeddings:
            saved_traj_names = os.listdir(self.precomputed_emb_dir)
            saved_traj_names = [x.split('.')[0] for x in saved_traj_names]
            saved_traj_names = set(saved_traj_names)
            self.traj_name_list = [x for x in self.traj_name_list if x in saved_traj_names]

        self.filter_traj_list()
        self.traj_name_list = sorted(self.traj_name_list)
        split_id = kwargs.get("split_id", -1)
        total_split = kwargs.get("total_split", -1)
        if split_id >= 0 and total_split > 0:
            import math
            total_len = len(self.traj_name_list)
            split_num = math.ceil(total_len / total_split)
            self.traj_name_list = self.traj_name_list[split_id*split_num: (split_id+1)*split_num]

    @staticmethod
    def read_pdb_list(pdb_list: Union[list, str]) -> Optional[list]:
        """
        Reads a list of PDB IDs from a file or directly from a list.

        Args:
            pdb_list: A list of PDB IDs or a file path containing PDB IDs.

        Returns:
            A list of PDB IDs if the input is valid, otherwise None.
        """
        if pdb_list is None:
            return None

        if isinstance(pdb_list, list):
            return pdb_list

        with open(pdb_list, "r") as f:
            pdb_filter_list = []
            for l in f.readlines():
                l = l.strip()
                if l:
                    pdb_filter_list.append(l)
        return pdb_filter_list

    def read_indices_list(self, indices_fpath: Union[str, Path]) -> pd.DataFrame:
        """
        Reads and processes a list of indices from a CSV file.

        Args:
            indices_fpath: Path to the CSV file containing the indices.

        Returns:
            A DataFrame containing the processed indices.
        """
        indices_list = read_indices_csv(indices_fpath)
        num_data = len(indices_list)
        logger.info(f"#Rows in indices list: {num_data}")
        # Filter by pdb_list
        if self.pdb_list is not None:
            pdb_filter_list = set(self.read_pdb_list(pdb_list=self.pdb_list))
            indices_list = indices_list[indices_list["pdb_id"].isin(pdb_filter_list)]
            logger.info(f"[filtered by pdb_list] #Rows: {len(indices_list)}")

        # Filter by max_n_token
        if self.max_n_token > 0:
            valid_mask = indices_list["num_tokens"].astype(int) <= self.max_n_token
            removed_list = indices_list[~valid_mask]
            indices_list = indices_list[valid_mask]
            logger.info(f"[removed] #Rows: {len(removed_list)}")
            logger.info(f"[removed] #PDB: {removed_list['pdb_id'].nunique()}")
            logger.info(
                f"[filtered by n_token ({self.max_n_token})] #Rows: {len(indices_list)}"
            )

        # Filter by exclusion_dict
        for col_name, exclusion_list in self.exclusion_dict.items():
            cols = col_name.split("|")
            exclusion_set = {tuple(excl.split("|")) for excl in exclusion_list}

            def is_valid(row):
                return tuple(row[col] for col in cols) not in exclusion_set

            valid_mask = indices_list.apply(is_valid, axis=1)
            indices_list = indices_list[valid_mask].reset_index(drop=True)
            logger.info(
                f"[Excluded by {col_name} -- {exclusion_list}] #Rows: {len(indices_list)}"
            )
        self.print_data_stats(indices_list)

        # Group by pdb_id
        # A list of dataframe. Each contains one pdb with multiple rows.
        if self.group_by_pdb_id:
            indices_list = [
                df.reset_index() for _, df in indices_list.groupby("pdb_id", sort=True)
            ]

        if self.sort_by_n_token:
            # Sort the dataset in a descending order, so that if OOM it will raise Error at an early stage.
            if self.group_by_pdb_id:
                indices_list = sorted(
                    indices_list,
                    key=lambda df: int(df["num_tokens"].iloc[0]),
                    reverse=True,
                )
            else:
                indices_list = indices_list.sort_values(
                    by="num_tokens", key=lambda x: x.astype(int), ascending=False
                ).reset_index(drop=True)

        if self.find_eval_chain_interface:
            # Remove data that does not contain eval_type in the EvaluationChainInterface list
            if self.group_by_pdb_id:
                indices_list = [
                    df
                    for df in indices_list
                    if len(
                        set(df["eval_type"].to_list()).intersection(
                            set(EvaluationChainInterface)
                        )
                    )
                    > 0
                ]
            else:
                indices_list = indices_list[
                    indices_list["eval_type"].apply(
                        lambda x: x in EvaluationChainInterface
                    )
                ]
        if self.limits > 0 and len(indices_list) > self.limits:
            logger.info(
                f"Limit indices list size from {len(indices_list)} to {self.limits}"
            )
            indices_list = indices_list[: self.limits]
        return indices_list

    def print_data_stats(self, df: pd.DataFrame) -> None:
        """
        Prints statistics about the dataset, including the distribution of molecular group types.

        Args:
            df: A DataFrame containing the indices list.
        """
        if self.name:
            logger.info("-" * 10 + f" Dataset {self.name}" + "-" * 10)
        df["mol_group_type"] = df.apply(
            lambda row: "_".join(
                sorted(
                    [
                        str(row["mol_1_type"]),
                        str(row["mol_2_type"]).replace("nan", "intra"),
                    ]
                )
            ),
            axis=1,
        )

        group_size_dict = dict(df["mol_group_type"].value_counts())
        for i, n_i in group_size_dict.items():
            logger.info(f"{i}: {n_i}/{len(df)}({round(n_i*100/len(df), 2)}%)")

        logger.info("-" * 30)
        if "cluster_id" in df.columns:
            n_cluster = df["cluster_id"].nunique()
            for i in group_size_dict:
                n_i = df[df["mol_group_type"] == i]["cluster_id"].nunique()
                logger.info(f"{i}: {n_i}/{n_cluster}({round(n_i*100/n_cluster, 2)}%)")
            logger.info("-" * 30)

        logger.info(f"Final pdb ids: {len(set(df.pdb_id.tolist()))}")
        logger.info("-" * 30)

    def __len__(self) -> int:
        # print("total data length:", len(self.traj_name_list))
        return len(self.traj_name_list)

    def save_error_data(self, idx: int, error_message: str) -> None:
        """
        Saves the error data for a specific index to a JSON file in the error directory.

        Args:
            idx: The index of the data sample that caused the error.
            error_message: The error message to be saved.
        """
        if self.error_dir is not None:
            sample_indice = self._get_sample_indice(idx=idx)
            data = sample_indice.to_dict()
            data["error"] = error_message

            filename = f"{sample_indice.pdb_id}-{sample_indice.chain_1_id}-{sample_indice.chain_2_id}.json"
            fpath = os.path.join(self.error_dir, filename)
            if not os.path.exists(fpath):
                with open(fpath, "w") as f:
                    json.dump(data, f)

    
    def get_time_interval(self, traj_name: str, spacing: float, **kwargs) -> float:
        raise NotImplementedError

    def get_spacing(self, traj_name: str, traj_length: int, **kwargs) -> float:
        raise NotImplementedError

    # ─────────────────────────────────────────────────────────────────────────
    # MSM-mode helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _msm_get_artifact_fields(self, system_name: str, k_use: int = None):
        """Extract (T, π, state_labels_dict, lagtime_frames, n_states) from MSM artifacts
        for a specific K. Defaults to self.n_coarse_states for backwards compat; the MSM
        sampling branch passes k_use = self.n_sampling_states explicitly.

        Returns None if the system has no usable artifacts.
        """
        if k_use is None:
            k_use = self.n_coarse_states
        if system_name not in self.msm_artifacts:
            return None
        msm = self.msm_artifacts[system_name]
        if isinstance(msm, dict):
            by_k = msm.get("by_k", {})
            if not by_k:
                return None
            available_ks = sorted(by_k.keys())
            k_selected = min(available_ks, key=lambda k: abs(k - k_use))
            k_entry = by_k[k_selected]
            return (
                np.asarray(k_entry["transition_matrix"], dtype=np.float64),
                np.asarray(k_entry["stationary_dist"], dtype=np.float64),
                k_entry["state_labels"],
                int(msm["lagtime_frames"]),
                int(k_entry["n_states"]),
            )
        # Old dataclass format (no multi-K) — fall through with whatever it has
        return (
            np.asarray(msm.coarse_transition_matrix, dtype=np.float64),
            np.asarray(msm.coarse_stationary_distribution, dtype=np.float64),
            msm.coarse_state_labels,
            int(msm.lagtime_frames),
            int(msm.coarse_n_states),
        )

    def _msm_build_state_to_frames(self, system_name: str, n_states: int, state_labels_dict):
        """Build {state_id: list[(traj_name, frame_idx)]} for a system.

        Cached by (system_name, n_states) so different K values get independent maps.
        """
        cache_key = (system_name, int(n_states))
        if cache_key in self._msm_state_to_frames:
            return self._msm_state_to_frames[cache_key]
        sibling_trajs = self.system_to_traj_names.get(system_name, [])
        state2frames = {s: [] for s in range(n_states)}
        for tn in sibling_trajs:
            labels = state_labels_dict.get(tn)
            if labels is None:
                continue
            labels = np.asarray(labels)
            for fid, lbl in enumerate(labels):
                lbl_i = int(lbl)
                if 0 <= lbl_i < n_states:
                    state2frames[lbl_i].append((tn, fid))
        self._msm_state_to_frames[cache_key] = state2frames
        return state2frames

    def _msm_sample_pairs_anchored(self, T: np.ndarray, pi: np.ndarray, state2frames: dict,
                                   k_msm_steps: int, n_pairs: int):
        """Anchored-MSM sampling: pick ONE c_0 per call, all pairs share it.
        Returns (pairs, c0, target_row) where target_row = T_pow[c0, :] sums to 1
        (the population-loss target). Returns None on failure.
        """
        n_states = T.shape[0]
        if k_msm_steps <= 1:
            T_pow = T
        else:
            T_pow = np.linalg.matrix_power(T, k_msm_steps)
            T_pow = np.clip(T_pow, 0.0, None)
            row_sums = T_pow.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            T_pow = T_pow / row_sums
        populated_states = [s for s in range(n_states) if len(state2frames.get(s, [])) > 0]
        if len(populated_states) == 0:
            return None
        pop_mask = np.zeros(n_states, dtype=bool)
        pop_mask[populated_states] = True
        pi_pop = np.where(pop_mask, pi, 0.0)
        pi_pop_sum = pi_pop.sum()
        if pi_pop_sum <= 0:
            return None
        # c_0 ~ UNIFORM over populated states. We explicitly avoid π-weighting because
        # high-π clusters tend to be protenix's "attractors" — those samples are already
        # well-learned. Uniform sampling effectively up-weights rare clusters, focusing
        # gradient on harder transitions the model hasn't mastered yet.
        c0 = int(np.random.choice(populated_states))
        target_row = T_pow[c0].astype(np.float64)
        sample_row = np.where(pop_mask, target_row, 0.0)
        sample_row_sum = sample_row.sum()
        if sample_row_sum <= 0:
            sample_row = pi_pop / pi_pop_sum
        else:
            sample_row = sample_row / sample_row_sum
        pairs = []
        for _ in range(n_pairs):
            s_t = int(np.random.choice(n_states, p=sample_row))
            frame0 = random.choice(state2frames[c0])
            frame_t = random.choice(state2frames[s_t])
            pairs.append((frame0[0], frame0[1], frame_t[0], frame_t[1]))
        return pairs, c0, target_row

    def _msm_sample_pairs(self, T: np.ndarray, pi: np.ndarray, state2frames: dict,
                         k_msm_steps: int, n_pairs: int):
        """Sample n_pairs (s0,s1) state pairs from π then T^k, return list of
        (traj_name_0, frame_idx_0, traj_name_1, frame_idx_1) tuples or None on failure."""
        n_states = T.shape[0]
        if k_msm_steps <= 1:
            T_pow = T
        else:
            T_pow = np.linalg.matrix_power(T, k_msm_steps)
            # numerical drift: clamp + renormalize rows
            T_pow = np.clip(T_pow, 0.0, None)
            row_sums = T_pow.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            T_pow = T_pow / row_sums
        # Restrict to populated states (some clusters may have 0 frames after labelling).
        populated_states = [s for s in range(n_states) if len(state2frames.get(s, [])) > 0]
        if len(populated_states) == 0:
            return None
        pop_mask = np.zeros(n_states, dtype=bool)
        pop_mask[populated_states] = True
        pi_eff = np.where(pop_mask, pi, 0.0)
        pi_sum = pi_eff.sum()
        if pi_sum <= 0:
            return None
        pi_eff = pi_eff / pi_sum
        pairs = []
        for _ in range(n_pairs):
            s0 = int(np.random.choice(n_states, p=pi_eff))
            row = T_pow[s0].copy()
            row = np.where(pop_mask, row, 0.0)
            row_sum = row.sum()
            if row_sum <= 0:
                # state s0 transitions only to empty states; fall back to π
                s1 = int(np.random.choice(n_states, p=pi_eff))
            else:
                s1 = int(np.random.choice(n_states, p=row / row_sum))
            frame0 = random.choice(state2frames[s0])
            frame1 = random.choice(state2frames[s1])
            pairs.append((frame0[0], frame0[1], frame1[0], frame1[1]))
        return pairs

    def __getitem__(self, idx: int):
        """
        Retrieves a data sample by processing the given index.
        If an error occurs, it attempts to handle it by either saving the error data or randomly sampling another index.

        Args:
            idx: The index of the data sample to retrieve.

        Returns:
            A dictionary containing the processed data sample.
        """
        # Try at most 10 times
        for _ in range(10):
            try:
                traj_name = self.traj_name_list[idx]
                target_indices = self.indices_list[self.indices_list.traj_name == traj_name]
                target_indices = target_indices.sort_values(by='frame_id')
                
                sample_indice = self.indices_list.iloc[target_indices.index[0]]
                
                data = self.process_one(target_indices.index[0], return_atom_token_array=True)
                selected_token_indices = data["basic"]["selected_token_indices"]
                if selected_token_indices is not None:
                    assert selected_token_indices.shape[0] == selected_token_indices.max() - selected_token_indices.min() + 1
                # assert data["basic"]["selected_token_indices"] is None, "selected_token_indices should be None in preference mode"
                # if data["basic"]["selected_token_indices"] is not None:
                #     print(data["basic"]["selected_token_indices"])
                
                # alt_idx = random.choice(target_indices.index)
                # random sample an alt structure other than the current one
                traj_length = len(target_indices)
                spacing = self.get_spacing(traj_name, traj_length) # time interval between frames = 1ns
                # MSR-* (BioEmu) datasets are used for training; must get is_train=True for
                # random traj_begin, trajectory reversal, and 20% use_time_attn=False (conformation sampling).
                # Exclude -test / -val splits so eval uses deterministic frame selection.
                _name_upper = self.name.upper()
                _is_msr_test = _name_upper.startswith("MSR-") and (
                    _name_upper.endswith("-TEST") or _name_upper.endswith("-VAL")
                )
                is_train = (
                    "train" in self.name
                    or "mdposit" in self.name.lower()
                    or (_name_upper.startswith("MSR-") and not _is_msr_test)
                )

                if traj_length == 1 and is_train:
                    raise ValueError(f"training traj length must be larger than 1")

                if traj_length == 1 or self.dump_embeddings:
                    selected_index = [0]
                else:
                    N_token = data["input_feature_dict"]["token_index"].shape[0]
                    N_atom = data["input_feature_dict"]["is_dna"].shape[0]
                    temp = max(N_token, N_atom // 8)
    
                    # random number of frames, push model to attention both on short and long hist/future
                    max_frame = int(5000000 / (temp*temp))
                    max_frame = min(max_frame, 50)
                    max_frame = min(max_frame, traj_length // spacing)
                    if max_frame > 10 and is_train: 
                        max_frame = random.randrange(10, max_frame+1)
                        
                    selected_index = []
                    if not is_train: 
                        traj_begin = 0
                    else:
                        traj_begin = random.randrange(0, traj_length - (max_frame-1) * spacing )
                    while True:
                        if traj_begin >= traj_length:
                            break
                        if len(selected_index) >= max_frame:
                            break
                        selected_index.append(traj_begin)
                        traj_begin += spacing 
    
                    # reversed traj is also a real traj
                    if "unbinding" not in self.name.lower():
                        if is_train and random.random() >= 0.5:
                            selected_index.reverse()
                            
                    if len(selected_index) < 2:
                        raise ValueError(f"traj_len must be at least 2")
                    # traj_len = 2 is allowed: ensemble losses (RMSF / velocity / ACF /
                    # TICA dynamics) are gated on traj_len >= 3 in loss.py, while per-frame
                    # losses (smooth_lddt / bond / mse / center) still apply.
                    

                frame_id = []
                # Load frames from the real trajectory.
                for i, alt_idx in enumerate(selected_index):
                    label, basic_info = self.process_one(
                        target_indices.index[alt_idx],
                        return_atom_token_array=True,
                        only_return_label=True,
                    )
                    frame_id.append(target_indices.iloc[alt_idx]['frame_id'])
                    data["label_dict"][f"coordinate_{i}"] = label["coordinate"]
                    assert label["coordinate"].shape == data["label_dict"]["coordinate_0"].shape
                data["label_dict"]["traj_len"] = len(selected_index)

                data["input_feature_dict"]["time_interval"] = self.get_time_interval(
                    traj_name, spacing, traj_length=traj_length
                )
                # use_time_attn: 20% of training samples disable time attention; val/test always True.
                if is_train:
                    data["input_feature_dict"]["use_time_attn"] = torch.tensor(
                        random.random() >= 0.2, dtype=torch.bool
                    )
                else:
                    data["input_feature_dict"]["use_time_attn"] = torch.tensor(True, dtype=torch.bool)

                data["input_feature_dict"]["frame0_coordinate_mask"] = data["label_dict"]["coordinate_mask"]

                # Attach MSM data for the TICA-dynamics loss (TICA basis + cluster artifacts
                # + per-traj state labels). Systems without an MSM cache set has_msm=False and
                # the loss skips them.
                self._attach_msm_data_to_sample(data, traj_name, selected_index, frame_id)

                # print(f"reading {traj_name}", frame_id, data["input_feature_dict"]["time_interval"])

                # # asign ligand reference position to be random GT conformation
                # is_ligand = data["input_feature_dict"]["is_ligand"]
                # ref_pos = data["input_feature_dict"]["ref_pos"]
                # if is_ligand.sum() > 0:
                #     # coords = [data["label_dict"][f"coordinate_{i}"] for i in range(data["label_dict"]["traj_len"])]
                #     # random_conf_ligand = random.choice(coords)
                #     random_conf_ligand = data["label_dict"][f"coordinate_0"]
                #     random_conf_ligand = random_conf_ligand[is_ligand==1, :] # [N_ligand_atom, 3]
                #     # normalize to center
                #     random_conf_ligand = random_conf_ligand - random_conf_ligand.mean(axis=0)
                #     # random augment
                #     random_conf_ligand =random_transform(
                #         random_conf_ligand,
                #         apply_augmentation=self.ref_pos_augment,
                #         centralize=True,
                #     )
                #     ref_pos[is_ligand==1, :] = torch.Tensor(random_conf_ligand)
                #     data["input_feature_dict"]["ref_pos"] = ref_pos
                #     data["input_feature_dict"]["ref_mask"][is_ligand==1] = 1.0

                return data

            except Exception as e:
                # raise Exception(e)
                error_message = f"{e} at {traj_name} idx {idx}:\n{traceback.format_exc()}"
                self.save_error_data(idx, error_message)

                if self.random_sample_if_failed:
                    logger.exception(f"[skip data {idx}] {error_message}")
                    # Random sample an index
                    idx = random.choice(range(len(self.traj_name_list)))
                    continue
                else:
                    raise Exception(e)
        return data

    def _get_bioassembly_data(
        self, idx: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        sample_indice = self._get_sample_indice(idx=idx)

        # Compressed-trajectory codec (MISATO / MDposit / unbinding): if the
        # bioassembly dir holds a per-trajectory template (<traj>.tpl.pkl.gz),
        # reconstruct the frame from template + stacked coords instead of reading a
        # per-frame pickle. See protenix/data/traj_codec.py.
        if self.bioassembly_dict_dir is not None:
            from protenix.data import traj_codec

            _traj = str(sample_indice.traj_name).split("|")[0]
            if traj_codec.has_codec(self.bioassembly_dict_dir, _traj):
                _fid = int(getattr(sample_indice, "frame_id"))
                bioassembly_dict = traj_codec.load_frame(
                    self.bioassembly_dict_dir, _traj, _fid
                )
                bioassembly_dict["pdb_id"] = sample_indice.pdb_id
                return sample_indice, bioassembly_dict, None

        bioassembly_dict_fpath = None
        if self.bioassembly_dict_dir is not None:
            bioassembly_dict_fpath = os.path.join(
                self.bioassembly_dict_dir, sample_indice.pdb_id + ".pkl.gz"
            )
        if not os.path.exists(bioassembly_dict_fpath):
            bioassembly_dict_fpath = os.path.join(self.bioassembly_dict_dir, sample_indice.traj_name, sample_indice.pdb_id + ".pkl.gz")
        if not os.path.exists(bioassembly_dict_fpath):
            bioassembly_dict_fpath = os.path.join(self.bioassembly_dict_dir, sample_indice.traj_name.split("|")[0], sample_indice.pdb_id + ".pkl.gz")

        bioassembly_dict = DataPipeline.get_data_bioassembly(
            bioassembly_dict_fpath=bioassembly_dict_fpath
        )
        bioassembly_dict["pdb_id"] = sample_indice.pdb_id
        return sample_indice, bioassembly_dict, bioassembly_dict_fpath

    @staticmethod
    def _reassign_atom_array_chain_id(atom_array: AtomArray):
        """
        In experiments conducted to observe overfitting effects using training sets,
        the pre-stored AtomArray in the training set may experience issues with discontinuous chain IDs due to filtering.
        Consequently, a temporary patch has been implemented to resolve this issue.

        e.g. 3x6u asym_id_int = [0, 1, 2, ... 18, 20] -> reassigned_asym_id_int [0, 1, 2, ..., 18, 19]
        """

        def _get_contiguous_array(array):
            array_uniq = np.sort(np.unique(array))
            map_dict = {i: idx for idx, i in enumerate(array_uniq)}
            new_array = np.vectorize(map_dict.get)(array)
            return new_array

        atom_array.asym_id_int = _get_contiguous_array(atom_array.asym_id_int)
        atom_array.entity_id_int = _get_contiguous_array(atom_array.entity_id_int)
        atom_array.sym_id_int = _get_contiguous_array(atom_array.sym_id_int)
        return atom_array

    @staticmethod
    def _shuffle_array_based_on_mol_id(token_array: TokenArray, atom_array: AtomArray):
        """
        Shuffle both token_array and atom_array.
        Atoms/tokens with the same mol_id will be shuffled as a integrated component.
        """

        # Get token mol_id
        centre_atom_indices = token_array.get_annotation("centre_atom_index")
        token_mol_id = atom_array[centre_atom_indices].mol_id

        # Get unique molecule IDs and shuffle them in place
        shuffled_mol_ids = np.unique(token_mol_id).copy()
        np.random.shuffle(shuffled_mol_ids)

        # Get shuffled token indices
        original_token_indices = np.arange(len(token_mol_id))
        shuffled_token_indices = []
        for mol_id in shuffled_mol_ids:
            mol_token_indices = original_token_indices[token_mol_id == mol_id]
            shuffled_token_indices.append(mol_token_indices)
        shuffled_token_indices = np.concatenate(shuffled_token_indices)

        # Get shuffled token and atom array
        # Use `CropData.select_by_token_indices` to shuffle safely
        token_array, atom_array, _, _ = CropData.select_by_token_indices(
            token_array=token_array,
            atom_array=atom_array,
            selected_token_indices=shuffled_token_indices,
        )

        return token_array, atom_array

    @staticmethod
    def _assign_random_sym_id(atom_array: AtomArray):
        """
        Assign random sym_id for chains of the same entity_id
        e.g.
        when entity_id = 0
            sym_id_int = [0, 1, 2] -> random_sym_id_int = [2, 0, 1]
        when entity_id = 1
            sym_id_int = [0, 1, 2, 3] -> random_sym_id_int = [3, 0, 1, 2]
        """

        def _shuffle(x):
            x_unique = np.sort(np.unique(x))
            x_shuffled = x_unique.copy()
            np.random.shuffle(x_shuffled)  # shuffle in-place
            map_dict = dict(zip(x_unique, x_shuffled))
            new_x = np.vectorize(map_dict.get)(x)
            return new_x.copy()

        for entity_id in np.unique(atom_array.label_entity_id):
            mask = atom_array.label_entity_id == entity_id
            atom_array.sym_id_int[mask] = _shuffle(atom_array.sym_id_int[mask])
        return atom_array

    def _ligand_valance_check(self, atom_array):
        def valance_check(valence):
            for atom, v in valence.items():
                if v == 0 or v > 4:
                    return False
                if atom[-1] == "O" and v > 2:
                    return False
            return True

        bonds = atom_array.bonds._bonds.tolist()
        is_ligand = atom_array.is_ligand
        element = atom_array.element
        atom_name = atom_array.atom_name

        if not np.any(is_ligand):
            return True

        ligand_atom_index = np.where(is_ligand)[0]
        global_to_local = {idx: i for i, idx in enumerate(ligand_atom_index)}
        ligand_bonds = [b for b in bonds if b[0] in ligand_atom_index and b[1] in ligand_atom_index]

        ligand_bonds_reindex = [[global_to_local[b[0]], global_to_local[b[1]]] for b in ligand_bonds]
        ligand_element = element[is_ligand == 1]
        ligand_atom_name = atom_name[is_ligand == 1]
        valence = {f"{ligand_atom_name[x]}_{ligand_element[x]}": 0 for x in range(len(ligand_atom_name))}
        for b in sorted(ligand_bonds_reindex, key=lambda x: (x[0], x[1])):
            # print(ligand_atom_name[b[0]], ligand_atom_name[b[1]])
            atom = f"{ligand_atom_name[b[0]]}_{ligand_element[b[0]]}"
            valence[atom] += 1
            atom = f"{ligand_atom_name[b[1]]}_{ligand_element[b[1]]}"
            valence[atom] += 1

        return valance_check(valence)

    def process_one(
        self, idx: int, return_atom_token_array: bool = False, only_return_label: bool = False
    ) -> dict[str, dict]:
        """
        Processes a single data sample by retrieving bioassembly data, applying various transformations, and cropping the data.
        It then extracts features and labels, and optionally returns the processed atom and token arrays.

        Args:
            idx: The index of the data sample to process.
            return_atom_token_array: Whether to return the processed atom and token arrays.

        Returns:
            A dict containing the input features, labels, basic_info and optionally the processed atom and token arrays.
        """

        sample_indice, bioassembly_dict, bioassembly_dict_fpath = (
            self._get_bioassembly_data(idx=idx)
        )
        # print(bioassembly_dict['num_tokens'], len(bioassembly_dict["token_array"]), self.traj_name2num_tokens[bioassembly_dict["traj_name"]], bioassembly_dict["traj_name"])
        # print(bioassembly_dict["traj_name"] in self.traj_name_list, len(self.traj_name_list))
        if not only_return_label and "example" not in self.name:
            if not self._ligand_valance_check(bioassembly_dict["atom_array"]):
                # The valence check is a training-data sanity gate (catches mmCIF ligands
                # whose on-the-fly bond perception is broken). At inference it must not be
                # fatal: generic ligands (e.g. peptidic inhibitors) can trip the heuristic
                # while still being valid generation targets — so warn and proceed, using
                # the same bonds the released runs used.
                if getattr(self, "dataset_type", "") == "inference" or "inference" in self.name:
                    logger.warning(
                        f"Ligand valence check failed for {bioassembly_dict['traj_name']}; "
                        f"proceeding (inference)."
                    )
                else:
                    raise ValueError(f"Valance check failed for {bioassembly_dict['traj_name']}")

        if self.use_reference_chains_only:
            # Get the reference chains
            ref_chain_ids = [sample_indice.chain_1_id, sample_indice.chain_2_id]
            # print(ref_chain_ids)
            if sample_indice.type == "chain":
                ref_chain_ids.pop(-1)
            # Remove other chains from the bioassembly_dict
            # Remove them safely using the crop method
            token_centre_atom_indices = bioassembly_dict["token_array"].get_annotation(
                "centre_atom_index"
            )
            token_chain_id = bioassembly_dict["atom_array"][
                token_centre_atom_indices
            ].chain_id
            is_ref_chain = np.isin(token_chain_id, ref_chain_ids)
            bioassembly_dict["token_array"], bioassembly_dict["atom_array"], _, _ = (
                CropData.select_by_token_indices(
                    token_array=bioassembly_dict["token_array"],
                    atom_array=bioassembly_dict["atom_array"],
                    selected_token_indices=np.arange(len(is_ref_chain))[is_ref_chain],
                )
            )

        if self.shuffle_mols:
            bioassembly_dict["token_array"], bioassembly_dict["atom_array"] = (
                self._shuffle_array_based_on_mol_id(
                    token_array=bioassembly_dict["token_array"],
                    atom_array=bioassembly_dict["atom_array"],
                )
            )

        if self.shuffle_sym_ids:
            bioassembly_dict["atom_array"] = self._assign_random_sym_id(
                bioassembly_dict["atom_array"]
            )

        if self.reassign_continuous_chain_ids:
            bioassembly_dict["atom_array"] = self._reassign_atom_array_chain_id(
                bioassembly_dict["atom_array"]
            )

        # Crop
        (
            crop_method,
            cropped_token_array,
            cropped_atom_array,
            cropped_msa_features,
            cropped_template_features,
            reference_token_index,
            selected_token_indices,
        ) = self.crop(
            sample_indice=sample_indice,
            bioassembly_dict=bioassembly_dict,
            **self.cropping_configs,
        )

        feat, label, label_full = self.get_feature_and_label(
            idx=idx,
            token_array=cropped_token_array,
            atom_array=cropped_atom_array,
            msa_features=cropped_msa_features,
            template_features=cropped_template_features,
            full_atom_array=bioassembly_dict["atom_array"],
            is_spatial_crop="spatial" in crop_method.lower(),
        )
        
        # Basic info, e.g. dimension related items
        basic_info = {
            "pdb_id": (
                bioassembly_dict["pdb_id"]
                if self.is_distillation is False
                else sample_indice["pdb_id"]
            ),
            "N_asym": torch.tensor([len(torch.unique(feat["asym_id"]))]),
            "N_token": torch.tensor([feat["token_index"].shape[0]]),
            "N_atom": torch.tensor([feat["atom_to_token_idx"].shape[0]]),
            "N_msa": torch.tensor([feat["msa"].shape[0]]),
            "bioassembly_dict_fpath": bioassembly_dict_fpath,
            "N_msa_prot_pair": torch.tensor([feat["prot_pair_num_alignments"]]),
            "N_msa_prot_unpair": torch.tensor([feat["prot_unpair_num_alignments"]]),
            "N_msa_rna_pair": torch.tensor([feat["rna_pair_num_alignments"]]),
            "N_msa_rna_unpair": torch.tensor([feat["rna_unpair_num_alignments"]]),
        }

        if only_return_label:
            return label, basic_info

        for mol_type in ("protein", "ligand", "rna", "dna"):
            abbr = {"protein": "prot", "ligand": "lig"}
            abbr_type = abbr.get(mol_type, mol_type)
            mol_type_mask = feat[f"is_{mol_type}"].bool()
            n_atom = int(mol_type_mask.sum(dim=-1).item())
            n_token = len(torch.unique(feat["atom_to_token_idx"][mol_type_mask]))
            basic_info[f"N_{abbr_type}_atom"] = torch.tensor([n_atom])
            basic_info[f"N_{abbr_type}_token"] = torch.tensor([n_token])

        # Add chain level chain_id
        asymn_id_to_chain_id = {
            atom.asym_id_int: atom.chain_id for atom in cropped_atom_array
        }
        chain_id_list = [
            asymn_id_to_chain_id[asymn_id_int]
            for asymn_id_int in sorted(asymn_id_to_chain_id.keys())
        ]
        basic_info["chain_id"] = chain_id_list

        data = {
            "input_feature_dict": feat,
            "label_dict": label,
            "label_full_dict": label_full,
            "basic": basic_info,
        }

        if return_atom_token_array:
            data["cropped_atom_array"] = cropped_atom_array
            data["cropped_token_array"] = cropped_token_array

        # precomputed embedding        
        if self.precomputed_emb_dir is not None:
            if "traj_name" in bioassembly_dict:
                traj_name = bioassembly_dict["traj_name"]
            else:
                pid = data["basic"]["pdb_id"]
                traj_name = '_'.join(pid.split("_")[:3])

            # BUG: 命名不一致，暂时打补丁fix一下。。。
            load_path = self.precomputed_emb_dir / f"{traj_name}.pt"
            if not os.path.exists(load_path):
                load_path = self.precomputed_emb_dir / f"{traj_name}_0.pt"
            if not os.path.exists(load_path):
                load_path = self.precomputed_emb_dir / f"{traj_name[:-2]}.pt"
            if not os.path.exists(load_path):
                load_path = self.precomputed_emb_dir / f"{traj_name.split('_Pro_lig')[0]}.pt"
            if not os.path.exists(load_path):
                load_path = self.precomputed_emb_dir / f"{traj_name.split("|")[0]}.pt"
            
            # print("loading msa from", load_path)
            if os.path.exists(load_path):
                # print("reading from", load_path)
                emb = torch.load(load_path, map_location='cpu', weights_only=False)
                # crop the embedding
                data["input_feature_dict"].update(emb)
                # TODO, meet new bug of none identical shape.... fix it here for now
                assert emb["s"].shape[0] == emb["z"].shape[0] == data["input_feature_dict"]["token_index"].shape[0]
            # else:
            #     if not self.dump_embeddings:
            #         raise ValueError(
            #             f"load_path must exists for now"
            #         )
        else:
            # No precomputed embeddings configured (e.g. inference): the model computes
            # s_inputs / s / z on the fly in its forward pass. Training configs should set
            # precomputed_emb_dir (regenerate via scripts/encode_embeddings.sh) for speed.
            if not self.dump_embeddings and getattr(self, "dataset_type", None) != "inference":
                logger.warning(
                    "precomputed_emb_dir not set; embeddings will be computed on the fly "
                    "(slow). Set precomputed_emb_dir for training."
                )

        data["basic"]["selected_token_indices"] = selected_token_indices

        return data

    def crop(
        self,
        sample_indice: pd.Series,
        bioassembly_dict: dict[str, Any],
        crop_size: int,
        method_weights: list[float],
        contiguous_crop_complete_lig: bool = True,
        spatial_crop_complete_lig: bool = True,
        drop_last: bool = True,
        remove_metal: bool = True,
    ) -> tuple[str, TokenArray, AtomArray, dict[str, Any], dict[str, Any]]:
        """
        Crops the bioassembly data based on the specified configurations.

        Returns:
            A tuple containing the cropping method, cropped token array, cropped atom array,
                cropped MSA features, and cropped template features.
        """
        return DataPipeline.crop(
            one_sample=sample_indice,
            bioassembly_dict=bioassembly_dict,
            crop_size=crop_size,
            msa_featurizer=self.msa_featurizer,
            template_featurizer=self.template_featurizer,
            method_weights=method_weights,
            contiguous_crop_complete_lig=contiguous_crop_complete_lig,
            spatial_crop_complete_lig=spatial_crop_complete_lig,
            drop_last=drop_last,
            remove_metal=remove_metal,
        )

    def _get_sample_indice(self, idx: int) -> pd.Series:
        """
        Retrieves the sample indice for a given index. If the dataset is grouped by PDB ID, it returns the first row of the PDB-idx.
        Otherwise, it returns the row at the specified index.

        Args:
            idx: The index of the data sample to retrieve.

        Returns:
            A pandas Series containing the sample indice.
        """
        if self.group_by_pdb_id:
            # Row-0 of PDB-idx
            sample_indice = self.indices_list[idx].iloc[0]
        else:
            sample_indice = self.indices_list.iloc[idx]
        return sample_indice

    def _get_eval_chain_interface_mask(
        self, idx: int, atom_array_chain_id: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Retrieves the evaluation chain/interface mask for a given index.

        Args:
            idx: The index of the data sample.
            atom_array_chain_id: An array containing the chain IDs of the atom array.

        Returns:
            A tuple containing the evaluation type, cluster ID, chain 1 mask, and chain 2 mask.
        """
        if self.group_by_pdb_id:
            df = self.indices_list[idx]
        else:
            df = self.indices_list.iloc[idx : idx + 1]

        # Only consider chain/interfaces defined in EvaluationChainInterface
        df = df[df["eval_type"].apply(lambda x: x in EvaluationChainInterface)].copy()
        if len(df) < 1:
            raise ValueError(
                f"Cannot find a chain/interface for evaluation in the PDB."
            )

        def get_atom_mask(row):
            chain_1_mask = atom_array_chain_id == row["chain_1_id"]
            if row["type"] == "chain":
                chain_2_mask = chain_1_mask
            else:
                chain_2_mask = atom_array_chain_id == row["chain_2_id"]
            chain_1_mask = torch.tensor(chain_1_mask).bool()
            chain_2_mask = torch.tensor(chain_2_mask).bool()
            if chain_1_mask.sum() == 0 or chain_2_mask.sum() == 0:
                return None, None
            return chain_1_mask, chain_2_mask

        df["chain_1_mask"], df["chain_2_mask"] = zip(*df.apply(get_atom_mask, axis=1))
        df = df[df["chain_1_mask"].notna()]  # drop NaN

        if len(df) < 1:
            raise ValueError(
                f"Cannot find a chain/interface for evaluation in the atom_array."
            )

        eval_type = np.array(df["eval_type"].tolist())
        cluster_id = np.array(df["cluster_id"].tolist())
        # [N_eval, N_atom]
        chain_1_mask = torch.stack(df["chain_1_mask"].tolist())
        # [N_eval, N_atom]
        chain_2_mask = torch.stack(df["chain_2_mask"].tolist())

        return eval_type, cluster_id, chain_1_mask, chain_2_mask

    def get_feature_and_label(
        self,
        idx: int,
        token_array: TokenArray,
        atom_array: AtomArray,
        msa_features: dict[str, Any],
        template_features: dict[str, Any],
        full_atom_array: AtomArray,
        is_spatial_crop: bool = True,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Get feature and label information for a given data point.
        It uses a Featurizer object to obtain input features and labels, and applies several
        steps to add other features and labels. Finally, it returns the feature dictionary, label
        dictionary, and a full label dictionary.

        Args:
            idx: Index of the data point.
            token_array: Token array representing the amino acid sequence.
            atom_array: Atom array containing atomic information.
            msa_features: Dictionary of MSA features.
            template_features: Dictionary of template features.
            full_atom_array: Full atom array containing all atoms.
            is_spatial_crop: Flag indicating whether spatial cropping is applied, by default True.

        Returns:
            A tuple containing the feature dictionary and the label dictionary.

        Raises:
            ValueError: If the ligand cannot be found in the data point.
        """
        # Get feature and labels from Featurizer
        feat = Featurizer(
            cropped_token_array=token_array,
            cropped_atom_array=atom_array,
            ref_pos_augment=self.ref_pos_augment,
            lig_atom_rename=self.lig_atom_rename,
        )
        features_dict = feat.get_all_input_features()
        labels_dict = feat.get_labels()

        # Permutation list for atom permutation
        features_dict["atom_perm_list"] = feat.get_atom_permutation_list()

        # Labels for multi-chain permutation
        # Note: the returned full_atom_array may contain fewer atoms than the input
        label_full_dict, full_atom_array = Featurizer.get_gt_full_complex_features(
            atom_array=full_atom_array,
            cropped_atom_array=atom_array,
            get_cropped_asym_only=is_spatial_crop,
        )

        # Masks for Pocket Metrics
        if self.find_pocket:
            # Get entity_id of the interested ligand
            sample_indice = self._get_sample_indice(idx=idx)
            if sample_indice.mol_1_type == "ligand":
                lig_entity_id = str(sample_indice.entity_1_id)
                lig_chain_id = str(sample_indice.chain_1_id)
            elif sample_indice.mol_2_type == "ligand":
                lig_entity_id = str(sample_indice.entity_2_id)
                lig_chain_id = str(sample_indice.chain_2_id)
            else:
                raise ValueError(f"Cannot find ligand from this data point.")
            # Make sure the cropped array contains interested ligand
            assert lig_entity_id in set(atom_array.label_entity_id)
            assert lig_chain_id in set(atom_array.chain_id)

            # Get asym ID of the specific ligand in the `main` pocket
            lig_asym_id = atom_array.label_asym_id[atom_array.chain_id == lig_chain_id]
            assert len(np.unique(lig_asym_id)) == 1
            lig_asym_id = lig_asym_id[0]
            ligands = [lig_asym_id]

            if self.find_all_pockets:
                # Get asym ID of other ligands with the same entity_id
                all_lig_asym_ids = set(
                    full_atom_array[
                        full_atom_array.label_entity_id == lig_entity_id
                    ].label_asym_id
                )
                ligands.extend(list(all_lig_asym_ids - set([lig_asym_id])))

            # Note: the `main` pocket is the 0-indexed one.
            # [N_pocket, N_atom], [N_pocket, N_atom].
            # If not find_all_pockets, then N_pocket = 1.
            interested_ligand_mask, pocket_mask = feat.get_lig_pocket_mask(
                atom_array=full_atom_array, lig_label_asym_id=ligands
            )

            label_full_dict["pocket_mask"] = pocket_mask
            label_full_dict["interested_ligand_mask"] = interested_ligand_mask

        # Masks for Chain/Interface Metrics
        if self.find_eval_chain_interface:
            eval_type, cluster_id, chain_1_mask, chain_2_mask = (
                self._get_eval_chain_interface_mask(
                    idx=idx, atom_array_chain_id=full_atom_array.chain_id
                )
            )
            labels_dict["eval_type"] = eval_type  # [N_eval]
            labels_dict["cluster_id"] = cluster_id  # [N_eval]
            labels_dict["chain_1_mask"] = chain_1_mask  # [N_eval, N_atom]
            labels_dict["chain_2_mask"] = chain_2_mask  # [N_eval, N_atom]

        # Make dummy features for not implemented features
        dummy_feats = []
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)
        if len(template_features) == 0:
            dummy_feats.append("template")
        else:
            template_features = dict_to_tensor(template_features)
            features_dict.update(template_features)

        features_dict = make_dummy_feature(
            features_dict=features_dict, dummy_feats=dummy_feats
        )
        # Transform to right data type
        features_dict = data_type_transform(feat_or_label_dict=features_dict)
        labels_dict = data_type_transform(feat_or_label_dict=labels_dict)

        # Is_distillation
        features_dict["is_distillation"] = torch.tensor([self.is_distillation])
        if self.is_distillation is True:
            features_dict["resolution"] = torch.tensor([-1.0])
        return features_dict, labels_dict, label_full_dict


class AtlasSingleDataset(BaseSingleDataset):
    # def filter_traj_list(self):
    #     test_id_strict = {'6yhu', '7jrq', '6q10', '6pce', '7qsu', '6ovk', '6rrv', '7rm7', '6l8s', '6jpt', '6iah', '7s86', '6c62', '6oz1', '6lrd', '6hj6', '7ec1', '6c0h', '6fy5', '6o2v', '6d7y', '6cb7', '6y2x', '6h49', '6p5h', '7fd1', '7asg', '7n0j', '6a9a', '6tly', '6qj0', '6uof', '7wab', '6e33', '6hem', '7c45', '7p46', '6ao8', '6kty', '6rwt', '6dgk', '6fc0', '6ndw', '6b1z', '7k7p', '6f45', '7dmn', '6zsl', '6bwq', '6mbg', '6bk4', '6vjg', '6lus', '7a66', '6dnm', '6dlm', '6ono', '7ead', '6l34', '6q9c', '6l4l', '6cka', '6nl2', '6l4p', '6h86', '6odd', '6o6y', '6bn0', '7la6', '6okd', '7bwf', '6e5y', '7mf4', '6ro6', '6e7e', '6eu8', '6anz', '6in7', '6idx', '6as3', '6irx', '6gus', '6mdw', '6sms', '6xb3'}
    #     self.traj_name_list = [x for x in self.traj_name_list if x.split("_")[0].lower() in test_id_strict]
    #     return

    def get_time_interval(self, traj_name: str, spacing: float, **kwargs) -> float:
        return 0.1 * spacing
    
    def get_spacing(self, traj_name: str, traj_length: int, **kwargs) -> float:
        # Val spacing aligned with MSM lagtime_frames (=10 for Atlas), which is within the
        # training spacing distribution. Previous value (50) created a 100 ns ergodic window
        # outside what training teaches; see BioKinema/notes/val_pop_loss_analysis.md.
        # ATLAS_VAL_SPACING env var overrides (used by eval_dump_tica.sh to reproduce the
        # original problematic spacing=50 for diagnostics without code-level toggles).
        if "test" in self.name or "val" in self.name:
            _override = os.environ.get("ATLAS_VAL_SPACING", "")
            if _override:
                try:
                    return int(_override)
                except ValueError:
                    pass
            return 10

        # --- training: 10% spike at inference scale + 90% log-uniform ---
        # Atlas: 0.1 ns/frame, MSM lagtime = 10 frames = 1 ns. spike=10 oversamples
        # the inference / val time scale; log-uniform covers up to traj_length*2//3.
        # Spacing rounded to a multiple of 10 (= 1 ns) to match the MSM lagtime grid.
        short_scale = 10
        if traj_length < short_scale * 2:
            return short_scale
        if random.random() < 0.10:
            return short_scale
        # max_spacing also rounded to a multiple of 10 so clamping never breaks the grid
        max_spacing = max(short_scale, 10 * (traj_length // 4 // 10))
        log_min, log_max = math.log10(float(short_scale)), math.log10(float(max_spacing))
        spacing = int(round(math.pow(10.0, random.uniform(log_min, log_max))))
        spacing = 10 * int(round(spacing / 10.0))
        return max(short_scale, min(spacing, max_spacing))


class MisatoSingleDataset(BaseSingleDataset):
    def get_time_interval(self, traj_name: str, spacing: float, **kwargs) -> float:
        return 0.08 * spacing
    
    def get_spacing(self, traj_name: str, traj_length: int, **kwargs) -> float:
        if "test" in self.name or "val" in self.name:
            return 10

        # --- training: 20% spike at inference scale + 80% log-uniform ---
        # Misato: 0.08 ns/frame. Trajectories are typically short; cap max
        # at min(10, traj_length*2//3) to stay within historical range while
        # giving log-uniform coverage of intermediate scales.
        short_scale = 1
        if traj_length < 4:
            return short_scale
        if random.random() < 0.10:
            return short_scale
        max_spacing = max(short_scale + 1, min(10, traj_length // 4))
        log_min, log_max = math.log10(float(short_scale)), math.log10(float(max_spacing))
        spacing = int(round(math.pow(10.0, random.uniform(log_min, log_max))))
        return max(short_scale, min(spacing, max_spacing))

    def preprocess_indices_list(self, kwargs):
        self.traj_name2num_tokens = pd.Series(self.indices_list['num_tokens'].values, index=self.indices_list['traj_name'].values).to_dict()
        token_crop_size = kwargs.get("token_crop_size", 384)
        # print("token_crop_size", token_crop_size)
        # print(self.traj_name2num_tokens)
        if kwargs["interval_fpath"] is not None:
            self.interval_dict = json.load(open(kwargs["interval_fpath"], "r"))
        
        self.traj_name_list = sorted(list(set(self.indices_list.traj_name)))
        random.seed(66)
        random.shuffle(self.traj_name_list)
        # TODO, filter out traj_name with num_tokens > token_crop_size for now. fix this error later
        self.traj_name_list = [x for x in self.traj_name_list if int(self.traj_name2num_tokens[x]) <= token_crop_size]
        self.precomputed_emb_dir = kwargs.get("precomputed_emb_dir", None)
        self.precomputed_emb_dir = Path(self.precomputed_emb_dir)
        
        if not (kwargs["dump_embeddings"] or "example" in self.name):
            saved_traj_names = os.listdir(self.precomputed_emb_dir)
            saved_traj_names = [x.split('.')[0] for x in saved_traj_names]
            saved_traj_names = [x[:-2] if x.endswith("_0") else x for x in saved_traj_names]
            saved_traj_names = set(saved_traj_names)
            self.traj_name_list = [x for x in self.traj_name_list if x.split("|")[0] in saved_traj_names]

        self.filter_traj_list()
        self.traj_name_list = sorted(self.traj_name_list)
        random.seed(66)
        random.shuffle(self.traj_name_list)
        split_id = kwargs.get("split_id", -1)
        total_split = kwargs.get("total_split", -1)
        if split_id >= 0 and total_split > 0:
            import math
            total_len = len(self.traj_name_list)
            split_num = math.ceil(total_len / total_split)
            self.traj_name_list = self.traj_name_list[split_id*split_num: (split_id+1)*split_num]


    def super_read_indices_list(self, indices_fpath: Union[str, Path]) -> pd.DataFrame:
        """
        identical to super().read_indices_list, removes all logging info
        """
        indices_list = read_indices_csv(indices_fpath)
        num_data = len(indices_list)
        # logger.info(f"#Rows in indices list: {num_data}")
        # Filter by pdb_list
        if self.pdb_list is not None:
            pdb_filter_list = set(self.read_pdb_list(pdb_list=self.pdb_list))
            indices_list = indices_list[indices_list["pdb_id"].isin(pdb_filter_list)]
            # logger.info(f"[filtered by pdb_list] #Rows: {len(indices_list)}")

        # Filter by max_n_token
        if self.max_n_token > 0:
            valid_mask = indices_list["num_tokens"].astype(int) <= self.max_n_token
            removed_list = indices_list[~valid_mask]
            indices_list = indices_list[valid_mask]
            # logger.info(f"[removed] #Rows: {len(removed_list)}")
            # logger.info(f"[removed] #PDB: {removed_list['pdb_id'].nunique()}")
            # logger.info(
            #     f"[filtered by n_token ({self.max_n_token})] #Rows: {len(indices_list)}"
            # )

        # Filter by exclusion_dict
        for col_name, exclusion_list in self.exclusion_dict.items():
            cols = col_name.split("|")
            exclusion_set = {tuple(excl.split("|")) for excl in exclusion_list}

            def is_valid(row):
                return tuple(row[col] for col in cols) not in exclusion_set

            valid_mask = indices_list.apply(is_valid, axis=1)
            indices_list = indices_list[valid_mask].reset_index(drop=True)
            # logger.info(
            #     f"[Excluded by {col_name} -- {exclusion_list}] #Rows: {len(indices_list)}"
            # )
        # self.print_data_stats(indices_list)

        # Group by pdb_id
        # A list of dataframe. Each contains one pdb with multiple rows.
        if self.group_by_pdb_id:
            indices_list = [
                df.reset_index() for _, df in indices_list.groupby("pdb_id", sort=True)
            ]

        if self.sort_by_n_token:
            # Sort the dataset in a descending order, so that if OOM it will raise Error at an early stage.
            if self.group_by_pdb_id:
                indices_list = sorted(
                    indices_list,
                    key=lambda df: int(df["num_tokens"].iloc[0]),
                    reverse=True,
                )
            else:
                indices_list = indices_list.sort_values(
                    by="num_tokens", key=lambda x: x.astype(int), ascending=False
                ).reset_index(drop=True)

        if self.find_eval_chain_interface:
            # Remove data that does not contain eval_type in the EvaluationChainInterface list
            if self.group_by_pdb_id:
                indices_list = [
                    df
                    for df in indices_list
                    if len(
                        set(df["eval_type"].to_list()).intersection(
                            set(EvaluationChainInterface)
                        )
                    )
                    > 0
                ]
            else:
                indices_list = indices_list[
                    indices_list["eval_type"].apply(
                        lambda x: x in EvaluationChainInterface
                    )
                ]
        if self.limits > 0 and len(indices_list) > self.limits:
            # logger.info(
            #     f"Limit indices list size from {len(indices_list)} to {self.limits}"
            # )
            indices_list = indices_list[: self.limits]
        return indices_list


    def read_indices_list(self, indices_fpath):
        # read all indices csv files, and concat them
        indices_list_all = []
        print(indices_fpath)
        
        # for fpath in tqdm(list(glob.glob(os.path.join(indices_fpath, "*.csv")))[:10]):
        for fpath in tqdm(glob.glob(os.path.join(indices_fpath, "*.csv"))): 
            indices_list = self.super_read_indices_list(fpath) 
            indices_list_all.append(indices_list)
        indices_list_all = pd.concat(indices_list_all, axis=0)
        indices_list_all = indices_list_all.drop_duplicates(subset="pdb_id", keep="first")
        self.print_data_stats(indices_list_all)
        return indices_list_all


class MDpositSingleDataset(MisatoSingleDataset):
    
    def get_time_interval(self, traj_name: str, spacing: float, **kwargs) -> float:
        """
        Retrieves the time interval for a specific trajectory name.

        Args:
            traj_name: The name of the trajectory.

        Returns:
            The time interval as a float value.
        """
        return spacing * self.interval_dict[traj_name] 
    
    def get_spacing(self, traj_name: str, traj_length: int, **kwargs) -> float:
        """
        Retrieves the spacing for a specific trajectory name.

        Args:
            traj_name: The name of the trajectory.

        Returns:
            The spacing as a float value.
        """
        interval = self.interval_dict[traj_name]

        # min_spacing such that physical interval*spacing >= 0.1 ns (inference scale)
        min_spacing = 1
        while True:
            if interval*min_spacing >= 0.1:
                break
            min_spacing = math.ceil(min_spacing * 1.2)

        # val/test: keep deterministic (original behaviour)
        if "test" in self.name or "val" in self.name:
            return traj_length // 20

        # --- training: 20% spike at inference scale + 80% log-uniform ---
        # MDposit: variable interval per trajectory, min_spacing is the physical
        # "inference scale" (>= 0.1 ns). Log-uniform expands the upper bound to
        # traj_length*2//3 so the model sees long-range correlations.
        max_spacing = max(min_spacing + 1, traj_length // 4)
        log_min, log_max = math.log10(float(min_spacing)), math.log10(float(max_spacing))
        spacing = int(round(math.pow(10.0, random.uniform(log_min, log_max))))
        return max(min_spacing, min(spacing, max_spacing))


class MSRSingleDataset(MisatoSingleDataset):
    
    def get_time_interval(self, traj_name: str, spacing: float, **kwargs) -> float:
        """
        Retrieves the time interval for a specific trajectory name.

        Args:
            traj_name: The name of the trajectory.

        Returns:
            The time interval as a float value.
        """
        return spacing * 10 # fixed interval with 10 ns
    
    def get_spacing(self, traj_name: str, traj_length: int, **kwargs) -> float:
        """
        Retrieves the spacing for a specific trajectory name.

        Spacing strategy (training):
          • log-uniform   so model sees frame pairs at
            10 ns – 1 μs timescales (training distribution covers inference scale
            and the slow MFPT scales).
          • With 20% probability force spacing=1 to oversample the inference
            timescale (avoids the attractor that appears when inference time
            scale falls at the lower edge of training distribution).

        Test/val:
          • Single deterministic spacing = traj_length//50 (unchanged).
        """
        max_spacing_eval = max(1, traj_length // 50)
        if "test" in self.name or "val" in self.name:
            return max_spacing_eval

        # --- training ---
        # 10% of the time: short spacing (= inference scale, 10 ns frame pair)
        if random.random() < 0.10:
            return 1

        max_spacing = max(2, traj_length * 2 // 3)
        log_min, log_max = 0.0, math.log10(float(max_spacing))
        spacing = int(round(math.pow(10.0, random.uniform(log_min, log_max))))
        return max(1, min(spacing, max_spacing))


class ExampleDataset(MisatoSingleDataset):
    def filter_traj_list(self):
        # demo_list = ["1AKE", "4AKE", "CASE"] #, "1Q21", "5P21", "2CEY", "6H76"]
        # self.traj_name_list = [x for x in self.traj_name_list if x.split("_")[0].upper() in demo_list]
        return

    def get_time_interval(self, traj_name: str, spacing: float, **kwargs) -> float:
        print(f"time interval = {0.1 * spacing} ns")
        return 0.1 * spacing
    
    def get_spacing(self, traj_name: str, traj_length: int, **kwargs) -> float:
        return 200


class UnbindingSingleDataset(MisatoSingleDataset):
    def get_time_interval(self, traj_name: str, spacing: float, **kwargs) -> float:
        traj_length = kwargs["traj_length"]
        max_spacing = math.ceil(traj_length / 50.)
        return float(spacing) / float(max_spacing)
    
    def get_spacing(self, traj_name: str, traj_length: int, **kwargs) -> float:
        # return 1
        min_spacing = 1
        max_spacing = math.ceil(traj_length / 50.)

        if "test" in self.name or "val" in self.name: 
            return max_spacing

        if min_spacing >= max_spacing:
            return max_spacing
        else:
            spacing = random.randint(min_spacing, max_spacing)
            return spacing

    def preprocess_indices_list(self, kwargs):
        self.traj_name2num_tokens = pd.Series(self.indices_list['num_tokens'].values, index=self.indices_list['traj_name'].values).to_dict()
        token_crop_size = kwargs.get("token_crop_size", 384)
        # print("token_crop_size", token_crop_size)
        # print(self.traj_name2num_tokens)
        if kwargs["interval_fpath"] is not None:
            self.interval_dict = json.load(open(kwargs["interval_fpath"], "r"))
        
        self.traj_name_list = sorted(list(set(self.indices_list.traj_name)))
        random.seed(66)
        random.shuffle(self.traj_name_list)
        # TODO, filter out traj_name with num_tokens > token_crop_size for now. fix this error later
        self.traj_name_list = [x for x in self.traj_name_list if int(self.traj_name2num_tokens[x]) <= token_crop_size]
        self.precomputed_emb_dir = kwargs.get("precomputed_emb_dir", None)
        self.precomputed_emb_dir = Path(self.precomputed_emb_dir)
        
        if not (kwargs["dump_embeddings"] or "example" in self.name):
            saved_traj_names = os.listdir(self.precomputed_emb_dir)
            saved_traj_names = [x.split('.')[0] for x in saved_traj_names]
            saved_traj_names = [x[:-2] if x.endswith("_0") else x for x in saved_traj_names]
            saved_traj_names = set(saved_traj_names)
            self.traj_name_list = [x for x in self.traj_name_list if x.split('_Pro_lig')[0] in saved_traj_names]

        self.filter_traj_list()
        self.traj_name_list = sorted(self.traj_name_list)
        random.seed(66)
        random.shuffle(self.traj_name_list)
        split_id = kwargs.get("split_id", -1)
        total_split = kwargs.get("total_split", -1)
        if split_id >= 0 and total_split > 0:
            import math
            total_len = len(self.traj_name_list)
            split_num = math.ceil(total_len / total_split)
            self.traj_name_list = self.traj_name_list[split_id*split_num: (split_id+1)*split_num]


class InferenceDataset(BaseSingleDataset):
    """
    Dataset for inference from a single PDB or CIF file.
    """
    
    def __init__(
        self,
        input_file: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> None:
        self.input_file = Path(input_file)
        # Unique scratch per job so parallel infer processes never share indices.csv /
        # bioassembly under the same dump_dir (would corrupt reads and raise EmptyDataError).
        if output_dir:
            scratch_root = Path(output_dir) / ".infer_scratch"
            scratch_root.mkdir(parents=True, exist_ok=True)
            self.output_dir = scratch_root / f"{self.input_file.stem}_{uuid.uuid4().hex}"
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path(tempfile.mkdtemp())
        self.base_time_interval = kwargs.pop("time_interval", 0.1)
        print("time_interval:", self.base_time_interval)
        self.dataset_type = "inference"
        
        # Handle precomputed_emb_path -> precomputed_emb_dir
        precomputed_emb_path = kwargs.pop("precomputed_emb_path", None)
        if precomputed_emb_path and "precomputed_emb_dir" not in kwargs:
            kwargs["precomputed_emb_dir"] = str(Path(precomputed_emb_path).parent)
        
        # Create output directories
        self.bioassembly_output_dir = self.output_dir / "bioassembly"
        self.bioassembly_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process input file
        self.cif_file = self._ensure_cif_format()
        self._generated_bioassembly_dict, self._generated_sample_indices = self._preprocess_cif()
        self._apply_precomputed_ligand_bonds()

        # Save indices to CSV
        self.indices_csv_path = self.output_dir / "indices.csv"
        pd.DataFrame(self._generated_sample_indices).to_csv(
            self.indices_csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC
        )

        kwargs["mmcif_dir"] = str(self.output_dir)
        kwargs["bioassembly_dict_dir"] = str(self.bioassembly_output_dir)
        kwargs["indices_fpath"] = str(self.indices_csv_path)

        super().__init__(**kwargs)
        
        logger.info(f"InferenceDataset initialized: {len(self.traj_name_list)} trajectories from {self.input_file}")
    
    @staticmethod
    def _update_nested_dict(base: dict, update: dict) -> None:
        """Recursively update nested dictionary."""
        for k, v in update.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                InferenceDataset._update_nested_dict(base[k], v)
            else:
                base[k] = v

    def _pdb_to_cif(self, pdb_path: Path, cif_path: Path) -> None:
        """Convert PDB to CIF format."""
        from biotite.structure.io import load_structure, save_structure
        pdb_structure = load_structure(str(pdb_path))
        save_structure(str(cif_path), pdb_structure)
        logger.info(f"Converted PDB to CIF: {pdb_path} -> {cif_path}")
        
    def _ensure_cif_format(self) -> Path:
        """Ensure input file is CIF format, convert if necessary."""
        suffix = self.input_file.suffix.lower()
        
        if suffix == ".pdb":
            cif_path = self.output_dir / f"{self.input_file.stem}.cif"
            self._pdb_to_cif(self.input_file, cif_path)
            return cif_path
        elif suffix in [".cif", ".mmcif"]:
            return self.input_file
        elif suffix == ".gz":
            import gzip
            import shutil
            
            if self.input_file.name.endswith(".cif.gz"):
                return self.input_file
            elif self.input_file.name.endswith(".pdb.gz"):
                decompressed = self.output_dir / self.input_file.stem
                with gzip.open(self.input_file, 'rb') as f_in:
                    with open(decompressed, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                cif_path = self.output_dir / f"{decompressed.stem}.cif"
                self._pdb_to_cif(decompressed, cif_path)
                return cif_path
        
        raise ValueError(f"Unsupported file format: {suffix}")
    
    def _preprocess_cif(self) -> tuple[dict[str, Any], list[dict]]:
        """Preprocess CIF file to generate bioassembly data."""
        try:
            sample_indices_list, bioassembly_dict = DataPipeline.get_data_from_mmcif(
                self.cif_file, pdb_cluster_file=None, dataset=self.dataset_type
            )
        except Exception as e:
            logger.error(f"Failed to preprocess {self.cif_file}: {e}")
            raise
        
        if not sample_indices_list or not bioassembly_dict:
            raise ValueError(f"Failed to generate data from {self.cif_file}")
        
        pdb_id = bioassembly_dict.get("pdb_id") or self.cif_file.stem
        bioassembly_dict["pdb_id"] = pdb_id
        
        self.bioassembly_fpath = self.bioassembly_output_dir / f"{pdb_id}.pkl.gz"
        dump_gzip_pickle(bioassembly_dict, self.bioassembly_fpath)
        
        logger.info(f"Preprocessed: {len(sample_indices_list)} samples from {self.cif_file}")
        return bioassembly_dict, sample_indices_list

    def _apply_precomputed_ligand_bonds(self):
        """Optionally replace this system's atom-array bonds with known-correct ones from a
        precomputed bioassembly pkl. On-the-fly ligand bond perception from a cif can DROP
        ring-closure bonds (verified: ~half of MISATO-OOD systems), which leaves those bonds
        unconstrained and badly inflates the bond-length/angle error. The precomputed pkls used
        in training (e.g. misato_bio_noref/<stem>/<stem>_*.pkl.gz) have the correct connectivity,
        and the atom array is built by the same parser so atom order matches exactly.

        Enable by setting env BIOKINEMA_LIGAND_BONDS_DIR=<dir>; for input <stem>.cif it loads
        <dir>/<stem>/<stem>_*.pkl.gz and copies its bonds (verifying atom count + element order)."""
        import glob as _glob
        bonds_dir = os.environ.get("BIOKINEMA_LIGAND_BONDS_DIR", "").strip()
        if not bonds_dir:
            return
        stem = self.input_file.stem
        cand = sorted(_glob.glob(os.path.join(bonds_dir, stem, f"{stem}_*.pkl.gz")))
        if not cand:
            logger.warning(f"[ligand_bonds] no precomputed bio for {stem} under {bonds_dir}")
            return
        try:
            ref = load_gzip_pickle(cand[0]) if "load_gzip_pickle" in globals() else None
            if ref is None:
                import gzip as _gz, pickle as _pk
                with _gz.open(cand[0], "rb") as f:
                    ref = _pk.load(f)
        except Exception as e:
            logger.warning(f"[ligand_bonds] failed to load {cand[0]}: {e}")
            return
        aa = self._generated_bioassembly_dict["atom_array"]
        ref_aa = ref["atom_array"]
        if len(aa) != len(ref_aa) or not np.array_equal(
            np.asarray(aa.element), np.asarray(ref_aa.element)
        ):
            logger.warning(
                f"[ligand_bonds] atom mismatch for {stem} (n={len(aa)} vs {len(ref_aa)}); "
                f"skipping bond replacement")
            return
        import copy as _copy
        aa.bonds = _copy.deepcopy(ref_aa.bonds)
        # persist the corrected bioassembly so downstream reads see the right bonds
        dump_gzip_pickle(self._generated_bioassembly_dict, self.bioassembly_fpath)
        logger.info(f"[ligand_bonds] replaced bonds for {stem} from {os.path.basename(cand[0])}")
    
    def preprocess_indices_list(self, kwargs: dict) -> None:
        """Override parent's method for inference."""
        if "traj_name" not in self.indices_list.columns:
            self.indices_list["traj_name"] = self.indices_list["pdb_id"].apply(
                lambda x: "_".join(x.split("_")[:3]) if "_" in x else x
            )
        if "frame_id" not in self.indices_list.columns:
            self.indices_list["frame_id"] = self.indices_list["pdb_id"].apply(
                lambda x: int(x.split("_")[-1]) if "_" in x and x.split("_")[-1].isdigit() else 0
            )
        
        self.traj_name_list = sorted(set(self.indices_list.traj_name))
        self.precomputed_emb_dir = Path(kwargs.get("precomputed_emb_dir")) if kwargs.get("precomputed_emb_dir") else None
    
    def get_time_interval(self, traj_name: str, spacing: float, **kwargs) -> float:
        return self.base_time_interval * spacing
    
    def get_spacing(self, traj_name: str, traj_length: int, **kwargs) -> int:
        return 1
    
    def _get_bioassembly_data(self, idx: int) -> tuple[pd.Series, dict[str, Any], str]:
        """Override to use preprocessed bioassembly data."""
        sample_indice = self._get_sample_indice(idx)
        bioassembly_dict = deepcopy(self._generated_bioassembly_dict)
        bioassembly_dict["pdb_id"] = sample_indice.pdb_id
        return sample_indice, bioassembly_dict, str(self.bioassembly_fpath)
    
    def get_bioassembly_dict(self) -> dict[str, Any]:
        """Return the bioassembly dictionary."""
        return self._generated_bioassembly_dict
    
    def get_indices_dataframe(self) -> pd.DataFrame:
        """Return the indices DataFrame."""
        return self.indices_list


class EmptyDataset(Dataset):
    def __init__(self):
        self.merged_datapoint_weights = []
        pass
    
    def __len__(self):
        return 0
    
    def __getitem__(self, idx):
        raise IndexError("Empty dataset")
        

def get_msa_featurizer(configs, dataset_name: str, stage: str, msa_cache_path: str) -> Optional[Callable]:
    """
    Creates and returns an MSAFeaturizer object based on the provided configurations.

    Args:
        configs: A dictionary containing the configurations for the MSAFeaturizer.
        dataset_name: The name of the dataset.
        stage: The stage of the dataset (e.g., 'train', 'test').

    Returns:
        An MSAFeaturizer object if MSA is enabled in the configurations, otherwise None.
    """
    if "msa" in configs["data"] and configs["data"]["msa"]["enable"]:
        logger.info("MSA cache root: %s", Path(msa_cache_path).expanduser().resolve())
        msa_info = configs["data"]["msa"]
        msa_args = deepcopy(msa_info)

        if "msa" in (dataset_config := configs["data"][dataset_name]):
            for k, v in dataset_config["msa"].items():
                if k not in ["prot", "rna"]:
                    msa_args[k] = v
                else:
                    for kk, vv in dataset_config["msa"][k].items():
                        msa_args[k][kk] = vv

        prot_msa_args = msa_args["prot"]
        prot_msa_args.update(
            {
                "dataset_name": dataset_name,
                "msa_cache_path": msa_cache_path,
                "merge_method": msa_args["merge_method"],
                "max_size": msa_args["max_size"][stage],
            }
        )

        rna_msa_args = msa_args["rna"]
        rna_msa_args.update(
            {
                "dataset_name": dataset_name,
                "merge_method": msa_args["merge_method"],
                "max_size": msa_args["max_size"][stage],
            }
        )

        return MSAFeaturizer(
            prot_msa_args=prot_msa_args,
            rna_msa_args=rna_msa_args,
            enable_rna_msa=configs.data.msa.enable_rna_msa,
        )

    else:
        return None


# class WeightedMultiDatasetMD(Dataset):
#     """
#     Merges multiple datasets based on a specified repetition weight for each sample
#     within each dataset.

#     How it works:
#     For the i-th dataset in the `datasets` list, `datasets[i]`, all of its samples
#     will be duplicated `dataset_sample_weights[i]` times. Finally, all the duplicated
#     sample indices are concatenated and randomly shuffled to form a new, unified dataset.

#     Args:
#         datasets (List[Dataset]): A list of PyTorch Dataset objects.
#         dataset_names (List[str]): A list of dataset names for debugging and logging.
#         dataset_sample_weights (List[int]): A list of integers, where each integer
#                                            corresponds to a dataset and specifies how
#                                            many times each of its samples should be
#                                            duplicated.
#     """
#     def __init__(self, 
#                  datasets: list[Dataset], 
#                  dataset_names: list[str], 
#                  dataset_sample_weights: list[int]):
        
#         # Validate that the input lists have the same length
#         if not (len(datasets) == len(dataset_names) == len(dataset_sample_weights)):
#             raise ValueError("The lengths of datasets, dataset_names, and dataset_sample_weights must be the same.")

#         self.datasets = datasets
#         self.dataset_names = dataset_names
#         self.weights = dataset_sample_weights
        
#         # self.indices will store pointers to the original samples as (dataset_index, sample_index)
#         self.indices = []

#         print("--- WeightedMultiDatasetMD Initialization ---")
#         total_new_samples = 0

#         # Iterate over each dataset to build the index map
#         for i, dataset in enumerate(self.datasets):
#             weight = self.weights[i]
#             if not isinstance(weight, int) or weight < 0:
#                 raise TypeError(f"Weight must be a non-negative integer, but dataset '{self.dataset_names[i]}' has a weight of {weight}.")
            
#             num_original_samples = len(dataset)
#             num_new_samples = num_original_samples * weight
#             total_new_samples += num_new_samples

#             print(f"  - Dataset '{self.dataset_names[i]}':")
#             print(f"    Original size: {num_original_samples}, Weight per sample: {weight}")
#             print(f"    Generating {num_new_samples} new sample indices.")

#             # For each sample in this dataset, add its index 'weight' times
#             for sample_idx in range(num_original_samples):
#                 self.indices.extend([(i, sample_idx)] * weight)
        
#         # Shuffle all indices to ensure the data is well-mixed
#         random.seed(66)
#         random.shuffle(self.indices)
        
#         print(f"------------------------------------")
#         print(f"Initialization complete. Total size of the new dataset: {len(self.indices)}")
#         assert len(self.indices) == total_new_samples

#     def __len__(self):
#         """Returns the total length of the new dataset."""
#         return len(self.indices)

#     def __getitem__(self, idx: int):
#         """Retrieves a data sample based on the index."""
#         # Look up the pre-built list of indices
#         dataset_idx, sample_idx = self.indices[idx]
        
#         # Fetch the data from the corresponding original dataset
#         return self.datasets[dataset_idx][sample_idx]






class WeightedMultiDataset(Dataset):
    """
    A weighted dataset composed of multiple datasets with weights.
    """

    def __init__(
        self,
        datasets: list[Dataset],
        dataset_names: list[str],
        datapoint_weights: list[list[float]],
        dataset_sample_weights: list[torch.tensor],
    ):
        """
        Initializes the WeightedMultiDataset.
        Args:
            datasets: A list of Dataset objects.
            dataset_names: A list of dataset names corresponding to the datasets.
            datapoint_weights: A list of lists containing sampling weights for each datapoint in the datasets.
            dataset_sample_weights: A list of torch tensors containing sampling weights for each dataset.
        """
        self.datasets = datasets
        self.dataset_names = dataset_names
        self.datapoint_weights = datapoint_weights
        self.dataset_sample_weights = torch.Tensor(dataset_sample_weights)
        self.iteration = 0
        self.offset = 0
        self.init_datasets()

    def init_datasets(self):
        """Calculate global weights of each datapoint in datasets for future sampling."""
        self.merged_datapoint_weights = []
        self.weight = 0.0
        self.dataset_indices = []
        self.within_dataset_indices = []
        print("--- WeightedMultiDataset Initialization ---")

        for dataset_index, (
            dataset,
            dataset_name,
            datapoint_weight_list,
            dataset_weight,
        ) in enumerate(
            zip(self.datasets, self.dataset_names, self.datapoint_weights, self.dataset_sample_weights)
        ):
            # normalize each dataset weights
            weight_sum = sum(datapoint_weight_list)
            datapoint_weight_list = [
                dataset_weight * w / weight_sum for w in datapoint_weight_list
            ]
            print(f"  - Dataset '{dataset_name}':")
            print(f"    Dataset size: {len(dataset)}, Dataset Weight: {dataset_weight}")
            self.merged_datapoint_weights.extend(datapoint_weight_list)
            self.weight += dataset_weight
            self.dataset_indices.extend([dataset_index] * len(datapoint_weight_list))
            self.within_dataset_indices.extend(list(range(len(datapoint_weight_list))))
        self.merged_datapoint_weights = torch.tensor(
            self.merged_datapoint_weights, dtype=torch.float64
        )

    def __len__(self) -> int:
        return len(self.merged_datapoint_weights)

    def __getitem__(self, index: int) -> dict[str, dict]:
        return self.datasets[self.dataset_indices[index]][
            self.within_dataset_indices[index]
        ]


def get_weighted_pdb_weight(
    data_type: str,
    cluster_size: int,
    chain_count: dict,
    eps: float = 1e-9,
    beta_dict: Optional[dict] = None,
    alpha_dict: Optional[dict] = None,
) -> float:
    """
    Get sample weight for each example in a weighted PDB dataset.

        data_type (str): Type of data, either 'chain' or 'interface'.
        cluster_size (int): Cluster size of this chain/interface.
        chain_count (dict): Count of each kind of chains, e.g., {"prot": int, "nuc": int, "ligand": int}.
        eps (float, optional): A small epsilon value to avoid division by zero. Default is 1e-9.
        beta_dict (Optional[dict], optional): Dictionary containing beta values for 'chain' and 'interface'.
        alpha_dict (Optional[dict], optional): Dictionary containing alpha values for different chain types.

    Returns:
         float: Calculated weight for the given chain/interface.
    """
    if not beta_dict:
        beta_dict = {
            "chain": 0.5,
            "interface": 1,
        }
    if not alpha_dict:
        alpha_dict = {
            "prot": 3,
            "nuc": 3,
            "ligand": 1,
        }

    assert cluster_size > 0
    assert data_type in ["chain", "interface"]
    beta = beta_dict[data_type]
    assert set(chain_count.keys()).issubset(set(alpha_dict.keys()))
    weight = (
        beta
        * sum(
            [alpha * chain_count[data_mode] for data_mode, alpha in alpha_dict.items()]
        )
        / (cluster_size + eps)
    )
    return weight


def calc_weights_for_df(
    indices_df: pd.DataFrame, beta_dict: dict[str, Any], alpha_dict: dict[str, Any]
) -> pd.DataFrame:
    """
    Calculate weights for each example in the dataframe.

    Args:
        indices_df: A pandas DataFrame containing the indices.
        beta_dict: A dictionary containing beta values for different data types.
        alpha_dict: A dictionary containing alpha values for different data types.

    Returns:
        A pandas DataFrame with an column 'weights' containing the calculated weights.
    """
    # Specific to assembly, and entities (chain or interface)
    indices_df["pdb_sorted_entity_id"] = indices_df.apply(
        lambda x: f"{x['pdb_id']}_{x['assembly_id']}_{'_'.join(sorted([str(x['entity_1_id']), str(x['entity_2_id'])]))}",
        axis=1,
    )

    entity_member_num_dict = {}
    for pdb_sorted_entity_id, sub_df in indices_df.groupby("pdb_sorted_entity_id"):
        # Number of repeatative entities in the same assembly
        entity_member_num_dict[pdb_sorted_entity_id] = len(sub_df)
    indices_df["pdb_sorted_entity_id_member_num"] = indices_df.apply(
        lambda x: entity_member_num_dict[x["pdb_sorted_entity_id"]], axis=1
    )

    cluster_size_record = {}
    for cluster_id, sub_df in indices_df.groupby("cluster_id"):
        cluster_size_record[cluster_id] = len(set(sub_df["pdb_sorted_entity_id"]))

    weights = []
    for _, row in indices_df.iterrows():
        data_type = row["type"]
        cluster_size = cluster_size_record[row["cluster_id"]]
        chain_count = {"prot": 0, "nuc": 0, "ligand": 0}
        for mol_type in [row["mol_1_type"], row["mol_2_type"]]:
            if chain_count.get(mol_type) is None:
                continue
            chain_count[mol_type] += 1
        # Weight specific to (assembly, entity(chain/interface))
        weight = get_weighted_pdb_weight(
            data_type=data_type,
            cluster_size=cluster_size,
            chain_count=chain_count,
            beta_dict=beta_dict,
            alpha_dict=alpha_dict,
        )
        weights.append(weight)
    indices_df["weights"] = weights / indices_df["pdb_sorted_entity_id_member_num"]
    return indices_df


def get_sample_weights(
    sampler_type: str,
    indices_df: pd.DataFrame = None,
    beta_dict: dict = {
        "chain": 0.5,
        "interface": 1,
    },
    alpha_dict: dict = {
        "prot": 3,
        "nuc": 3,
        "ligand": 1,
    },
    force_recompute_weight: bool = False,
) -> Union[pd.Series, list[float]]:
    """
    Computes sample weights based on the specified sampler type.

    Args:
        sampler_type: The type of sampler to use ('weighted' or 'uniform').
        indices_df: A pandas DataFrame containing the indices.
        beta_dict: A dictionary containing beta values for different data types.
        alpha_dict: A dictionary containing alpha values for different data types.
        force_recompute_weight: Whether to force recomputation of weights even if they already exist.

    Returns:
        A list of sample weights.

    Raises:
        ValueError: If an unknown sampler type is provided.
    """
    if sampler_type == "weighted":
        assert indices_df is not None
        if "weights" not in indices_df.columns or force_recompute_weight:
            indices_df = calc_weights_for_df(
                indices_df=indices_df,
                beta_dict=beta_dict,
                alpha_dict=alpha_dict,
            )
        return indices_df["weights"].astype("float32")
    elif sampler_type == "uniform":
        assert indices_df is not None
        return [1 / len(indices_df) for _ in range(len(indices_df))]
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

def get_datasets(
    configs: ConfigDict, error_dir: Optional[str]
) -> tuple[BaseSingleDataset, dict[str, BaseSingleDataset]]:
    """
    Get training and testing datasets given configs

    Args:
        configs: A ConfigDict containing the dataset configurations.
        error_dir: The directory where error logs will be saved.

    Returns:
        A tuple containing the training dataset and a dictionary of testing datasets.
    """

    def _get_dataset_param(config_dict, dataset_name: str, stage: str):
        # Template_featurizer is under development
        # Lig_atom_rename/shuffle_mols/shuffle_sym_ids do not affect the performance very much
        return {
            "name": dataset_name,
            **config_dict["base_info"],
            "cropping_configs": config_dict["cropping_configs"],
            "error_dir": error_dir,
            "msa_featurizer": get_msa_featurizer(configs, dataset_name, stage, config_dict["msa_cache_path"]),
            "template_featurizer": None,
            "lig_atom_rename": config_dict.get("lig_atom_rename", False),
            "shuffle_mols": config_dict.get("shuffle_mols", False),
            "shuffle_sym_ids": config_dict.get("shuffle_sym_ids", False),
            # MSM / TICA dynamics loss keys (top-level config, not in base_info)
            "msm_cache_dir": config_dict.get("msm_cache_dir"),
            "msm_build_missing": config_dict.get("msm_build_missing", True),
            "n_coarse_states": config_dict.get("n_coarse_states", 10),
            "n_sampling_states": config_dict.get("n_sampling_states", config_dict.get("n_coarse_states", 10)),
            "msm_configs": config_dict.get("msm_configs", {}),
        }

    data_config = configs.data
    logger.info(f"Using train sets {data_config.train_sets}")
    if len(data_config.train_sets) != len(
        data_config.train_sampler.train_sample_weights
    ):
        data_config.train_sampler.train_sample_weights = [1.]*len(data_config.train_sets)

    train_datasets = []
    datapoint_weights = []
    for train_name in data_config.train_sets:
        config_dict = data_config[train_name].to_dict()
        dataset_param = _get_dataset_param(
            config_dict, dataset_name=train_name, stage="train"
        )
        dataset_param["ref_pos_augment"] = data_config.get(
            "train_ref_pos_augment", True
        )
        dataset_param["limits"] = data_config.get("limits", -1)
        dataset_param["split_id"] = configs.split_id
        dataset_param["total_split"] = configs.total_split
        dataset_param["precomputed_emb_dir"] = data_config[train_name].get("precomputed_emb_dir", None) # load universal embedding across dataset
        dataset_param["token_crop_size"] = configs.train_crop_size
        dataset_param["dump_embeddings"] = configs.get("dump_embeddings", False)
        dataset_param["debug"] = configs.get("debug", False)
        dataset_param["interval_fpath"] = data_config[train_name].get("interval_fpath", None)
        print(train_name, dataset_param["precomputed_emb_dir"], dataset_param["interval_fpath"])

        if "unbinding" in train_name.lower():
            train_dataset = UnbindingSingleDataset(**dataset_param)
        elif "misato" in train_name.lower():
            train_dataset = MisatoSingleDataset(**dataset_param)
        elif "mdposit" in train_name.lower():
            train_dataset = MDpositSingleDataset(**dataset_param)
        elif "atlas" in train_name.lower():
            train_dataset = AtlasSingleDataset(**dataset_param)
        elif train_name.upper().startswith("MSR-"):
            train_dataset = MSRSingleDataset(**dataset_param)
        else:
            continue

        train_datasets.append(train_dataset)
        datapoint_weights.append(
            [1 / len(train_dataset) for _ in range(len(train_dataset))]
        )

    if len(train_datasets) >= 1:
        # train_sample_weights = data_config.train_sampler.get("train_sample_weights", [1.] * len(train_datasets))
        # sample_weights = [float(weight) / len(dataset) for weight, dataset in zip(train_sample_weights, train_datasets)]
        # sample_weights = [math.ceil(x / np.min(sample_weights)) for x in sample_weights]
        # train_dataset = WeightedMultiDatasetMD(
        #     datasets=train_datasets,
        #     dataset_names=data_config.train_sets,
        #     dataset_sample_weights=sample_weights,
        # )
        train_dataset = WeightedMultiDataset(
            datasets=train_datasets,
            dataset_names=data_config.train_sets,
            datapoint_weights=datapoint_weights,
            dataset_sample_weights=data_config.train_sampler.train_sample_weights,
        )
    else:
        train_dataset = EmptyDataset()

    test_datasets = {}
    test_sets = data_config.test_sets
    for test_name in test_sets:
        config_dict = data_config[test_name].to_dict()
        dataset_param = _get_dataset_param(
            config_dict, dataset_name=test_name, stage="test"
        )
        dataset_param["ref_pos_augment"] = data_config.get("test_ref_pos_augment", True)
        dataset_param["precomputed_emb_dir"] = data_config[test_name].get("precomputed_emb_dir", None)
        dataset_param["split_id"] = configs.split_id
        dataset_param["total_split"] = configs.total_split
        dataset_param["token_crop_size"] = configs.get("inference_crop_size", configs.train_crop_size)
        dataset_param["dump_embeddings"] = configs.get("dump_embeddings", False)
        dataset_param["debug"] = configs.get("debug", False)
        dataset_param["interval_fpath"] = data_config[test_name].get("interval_fpath", None)
        print(test_name, dataset_param["precomputed_emb_dir"], dataset_param["interval_fpath"])

        if "unbinding" in test_name.lower():
            test_dataset = UnbindingSingleDataset(**dataset_param)
        elif "misato" in test_name.lower():
            if "test" in test_name.lower() or "val" in test_name.lower():
                test_dataset = MisatoSingleDataset(**dataset_param, downsample=64)
            else:
                test_dataset = MisatoSingleDataset(**dataset_param)
        elif "mdposit" in test_name.lower():
            test_dataset = MDpositSingleDataset(**dataset_param)
        elif "atlas" in test_name.lower():
            test_dataset = AtlasSingleDataset(**dataset_param)
        elif "example" in test_name.lower() or "bioemu" in test_name.lower():
            test_dataset = ExampleDataset(**dataset_param)
        elif test_name.upper().startswith("MSR-"):
            test_dataset = MSRSingleDataset(**dataset_param, downsample=200)
        else:
            # InferenceDataset 需要不同的参数
            if "inference" in test_name.lower():
                dataset_param["time_interval"] = configs.get("time_interval", False)
                test_dataset = InferenceDataset(
                    input_file=configs.get("input_file", None),
                    output_dir=configs.get("dump_dir", None),
                    **dataset_param,
                )
            else:
                raise ValueError(f"Unknown dataset type: {test_name}")
            
        test_datasets[test_name] = test_dataset

    return train_dataset, test_datasets
