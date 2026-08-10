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

import logging
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from protenix.metrics.rmsd import weighted_rigid_align
from protenix.model.modules.frames import (
    expressCoordinatesInFrame,
    gather_frame_atom_by_indices,
)
from protenix.model.tica_dynamics_loss import TICADynamicsLoss
from protenix.model.utils import expand_at_dim
from protenix.openfold_local.utils.checkpointing import get_checkpoint_fn
from protenix.utils.torch_utils import cdist


def loss_reduction(loss: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """reduction wrapper

    Args:
        loss (torch.Tensor): loss
            [...]
        method (str, optional): reduction method. Defaults to "mean".

    Returns:
        torch.Tensor: reduced loss
            [] or [...]
    """

    if method is None:
        return loss
    assert method in ["mean", "sum", "add", "max", "min"]
    if method == "add":
        method = "sum"
    return getattr(torch, method)(loss)


class SmoothLDDTLoss(nn.Module):
    """
    Implements Algorithm 27 [SmoothLDDTLoss] in AF3
    """

    def __init__(
        self,
        eps: float = 1e-10,
        reduction: str = "mean",
        pp_weight: float = 1.0,
        pl_weight: float = 5.0,
        ll_weight: float = 10.0,
    ) -> None:
        """SmoothLDDTLoss

        Args:
            eps (float, optional): avoid nan. Defaults to 1e-10.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super(SmoothLDDTLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.pp_weight = pp_weight
        self.pl_weight = pl_weight
        self.ll_weight = ll_weight

    def _chunk_forward(self, pred_distance, true_distance, c_lm=None):
        dist_diff = torch.abs(pred_distance - true_distance)
        # For save cuda memory we use inplace op
        dist_diff_epsilon = 0
        for threshold in [0.5, 1, 2, 4]:
            dist_diff_epsilon += 0.25 * torch.sigmoid(threshold - dist_diff)

        # Compute mean
        if c_lm is not None:
            lddt = torch.sum(c_lm * dist_diff_epsilon, dim=(-1, -2)) / (
                torch.sum(c_lm, dim=(-1, -2)) + self.eps
            )  # [..., N_sample]
        else:
            # It's for sparse forward mode
            lddt = torch.mean(dist_diff_epsilon, dim=-1)
        return lddt


    def dense_forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        is_ligand: torch.Tensor,
        lddt_mask: torch.Tensor,
        diffusion_chunk_size: Optional[int] = None,
        loss_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """SmoothLDDTLoss sparse implementation

        Args:
            pred_coordinate (torch.Tensor): the diffusion denoised atom coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [..., N_sample, N_atom, 3]
            lddt_mask (torch.Tensor, optional): whether true distance is within radius (30A for nuc and 15A for others)
                [N_atom, N_atom]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the smooth lddt loss
                [...] if reduction is None else []
        """
        c_lm = lddt_mask.float().unsqueeze(dim=-3).detach()  # [..., 1, N_atom, N_atom]
        # prot_ligand: 5, ligand_ligand: 10, prot_prot: 1
        prot_ligand_mask = torch.logical_xor(is_ligand.unsqueeze(dim=-2) == 1, is_ligand.unsqueeze(dim=-1) == 1)
        ligand_ligand_mask = (is_ligand.unsqueeze(dim=-2) == 1) & (is_ligand.unsqueeze(dim=-1) == 1)
        prot_prot_mask = (is_ligand.unsqueeze(dim=-2) == 0) & (is_ligand.unsqueeze(dim=-1) == 0)
        weight = prot_ligand_mask * self.pl_weight + ligand_ligand_mask * self.ll_weight + prot_prot_mask * self.pp_weight
        c_lm = c_lm * weight

        # Compute distance error
        # [...,  N_sample , N_atom, N_atom]
        true_distance = torch.cdist(true_coordinate, true_coordinate)

        pred_distance = torch.cdist(pred_coordinate, pred_coordinate)
        lddt = self._chunk_forward(pred_distance=pred_distance, true_distance=true_distance, c_lm=c_lm)
        lddt_loss = 1 - lddt
        
        if loss_mask is not None:
            lddt_loss = lddt_loss * loss_mask
        lddt_loss = lddt_loss.mean()
        return lddt_loss


class BondLoss(nn.Module):
    """
    Implements Formula 5 [BondLoss] in AF3
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        """BondLoss

        Args:
            eps (float, optional): avoid nan. Defaults to 1e-6.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super(BondLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def _chunk_forward(self, pred_distance, true_distance, bond_mask):
        # Distance squared error
        # [...,  N_sample , N_atom, N_atom]
        dist_squared_err = (pred_distance - true_distance.unsqueeze(dim=-3)) ** 2
        bond_loss = torch.sum(dist_squared_err * bond_mask, dim=(-1, -2)) / torch.sum(
            bond_mask + self.eps, dim=(-1, -2)
        )  # [..., N_sample]
        return bond_loss

    def forward(
        self,
        pred_distance: torch.Tensor,
        true_distance: torch.Tensor,
        distance_mask: torch.Tensor,
        bond_mask: torch.Tensor,
        per_sample_scale: torch.Tensor = None,
        diffusion_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """BondLoss

        Args:
            pred_distance (torch.Tensor): the diffusion denoised atom-atom distance
                [..., N_sample, N_atom, N_atom]
            true_distance (torch.Tensor): the ground truth coordinates
                [..., N_atom, N_atom]
            distance_mask (torch.Tensor): whether true coordinates exist.
                [N_atom, N_atom] or [..., N_atom, N_atom]
            bond_mask (torch.Tensor): bonds considered in this loss
                [N_atom, N_atom] or [..., N_atom, N_atom]
            per_sample_scale (torch.Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the bond loss
                [...] if reduction is None else []
        """

        bond_mask = (bond_mask * distance_mask).unsqueeze(
            dim=-3
        )  # [1, N_atom, N_atom] or [..., 1, N_atom, N_atom]
        # Bond Loss
        if diffusion_chunk_size is None:
            bond_loss = self._chunk_forward(
                pred_distance=pred_distance,
                true_distance=true_distance,
                bond_mask=bond_mask,
            )
        else:
            checkpoint_fn = get_checkpoint_fn()
            bond_loss = []
            N_sample = pred_distance.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            for i in range(no_chunks):
                bond_loss_i = checkpoint_fn(
                    self._chunk_forward,
                    pred_distance[
                        ...,
                        i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                        :,
                        :,
                    ],
                    true_distance,
                    bond_mask,
                )
                bond_loss.append(bond_loss_i)
            bond_loss = torch.cat(bond_loss, dim=-1)
        if per_sample_scale is not None:
            bond_loss = bond_loss * per_sample_scale

        bond_loss = bond_loss.mean(dim=-1)  # [...]
        return loss_reduction(bond_loss, method=self.reduction)

    def sparse_forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        distance_mask: torch.Tensor,
        bond_mask: torch.Tensor,
        per_sample_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        """BondLoss sparse implementation

        Args:
            pred_coordinate (torch.Tensor): the diffusion denoised atom coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [..., N_sample, N_atom, 3]
            distance_mask (torch.Tensor): whether true coordinates exist.
                [N_atom, N_atom] or [..., N_atom, N_atom]
            bond_mask (torch.Tensor): bonds considered in this loss
                [N_atom, N_atom] or [..., N_atom, N_atom]
            per_sample_scale (torch.Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample]
        Returns:
            torch.Tensor: the bond loss
                [...] if reduction is None else []
        """
        bond_mask = bond_mask * distance_mask
        bond_indices = torch.nonzero(bond_mask, as_tuple=True)
        pred_coords_i = pred_coordinate.index_select(-2, bond_indices[0])
        pred_coords_j = pred_coordinate.index_select(-2, bond_indices[1])
        true_coords_i = true_coordinate.index_select(-2, bond_indices[0])
        true_coords_j = true_coordinate.index_select(-2, bond_indices[1])

        pred_distance_sparse = torch.norm(pred_coords_i - pred_coords_j, p=2, dim=-1)
        true_distance_sparse = torch.norm(true_coords_i - true_coords_j, p=2, dim=-1)
        dist_squared_err_sparse = (pred_distance_sparse - true_distance_sparse) ** 2
        # Protecting special data that has size: tensor([], size=(x, 0), grad_fn=<PowBackward0>)
        if dist_squared_err_sparse.numel() == 0:
            return torch.tensor(
                0.0, device=dist_squared_err_sparse.device, requires_grad=True
            )
        bond_loss = torch.mean(dist_squared_err_sparse, dim=-1)  # [N_frame, N_sample]
        if per_sample_scale is not None:
            bond_loss = bond_loss * per_sample_scale

        bond_loss = bond_loss.mean()
        return bond_loss


def compute_lddt_mask(
    true_distance: torch.Tensor,
    distance_mask: torch.Tensor,
    is_nucleotide: torch.Tensor,
    is_nucleotide_threshold: float = 30.0,
    is_not_nucleotide_threshold: float = 15.0,
) -> torch.Tensor:
    """calculate the atom pair mask with the bespoke radius

    Args:
        true_distance (torch.Tensor): the ground truth coordinates
            [N_frame, N_atom, N_atom]
        distance_mask (torch.Tensor): whether true coordinates exist.
            [N_atom, N_atom]
        is_nucleotide (torch.Tensor): Indicator for nucleotide atoms.
            [N_atom]
        is_nucleotide_threshold (float): Threshold distance for nucleotide atoms. Defaults to 30.0.
        is_not_nucleotide_threshold (float): Threshold distance for non-nucleotide atoms. Defaults to 15.0.

    Returns:
        c_lm (torch.Tenson): the atom pair mask c_lm, not symmetric
            [N_frame, N_atom, N_atom]
    """
    # Restrict to bespoke inclusion radius
    is_nucleotide_mask = is_nucleotide.bool()
    c_lm = (true_distance < is_nucleotide_threshold) * is_nucleotide_mask[None, :, None] + (
        true_distance < is_not_nucleotide_threshold
    ) * (
        ~is_nucleotide_mask[None, :, None]
    )  # [..., N_atom, N_atom]

    # Zero-out diagonals of c_lm and cast to float
    c_lm = c_lm * (
        1 - torch.eye(n=c_lm.size(-1), device=c_lm.device, dtype=true_distance.dtype)
    )[None, ...]
    # Zero-out atom pairs without true coordinates
    # Note: the sparsity of c_lm is ~10% in 5000 atom-pairs,
    # and becomes more sparse as the number of atoms increases,
    # change to sparse implementation can reduce cuda memory
    c_lm = c_lm * distance_mask[None, ...]  # [..., N_atom, N_atom]
    return c_lm


def softmax_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Softmax cross entropy

    Args:
        logits (torch.Tensor): classification logits
            [..., num_class]
        labels (torch.Tensor): classification labels (value = probability)
            [..., num_class]

    Returns:
        torch.Tensor: softmax cross entropy
            [...]
    """
    loss = -1 * torch.sum(
        labels * F.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


class DistogramLoss(nn.Module):
    """
    Implements DistogramLoss in AF3
    """

    def __init__(
        self,
        min_bin: float = 2.3125,
        max_bin: float = 21.6875,
        no_bins: int = 64,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        """Distogram loss
        This head and loss are identical to AlphaFold 2, where the pairwise token distances use the representative atom for each token:
            Cβ for protein residues (Cα for glycine),
            C4 for purines and C2 for pyrimidines.
            All ligands already have a single atom per token.

        Args:
            min_bin (float, optional): min boundary of bins. Defaults to 2.3125.
            max_bin (float, optional): max boundary of bins. Defaults to 21.6875.
            no_bins (int, optional): number of bins. Defaults to 64.
            eps (float, optional): small number added to denominator. Defaults to 1e-6.
            reduce (bool, optional): reduce dim. Defaults to True.
        """
        super(DistogramLoss, self).__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.eps = eps
        self.reduction = reduction

    def calculate_label(
        self,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        rep_atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """calculate the label as bins

        Args:
            true_coordinate (torch.Tensor): true coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist.
                [N_atom] or [..., N_atom]
            rep_atom_mask (torch.Tensor): representative atom mask
                [N_atom]

        Returns:
            true_bins (torch.Tensor): distance error assigned into bins (one-hot).
                [..., N_token, N_token, no_bins]
            pair_coordinate_mask (torch.Tensor): whether the coordinates of representative atom pairs exist.
                [N_token, N_token] or [..., N_token, N_token]
        """

        boundaries = torch.linspace(
            start=self.min_bin,
            end=self.max_bin,
            steps=self.no_bins - 1,
            device=true_coordinate.device,
        )

        # Compute label: the true bins
        # True distance
        rep_atom_mask = rep_atom_mask.bool()
        true_coordinate = true_coordinate[..., rep_atom_mask, :]  # [..., N_token, 3]
        gt_dist = cdist(true_coordinate, true_coordinate)  # [..., N_token, N_token]
        # Assign distance to bins
        true_bins = torch.sum(
            gt_dist.unsqueeze(dim=-1) > boundaries, dim=-1
        )  # range in [0, no_bins-1], shape = [..., N_token, N_token]

        # Mask
        token_mask = coordinate_mask[..., rep_atom_mask]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        return F.one_hot(true_bins, self.no_bins), pair_mask

    def forward(
        self,
        logits: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        rep_atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Distogram loss

        Args:
            logits (torch.Tensor): logits.
                [..., N_token, N_token, no_bins]
            true_coordinate (torch.Tensor): true coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist.
                [N_atom] or [..., N_atom]
            rep_atom_mask (torch.Tensor): representative atom mask.
                [N_atom]

        Returns:
            torch.Tensor: the return loss.
                [...] if self.reduction is not None else []
        """

        with torch.no_grad():
            true_bins, pair_mask = self.calculate_label(
                true_coordinate=true_coordinate,
                coordinate_mask=coordinate_mask,
                rep_atom_mask=rep_atom_mask,
            )

        errors = softmax_cross_entropy(
            logits=logits,
            labels=true_bins,
        )  # [..., N_token, N_token]

        denom = self.eps + torch.sum(pair_mask, dim=(-1, -2))
        loss = torch.sum(errors * pair_mask, dim=(-1, -2))
        loss = loss / denom

        return loss_reduction(loss, method=self.reduction)


class PDELoss(nn.Module):
    """
    Implements Predicted distance loss in AF3
    """

    def __init__(
        self,
        min_bin: float = 0,
        max_bin: float = 32,
        no_bins: int = 64,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        """PDELoss
        This loss are between representative token atoms i and j in the mini-rollout prediction

        Args:
            min_bin (float, optional): min boundary of bins. Defaults to 0.
            max_bin (float, optional): max boundary of bins. Defaults to 32.
            no_bins (int, optional): number of bins. Defaults to 64.
            eps (float, optional): small number added to denominator. Defaults to 1e-6.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super(PDELoss, self).__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.eps = eps
        self.reduction = reduction

    def calculate_label(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        rep_atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """calculate the label as bins

        Args:
            pred_coordinate (torch.Tensor): predicted coordinates.
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): true coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist.
                [N_atom] or [..., N_atom]
            rep_atom_mask (torch.Tensor):
                [N_atom]

        Returns:
            true_bins (torch.Tensor): distance error assigned into bins (one-hot).
                [..., N_sample, N_token, N_token, no_bins]
            pair_coordinate_mask (torch.Tensor): whether the coordinates of representative atom pairs exist.
                [N_token, N_token] or [..., N_token, N_token]
        """

        boundaries = torch.linspace(
            start=self.min_bin,
            end=self.max_bin,
            steps=self.no_bins + 1,
            device=pred_coordinate.device,
        )

        # Compute label: the true bins
        # True distance
        rep_atom_mask = rep_atom_mask.bool()
        true_coordinate = true_coordinate[..., rep_atom_mask, :]  # [..., N_token, 3]
        gt_dist = cdist(true_coordinate, true_coordinate)  # [..., N_token, N_token]
        # Predicted distance
        pred_coordinate = pred_coordinate[..., rep_atom_mask, :]
        pred_dist = cdist(
            pred_coordinate, pred_coordinate
        )  # [..., N_sample, N_token, N_token]
        # Distance error
        dist_error = torch.abs(pred_dist - gt_dist.unsqueeze(dim=-3))

        # Assign distance error to bins
        true_bins = torch.sum(
            dist_error.unsqueeze(dim=-1) > boundaries, dim=-1
        )  # range in [1, no_bins + 1], shape = [..., N_sample, N_token, N_token]
        true_bins = torch.clamp(
            true_bins, min=1, max=self.no_bins
        )  # just in case bin=0 occurs

        # Mask
        token_mask = coordinate_mask[..., rep_atom_mask]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        return F.one_hot(true_bins - 1, self.no_bins).detach(), pair_mask.detach()

    def forward(
        self,
        logits: torch.Tensor,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        rep_atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """PDELoss

        Args:
            logits (torch.Tensor): logits
                [..., N_sample, N_token, N_token, no_bins]
            pred_coordinate: (torch.Tensor): predict coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): true coordinates
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom] or [..., N_atom]
            rep_atom_mask (torch.Tensor): representative atom mask for this loss
                [N_atom]

        Returns:
            torch.Tensor: the return loss
                [...] if reduction is None else []
        """

        with torch.no_grad():
            true_bins, pair_mask = self.calculate_label(
                pred_coordinate=pred_coordinate,
                true_coordinate=true_coordinate,
                coordinate_mask=coordinate_mask,
                rep_atom_mask=rep_atom_mask,
            )

        errors = softmax_cross_entropy(
            logits=logits,
            labels=true_bins,
        )  # [..., N_sample, N_token, N_token]

        denom = self.eps + torch.sum(pair_mask, dim=(-1, -2))  # [...]
        loss = errors * pair_mask.unsqueeze(dim=-3)  # [..., N_sample, N_token, N_token]
        loss = torch.sum(loss, dim=(-1, -2))  # [..., N_sample]
        loss = loss / denom.unsqueeze(dim=-1)  # [..., N_sample]
        loss = loss.mean(dim=-1)  # [...]

        return loss_reduction(loss, method=self.reduction)


# Algorithm 30 Compute alignment error
def compute_alignment_error_squared(
    pred_coordinate: torch.Tensor,
    true_coordinate: torch.Tensor,
    pred_frames: torch.Tensor,
    true_frames: torch.Tensor,
) -> torch.Tensor:
    """Implements Algorithm 30 Compute alignment error, but do not take the square root

    Args:
        pred_coordinate (torch.Tensor): the predict coords [frame center]
            [..., N_sample, N_token, 3]
        true_coordinate (torch.Tensor): the ground truth coords [frame center]
            [..., N_token, 3]
        pred_frames (torch.Tensor): the predict frame
            [..., N_sample, N_frame, 3, 3]
        true_frames (torch.Tensor): the ground truth frame
            [..., N_frame, 3, 3]

    Returns:
        torch.Tensor: the computed alignment error
            [..., N_sample, N_frame, N_token]
    """
    x_transformed_pred = expressCoordinatesInFrame(
        coordinate=pred_coordinate, frames=pred_frames
    )  # [..., N_sample, N_frame, N_token, 3]
    x_transformed_true = expressCoordinatesInFrame(
        coordinate=true_coordinate, frames=true_frames
    )  # [..., N_frame, N_token, 3]
    squared_pae = torch.sum(
        (x_transformed_pred - x_transformed_true.unsqueeze(dim=-4)) ** 2, dim=-1
    )  # [..., N_sample, N_frame, N_token]
    return squared_pae


class PAELoss(nn.Module):
    """
    Implements Predicted Aligned distance loss in AF3
    """

    def __init__(
        self,
        min_bin: float = 0,
        max_bin: float = 32,
        no_bins: int = 64,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        """PAELoss
        This loss are between representative token atoms i and j in the mini-rollout prediction

        Args:
            min_bin (float, optional): min boundary of bins. Defaults to 0.
            max_bin (float, optional): max boundary of bins. Defaults to 32.
            no_bins (int, optional): number of bins. Defaults to 64.
            eps (float, optional): small number added to denominator. Defaults to 1e-6.
            reduce (bool, optional): reduce dim. Defaults to True.
        """
        super(PAELoss, self).__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.eps = eps
        self.reduction = reduction

    def calculate_label(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        rep_atom_mask: torch.Tensor,
        frame_atom_index: torch.Tensor,
        has_frame: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """calculate true PAE (squared) and true bins

        Args:
            pred_coordinate: (torch.Tensor): predict coordinates.
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): true coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom]
            rep_atom_mask (torch.Tensor): masks of the representative atom for each token.
                [N_atom]
            frame_atom_index (torch.Tensor): indices of frame atoms (three atoms per token(=per frame)).
                [N_token, 3[three atom]]
            has_frame (torch.Tensor): indicates whether token_i has a valid frame.
                [N_token]
        Returns:
            squared_pae (torch.Tensor): pairwise alignment error squared
                [..., N_sample, N_frame, N_token] where N_token = rep_atom_mask.sum()
            true_bins (torch.Tensor): the true bins
                [..., N_sample, N_frame, N_token, no_bins]
            frame_token_pair_mask (torch.Tensor): whether frame_i token_j both have true coordinates.
                [N_frame, N_token]
        """

        coordinate_mask = coordinate_mask.bool()
        rep_atom_mask = rep_atom_mask.bool()
        has_frame = has_frame.bool()

        # NOTE: to support frame_atom_index with batch_dims, need to expand its dims before constructing frames.
        assert len(frame_atom_index.shape) == 2

        # Take valid frames: N_token -> N_frame
        frame_atom_index = frame_atom_index[has_frame, :]  # [N_frame, 3[three atom]]

        # Get predicted frames and true frames
        pred_frames = gather_frame_atom_by_indices(
            coordinate=pred_coordinate, frame_atom_index=frame_atom_index, dim=-2
        )  # [..., N_sample, N_frame, 3[three atom], 3[coordinates]]
        true_frames = gather_frame_atom_by_indices(
            coordinate=true_coordinate, frame_atom_index=frame_atom_index, dim=-2
        )  # [..., N_frame, 3[three atom], 3[coordinates]]

        # Get pair_mask for computing the loss
        true_frame_coord_mask = gather_frame_atom_by_indices(
            coordinate=coordinate_mask, frame_atom_index=frame_atom_index, dim=-1
        )  # [N_frame, 3[three atom]]
        true_frame_coord_mask = (
            true_frame_coord_mask.sum(dim=-1) >= 3
        )  # [N_frame] whether all atoms in the frame has coordinates
        token_mask = coordinate_mask[rep_atom_mask]  # [N_token]
        frame_token_pair_mask = (
            true_frame_coord_mask[..., None] * token_mask[..., None, :]
        )  # [N_frame, N_token]

        squared_pae = (
            compute_alignment_error_squared(
                pred_coordinate=pred_coordinate[..., rep_atom_mask, :],
                true_coordinate=true_coordinate[..., rep_atom_mask, :],
                pred_frames=pred_frames,
                true_frames=true_frames,
            )
            * frame_token_pair_mask
        )  # [..., N_sample, N_frame, N_token]

        # Compute true bins
        boundaries = torch.linspace(
            start=self.min_bin,
            end=self.max_bin,
            steps=self.no_bins + 1,
            device=pred_coordinate.device,
        )
        boundaries = boundaries**2

        true_bins = torch.sum(
            squared_pae.unsqueeze(dim=-1) > boundaries, dim=-1
        )  # range [1, no_bins + 1]
        true_bins = torch.where(
            frame_token_pair_mask,
            true_bins,
            torch.ones_like(true_bins) * self.no_bins,
        )
        true_bins = torch.clamp(
            true_bins, min=1, max=self.no_bins
        )  # just in case bin=0 occurs

        return (
            squared_pae.detach(),
            F.one_hot(true_bins - 1, self.no_bins).detach(),
            frame_token_pair_mask.detach(),
        )

    def forward(
        self,
        logits: torch.Tensor,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        frame_atom_index: torch.Tensor,
        rep_atom_mask: torch.Tensor,
        has_frame: torch.Tensor,
    ) -> torch.Tensor:
        """PAELoss

        Args:
            logits (torch.Tensor): logits
                [..., N_sample, N_token, N_token, no_bins]
            pred_coordinate: (torch.Tensor): predict coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): true coordinates
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom]
            rep_atom_mask (torch.Tensor): masks of the representative atom for each token.
                [N_atom]
            frame_atom_index (torch.Tensor): indices of frame atoms (three atoms per token(=per frame)).
                [N_token, 3[three atom]]
            has_frame (torch.Tensor): indicates whether token_i has a valid frame.
                [N_token]
        Returns:
            torch.Tensor: the return loss
                [] if reduce
                [..., n] else
        """

        has_frame = has_frame.bool()
        rep_atom_mask = rep_atom_mask.bool()
        assert len(has_frame.shape) == 1
        assert len(frame_atom_index.shape) == 2

        with torch.no_grad():
            # true_bins: [..., N_sample, N_frame, N_token, no_bins]
            # pair_mask: [N_frame, N_token]
            _, true_bins, pair_mask = self.calculate_label(
                pred_coordinate=pred_coordinate,
                true_coordinate=true_coordinate,
                frame_atom_index=frame_atom_index,
                rep_atom_mask=rep_atom_mask,
                coordinate_mask=coordinate_mask,
                has_frame=has_frame,
            )

        loss = softmax_cross_entropy(
            logits=logits[
                ..., has_frame, :, :
            ],  # [..., N_sample, N_frame, N_token, no_bins]
            labels=true_bins,
        )  # [..., N_sample, N_frame, N_token]

        denom = self.eps + torch.sum(pair_mask, dim=(-1, -2))  # []
        loss = loss * pair_mask.unsqueeze(dim=-3)  # [..., N_sample, N_token, N_token]
        loss = torch.sum(loss, dim=(-1, -2))  # [..., N_sample]
        loss = loss / denom.unsqueeze(dim=-1)  # [..., N_sample]
        loss = loss.mean(dim=-1)  # [...]

        return loss_reduction(loss, self.reduction)


class ExperimentallyResolvedLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        """
        Args:
            eps (float, optional): avoid nan. Defaults to 1e-6.
        """
        super(ExperimentallyResolvedLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        coordinate_mask: torch.Tensor,
        atom_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): logits
                [..., N_sample, N_atom, no_bins:=2]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [..., N_atom] | [N_atom]
            atom_mask (torch.Tensor, optional): whether to conside the atom in the loss
                [..., N_atom]
        Returns:
            torch.Tensor: the experimentally resolved loss
        """
        is_resolved = F.one_hot(
            coordinate_mask.long(), 2
        )  # [..., N_atom, 2] or [N_atom, 2]
        errors = softmax_cross_entropy(
            logits=logits, labels=is_resolved.unsqueeze(dim=-3)
        )  # [..., N_sample, N_atom]
        if atom_mask is None:
            loss = errors.mean(dim=-1)  # [..., N_sample]
        else:
            loss = torch.sum(
                errors * atom_mask[..., None, :], dim=-1
            )  # [..., N_sample]
            loss = loss / (
                self.eps + torch.sum(atom_mask[..., None, :], dim=-1)
            )  # [..., N_sample]

        loss = loss.mean(dim=-1)  # [...]
        return loss_reduction(loss, method=self.reduction)


class MSELoss(nn.Module):
    """
    Implements Formula 2-4 [MSELoss] in AF3
    """

    def __init__(
        self,
        weight_mse: float = 1 / 3,
        weight_dna: float = 5.0,
        weight_rna=5.0,
        weight_ligand=10.0,
        eps=1e-6,
        reduction: str = "mean",
        mse_align: bool = False,
    ) -> None:
        super(MSELoss, self).__init__()
        self.weight_mse = weight_mse
        self.weight_dna = weight_dna
        self.weight_rna = weight_rna
        self.weight_ligand = weight_ligand
        self.eps = eps
        self.reduction = reduction
        # When True, compute MSE against true_coordinate aligned to pred (AF3 Algorithm 23).
        # When False, fall back to the codebase's trajectory behavior (MSE against raw GT,
        # which assumes the first frame is already given as an anchor).
        self.mse_align = mse_align

    def weighted_rigid_align(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        is_dna: torch.Tensor,
        is_rna: torch.Tensor,
        is_ligand: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """compute weighted rigid alignment results

        Args:
            pred_coordinate (torch.Tensor): the denoised coordinates from diffusion module
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth coordinates
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom] or [..., N_atom]
            is_dna / is_rna / is_ligand (torch.Tensor): mol type mask
                [N_atom] or [..., N_atom]

        Returns:
            true_coordinate_aligned (torch.Tensor): aligned coordinates for each sample
                [..., N_sample, N_atom, 3]
            weight (torch.Tensor): weights for each atom
                [N_atom] or [..., N_sample, N_atom]
        """
        N_sample = pred_coordinate.size(-3)
        weight = (
            1
            + self.weight_dna * is_dna
            + self.weight_rna * is_rna
            + self.weight_ligand * is_ligand
        )  # [N_atom] or [..., N_atom]
        # Apply coordinate_mask
        weight = weight * coordinate_mask  # [N_atom] or [..., N_atom]
        true_coordinate = true_coordinate * coordinate_mask[..., None, :, None]
        pred_coordinate = pred_coordinate * coordinate_mask[..., None, :, None]

        # Reshape to add "N_sample" dimension
        # true_coordinate = expand_at_dim(
        #     true_coordinate, dim=-3, n=N_sample
        # )  # [..., N_sample, N_atom, 3]
        if len(weight.shape) > 1:
            weight = expand_at_dim(
                weight, dim=-2, n=N_sample
            )  # [..., N_sample, N_atom]

        # ATTN: We don't perform weighted align, as the first frame is already given
        # Align GT coords to predicted coords
        d = pred_coordinate.dtype
        # Some ops in weighted_rigid_align do not support BFloat16 training
        with torch.amp.autocast("cuda", enabled=False):
            true_coordinate_aligned = weighted_rigid_align(
                x=true_coordinate.to(torch.float32),  # [..., N_sample, N_atom, 3]
                x_target=pred_coordinate.to(
                    torch.float32
                ),  # [..., N_sample, N_atom, 3]
                atom_weight=weight.to(
                    torch.float32
                ),  # [N_atom] or [..., N_sample, N_atom]
                stop_gradient=True,
            )  # [..., N_sample, N_atom, 3]
            true_coordinate_aligned = true_coordinate_aligned.to(d)

        return (true_coordinate_aligned.detach(), weight.detach())
        # return weight.detach()

    def forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        is_dna: torch.Tensor,
        is_rna: torch.Tensor,
        is_ligand: torch.Tensor,
        per_sample_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        """MSELoss

        Args:
            pred_coordinate (torch.Tensor): the denoised coordinates from diffusion module.
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist.
                [N_atom] or [..., N_atom]
            is_dna / is_rna / is_ligand (torch.Tensor): mol type mask.
                [N_atom] or [..., N_atom]
            per_sample_scale (torch.Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample]

        Returns:
            torch.Tensor: the weighted mse loss.
                [...] is self.reduction is None else []
        """
        # True_coordinate_aligned: [..., N_sample, N_atom, 3]
        # Weight: [N_atom] or [..., N_sample, N_atom]
        with torch.no_grad():
            true_coordinate_aligned, weight = self.weighted_rigid_align(
                pred_coordinate=pred_coordinate,
                true_coordinate=true_coordinate,
                coordinate_mask=coordinate_mask,
                is_dna=is_dna,
                is_rna=is_rna,
                is_ligand=is_ligand,
            )

        # Calculate MSE loss. AF3 Algorithm 23 uses aligned GT (mse_align=True). The
        # trajectory training path keeps the un-aligned formulation (first frame as anchor).
        if self.mse_align:
            per_atom_se = ((pred_coordinate - true_coordinate_aligned) ** 2).sum(
                dim=-1
            )  # [..., N_sample, N_atom]
        else:
            per_atom_se = ((pred_coordinate - true_coordinate) ** 2).sum(
                dim=-1
            )  # [..., N_sample, N_atom]
        per_sample_weighted_mse = (weight * per_atom_se).sum(dim=-1) / (
            coordinate_mask.sum(dim=-1, keepdim=True) + self.eps
        )  # [..., N_sample]

        if per_sample_scale is not None:
            per_sample_weighted_mse = per_sample_weighted_mse * per_sample_scale

        weighted_align_mse_loss = self.weight_mse * (per_sample_weighted_mse).mean()

        return weighted_align_mse_loss


def wasserstein_distance_1d(p_samples, g_samples, p=1, axis=-1):
    """
    计算两个一维经验分布之间的 p-Wasserstein 距离。

    参数:
    - p_samples (Tensor): 第一个分布的样本，形状为 (n, )。
    - g_samples (Tensor): 第二个分布的样本，形状为 (n, )。
    - p (int): p-Wasserstein 距离中的 p 值，通常为 1 或 2。

    返回:
    - Tensor: 计算出的 Wasserstein 距离。
    """
    # 确保两个样本集大小相同
    if p_samples.shape != g_samples.shape:
        raise ValueError("两个样本集的大小必须相同。")
    
    # 对两个样本集进行排序
    p_samples_sorted, _ = torch.sort(p_samples, dim=axis)
    g_samples_sorted, _ = torch.sort(g_samples, dim=axis)
    
    # 计算排序后样本之间的差值
    differences = p_samples_sorted - g_samples_sorted
    
    # 根据p值计算损失
    if p == 1:
        # 对于 p=1，计算差值绝对值的均值
        return torch.mean(torch.abs(differences), dim=axis)
    elif p == 2:
        # 对于 p=2，计算差值平方的均值的平方根 (RMSE)
        return torch.sqrt(torch.mean(differences ** 2, dim=axis))
    else:
        raise ValueError(f"p-Wasserstein 距离的 p 值必须为 1 或 2，当前为 {p}。")


class RelRMSFLoss(nn.Module):
    """Relative RMSF loss — the *shape* of the flexibility profile.

    Like :class:`RMSFLoss` but the per-atom RMSF is normalized per target (divided by
    its mean) before comparison, so the loss matches the relative pattern of which
    regions are more/less flexible, independent of the overall fluctuation scale.
    Complements the absolute RMSF term, which fixes the scale.
    """

    def __init__(
        self,
        eps=1e-6,
        reduction: str = "mean",
        pp_weight: float = 1.0,
        pl_weight: float = 5.0,
        ll_weight: float = 10.0,
    ) -> None:
        super(RelRMSFLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.min_std = 1
        self.pp_weight = pp_weight
        self.pl_weight = pl_weight
        self.ll_weight = ll_weight
        self.lamda = 0.5

    
    def forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        is_ligand: torch.Tensor,
        atom_to_tokatom_idx: torch.Tensor,
        lddt_mask: torch.Tensor,
        per_sample_scale: Optional[int] = None
    ) -> torch.Tensor:
        """SmoothLDDTLoss sparse implementation

        Args:
            pred_coordinate (torch.Tensor): the diffusion denoised atom coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [..., N_sample, N_atom, 3]
            lddt_mask (torch.Tensor, optional): whether true distance is within radius (30A for nuc and 15A for others)
                [N_frame, N_atom, N_atom]

        Returns:
            torch.Tensor: the smooth lddt loss
                [...] if reduction is None else []
        """

        # Compute distance error
        # [...,  N_sample , N_atom, N_atom]
        atom_is_ca = atom_to_tokatom_idx == 1
        atom_is_token = atom_is_ca | is_ligand.bool()
        mask = atom_is_token.unsqueeze(dim=-2) & atom_is_token.unsqueeze(dim=-1)

        indices = torch.where(mask & (lddt_mask.sum(dim=0) == lddt_mask.shape[0])) # pairs where all true distance is within 15A
        pred_coords_i = pred_coordinate.index_select(-2, indices[0])
        pred_coords_j = pred_coordinate.index_select(-2, indices[1])
        true_coords_i = true_coordinate.index_select(-2, indices[0])
        true_coords_j = true_coordinate.index_select(-2, indices[1])
        is_ligand_i = is_ligand.index_select(-1, indices[0])
        is_ligand_j = is_ligand.index_select(-1, indices[1])

        lig_lig_mask = torch.logical_and(is_ligand_i, is_ligand_j).float()
        prot_lig_mask = torch.logical_xor(is_ligand_i, is_ligand_j).float()
        prot_prot_mask = torch.logical_and(~is_ligand_i, ~is_ligand_i).float()
        weight = lig_lig_mask * self.ll_weight + prot_lig_mask * self.pl_weight + prot_prot_mask * self.pp_weight

        pred_distance_sparse = torch.norm(pred_coords_i - pred_coords_j, p=2, dim=-1)
        true_distance_sparse = torch.norm(true_coords_i - true_coords_j, p=2, dim=-1)

        # w2_dist = wasserstein_distance_1d(pred_distance_sparse, true_distance_sparse, p=1, axis=0) # [N_sample, N_pair]

        # pair with large std should give less weight
        # std = true_distance_sparse.std(dim=0)
        # loss_weight = 1. / std.clamp(min=self.min_std, max=None)
        # loss_weight = loss_weight

        pred_std = pred_distance_sparse.std(dim=0)
        true_std = true_distance_sparse.std(dim=0)
        true_dist = true_distance_sparse.mean(dim=0)
        # Initialize weights with a default value (e.g., 0 or 1, depending on your needs for >15)
        dist_weight = torch.ones_like(true_dist)
        # [0, 5]: 4
        dist_weight[true_dist <= 5] = 4
        
        # (5, 10]: 2
        dist_weight[(true_dist > 5) & (true_dist <= 10)] = 2
        
        # (10, 15]: 1
        dist_weight[(true_dist > 10)] = 1

        # normalize
        dist_weight = (dist_weight / dist_weight.sum(dim=-1)[:, None]) * dist_weight.shape[-1]

        rel_rmsf_loss = (pred_std - true_std) ** 2 # [N_sample, N_pair]
        rel_rmsf_loss = rel_rmsf_loss * weight[None, :]  # [N_sample, N_pair]
        rel_rmsf_loss = rel_rmsf_loss * dist_weight

        # using square, rather than sqrt
        # rel_rmsf_loss = w2_dist * loss_weight # [N_sample, N_pair]
        if per_sample_scale is not None:
            rel_rmsf_loss = rel_rmsf_loss * per_sample_scale.unsqueeze(-1)

        return rel_rmsf_loss.mean()



class LocalRMSFLoss(nn.Module):
    """Local RMSF loss — flexibility measured in a local reference frame.

    Computes RMSF after aligning each residue's local neighborhood, so it captures
    *local* backbone fluctuation decoupled from large-scale global/domain motion that
    a single global alignment would otherwise fold into the per-atom RMSF.
    """

    def __init__(
        self,
        eps=1e-6,
        reduction: str = "mean",
    ) -> None:
        super(LocalRMSFLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.min_std = 1

    
    def forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        is_ligand: torch.Tensor,
        atom_to_token_idx: torch.Tensor,
        lddt_mask: torch.Tensor,
        per_sample_scale: Optional[int] = None
    ) -> torch.Tensor:
        """SmoothLDDTLoss sparse implementation

        Args:
            pred_coordinate (torch.Tensor): the diffusion denoised atom coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [..., N_sample, N_atom, 3]
            lddt_mask (torch.Tensor, optional): whether true distance is within radius (30A for nuc and 15A for others)
                [N_frame, N_atom, N_atom]

        Returns:
            torch.Tensor: the smooth lddt loss
                [...] if reduction is None else []
        """

        # Compute distance error
        # [...,  N_sample , N_atom, N_atom]
        mask = atom_to_token_idx.unsqueeze(dim=-2) == atom_to_token_idx.unsqueeze(dim=-1)
        indices = torch.where(mask & (lddt_mask.sum(dim=0) == lddt_mask.shape[0])) # pairs where all true distance is within 15A
        pred_coords_i = pred_coordinate.index_select(-2, indices[0])
        pred_coords_j = pred_coordinate.index_select(-2, indices[1])
        true_coords_i = true_coordinate.index_select(-2, indices[0])
        true_coords_j = true_coordinate.index_select(-2, indices[1])

        pred_distance_sparse = torch.norm(pred_coords_i - pred_coords_j, p=2, dim=-1)
        true_distance_sparse = torch.norm(true_coords_i - true_coords_j, p=2, dim=-1)

        # w2_dist = wasserstein_distance_1d(pred_distance_sparse, true_distance_sparse, p=1, axis=0) # [N_sample, N_pair]

        # pair with large std should give less weight
        # std = true_distance_sparse.std(dim=0)
        # loss_weight = 1. / std.clamp(min=self.min_std, max=None)
        # loss_weight = loss_weight

        pred_std = pred_distance_sparse.std(dim=0)
        true_std = true_distance_sparse.std(dim=0)

        local_rmsf_loss = (pred_std - true_std) ** 2

        # using square, rather than sqrt
        # rel_rmsf_loss = w2_dist * loss_weight # [N_sample, N_pair]
        if per_sample_scale is not None:
            local_rmsf_loss = local_rmsf_loss * per_sample_scale.unsqueeze(-1)

        # 修复nan值
        local_rmsf_loss = torch.nan_to_num(local_rmsf_loss.mean(), nan=0.0)
        return local_rmsf_loss



class RMSFLoss(nn.Module):
    """Root-Mean-Square Fluctuation (RMSF) loss.

    RMSF is the per-atom standard deviation of position about the trajectory mean
    (after rigid alignment) — i.e. how much each atom fluctuates. This loss matches
    the predicted ensemble's per-atom RMSF to the ground-truth MD RMSF, so the model
    reproduces the magnitude of conformational flexibility (rigid core vs. mobile
    loops). DNA/RNA atoms are up-weighted (`weight_dna`/`weight_rna`).
    """

    def __init__(
        self,
        weight_rmsf: float = 1.0,
        weight_dna: float = 5.0,
        weight_rna=5.0,
        weight_ligand=10.0,
        eps=1e-6,
        reduction: str = "mean",
        **kwargs
    ) -> None:
        super(RMSFLoss, self).__init__()
        self.weight_rmsf = weight_rmsf
        self.weight_dna = weight_dna
        self.weight_rna = weight_rna
        self.weight_ligand = weight_ligand
        self.eps = eps
        self.reduction = reduction
        self.lamda = 0.5

    def get_atom_weight(
        self,
        coordinate_mask: torch.Tensor,
        is_dna: torch.Tensor,
        is_rna: torch.Tensor,
        is_ligand: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom] or [..., N_atom]
            is_dna / is_rna / is_ligand (torch.Tensor): mol type mask
                [N_atom] or [..., N_atom]

        Returns:
            true_coordinate_aligned (torch.Tensor): aligned coordinates for each sample
                [..., N_sample, N_atom, 3]
            weight (torch.Tensor): weights for each atom
                [N_atom] or [..., N_sample, N_atom]
        """
        weight = (
            1
            + self.weight_dna * is_dna
            + self.weight_rna * is_rna
            + self.weight_ligand * is_ligand
        )  # [N_atom] or [..., N_atom]
        # Apply coordinate_mask
        weight = weight * coordinate_mask  # [N_atom] or [..., N_atom]
        return weight.detach()

    def forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        is_dna: torch.Tensor,
        is_rna: torch.Tensor,
        is_ligand: torch.Tensor,
        per_sample_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        """MSELoss

        Args:
            pred_coordinate (torch.Tensor): the denoised coordinates from diffusion module.
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist.
                [N_atom] or [..., N_atom]
            is_dna / is_rna / is_ligand (torch.Tensor): mol type mask.
                [N_atom] or [..., N_atom]
            per_sample_scale (torch.Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample]

        Returns:
            torch.Tensor: the weighted mse loss.
                [...] is self.reduction is None else []
        """
        # Weight: [N_atom] 
        with torch.no_grad():
            weight = self.get_atom_weight(
                coordinate_mask=coordinate_mask,
                is_dna=is_dna,
                is_rna=is_rna,
                is_ligand=is_ligand,
            )

        # Calculate rmsf loss
        eps = 1e-6
        pred_deviation = pred_coordinate - pred_coordinate.mean(dim=0) 
        pred_square_deviation = (pred_deviation ** 2).sum(dim=-1) # [N_frame, N_sample, N_atom]
        pred_rmsf = torch.sqrt(pred_square_deviation.mean(dim=0) + eps) # [N_sample, N_atom]

        true_deviation = true_coordinate - true_coordinate.mean(dim=0)
        true_square_deviation = (true_deviation ** 2).sum(dim=-1)
        true_rmsf = torch.sqrt(true_square_deviation.mean(dim=0)+eps)

        per_atom_rmsf_loss = weight[None, ...] * (pred_rmsf - true_rmsf) ** 2 # [N_sample, N_atom]
        # per_atom_rmsf_loss = self.lamda*per_atom_rmsf_loss / (true_rmsf + 0.1) + (1.-self.lamda)*per_atom_rmsf_loss # pay attention to both flexible and rigid area

        if per_sample_scale is not None:
            per_atom_rmsf_loss = per_atom_rmsf_loss * per_sample_scale.unsqueeze(-1)

        rmsf_loss = self.weight_rmsf * (per_atom_rmsf_loss).mean()

        return rmsf_loss



class VelocityLoss(nn.Module):
    """Velocity (frame-to-frame displacement) loss.

    Matches the magnitude of per-step atomic displacement between consecutive frames
    of the predicted vs. ground-truth trajectory. This supervises the short-time
    motion scale (how far atoms move per frame), preventing trajectories that are too
    static or too diffusive at the chosen frame spacing.
    """

    def __init__(
        self,
        weight_vel: float = 1.0,
        weight_dna: float = 5.0,
        weight_rna=5.0,
        weight_ligand=10.0,
        eps=1e-6,
        reduction: str = "mean",
        **kwargs
    ) -> None:
        super(VelocityLoss, self).__init__()
        self.weight_vel = weight_vel
        self.weight_dna = weight_dna
        self.weight_rna = weight_rna
        self.weight_ligand = weight_ligand
        self.eps = eps
        self.reduction = reduction
        self.lamda = 0.5

    def get_atom_weight(
        self,
        coordinate_mask: torch.Tensor,
        is_dna: torch.Tensor,
        is_rna: torch.Tensor,
        is_ligand: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom] or [..., N_atom]
            is_dna / is_rna / is_ligand (torch.Tensor): mol type mask
                [N_atom] or [..., N_atom]

        Returns:
            true_coordinate_aligned (torch.Tensor): aligned coordinates for each sample
                [..., N_sample, N_atom, 3]
            weight (torch.Tensor): weights for each atom
                [N_atom] or [..., N_sample, N_atom]
        """
        weight = (
            1
            + self.weight_dna * is_dna
            + self.weight_rna * is_rna
            + self.weight_ligand * is_ligand
        )  # [N_atom] or [..., N_atom]
        # Apply coordinate_mask
        weight = weight * coordinate_mask  # [N_atom] or [..., N_atom]
        return weight.detach()

    def forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        is_dna: torch.Tensor,
        is_rna: torch.Tensor,
        is_ligand: torch.Tensor,
        per_sample_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        """MSELoss

        Args:
            pred_coordinate (torch.Tensor): the denoised coordinates from diffusion module.
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist.
                [N_atom] or [..., N_atom]
            is_dna / is_rna / is_ligand (torch.Tensor): mol type mask.
                [N_atom] or [..., N_atom]
            per_sample_scale (torch.Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample]

        Returns:
            torch.Tensor: the weighted mse loss.
                [...] is self.reduction is None else []
        """
        # Weight: [N_atom] 
        with torch.no_grad():
            weight = self.get_atom_weight(
                coordinate_mask=coordinate_mask,
                is_dna=is_dna,
                is_rna=is_rna,
                is_ligand=is_ligand,
            )

        # Calculate rmsf loss
        eps = 1e-6
        pred_velocity = pred_coordinate[1:] - pred_coordinate[:-1]
        pred_velocity_square = (pred_velocity ** 2).sum(dim=-1) # [N_frame-1, N_sample, N_atom]
        pred_velocity_l2 = torch.sqrt(pred_velocity_square + eps) # [N_frame-1, N_sample, N_atom]

        true_velocity = true_coordinate[1:] - true_coordinate[:-1]
        true_velocity_square = (true_velocity ** 2).sum(dim=-1)
        true_velocity_l2 = torch.sqrt(true_velocity_square + eps) # [N_frame-1, N_sample, N_atom]

        true_velocity_mean = true_velocity_l2.mean(dim=0) # [N_sample, N_atom]
        pred_velocity_mean = pred_velocity_l2.mean(dim=0) # [N_sample, N_atom]

        per_atom_v_loss = weight[None, ...] * (pred_velocity_mean - true_velocity_mean) ** 2 # [N_sample, N_atom]
        # per_atom_v_loss = self.lamda * per_atom_v_loss / (true_velocity_mean + 0.1) + (1-self.lamda)*per_atom_v_loss # pay attention to both flexible and rigid area

        if per_sample_scale is not None:
            per_atom_v_loss = per_atom_v_loss * per_sample_scale.unsqueeze(-1)

        v_loss = self.weight_vel * (per_atom_v_loss).mean()

        return v_loss


class ACFLoss(nn.Module):
    """
    Lag-1 autocorrelation function (ACF) loss on Cα backbone coordinates.

    Penalizes deviation of the predicted trajectory's normalized Cα ACF from the
    ground truth ACF at lag-1 (and optionally lag-2). This directly supervises the
    temporal correlation structure, which BioKinema was found to underestimate
    (Cα displacement ACF decorrelates ~3× faster than MD reference).

    Both pred_coordinate and true_coordinate are expected to be in the same
    augmented frame ([N_frame, N_sample, N_atom, 3]).
    """

    def __init__(self, eps: float = 1e-6, min_frames: int = 3, max_lag: int = 2):
        super().__init__()
        self.eps = eps
        self.min_frames = min_frames
        self.max_lag = max_lag

    def _normalized_acf(self, flat: torch.Tensor, lag: int) -> torch.Tensor:
        """
        Compute normalized lag-τ ACF for a flattened trajectory.

        Args:
            flat: [N_frame, N_sample, D]  (already mean-subtracted over frames)
            lag:  integer lag

        Returns:
            acf: [N_sample]  — normalized autocorrelation at the given lag
        """
        N_frame = flat.shape[0]
        num = (flat[:N_frame - lag] * flat[lag:]).sum(dim=(0, 2))   # [N_sample]
        den = (flat ** 2).sum(dim=(0, 2)) + self.eps                 # [N_sample]
        return num / den

    def forward(
        self,
        pred_coordinate: torch.Tensor,      # [N_frame, N_sample, N_atom, 3]
        true_coordinate: torch.Tensor,      # [N_frame, N_sample, N_atom, 3]
        coordinate_mask: torch.Tensor,      # [N_atom]
        atom_to_tokatom_idx: torch.Tensor,  # [N_atom]
        is_ligand: torch.Tensor,            # [N_atom]
    ) -> torch.Tensor:
        N_frame = pred_coordinate.shape[0]
        if N_frame < self.min_frames:
            # Too few frames to compute a meaningful ACF; return a differentiable zero
            return pred_coordinate.sum() * 0.0

        # Cα mask: index-1 atom within each residue is Cα (AF3 atom ordering),
        # including ligand atoms and atoms without valid coordinates.
        ca_mask = (
            ( (atom_to_tokatom_idx == 1) | is_ligand.bool() )
            & coordinate_mask.bool()
        )
        if ca_mask.sum() == 0:
            return pred_coordinate.sum() * 0.0

        # [N_frame, N_sample, N_ca, 3]
        pred_ca = pred_coordinate[:, :, ca_mask, :]
        true_ca = true_coordinate[:, :, ca_mask, :]

        # Center over the time dimension so that ACF measures temporal fluctuations
        pred_ca = pred_ca - pred_ca.mean(dim=0, keepdim=True)
        true_ca = true_ca - true_ca.mean(dim=0, keepdim=True)

        # Flatten spatial dims: [N_frame, N_sample, N_ca*3]
        N_sample = pred_ca.shape[1]
        D = pred_ca.shape[2] * 3
        pred_flat = pred_ca.reshape(N_frame, N_sample, D)
        true_flat = true_ca.reshape(N_frame, N_sample, D)

        total_loss = torch.tensor(0.0, device=pred_coordinate.device, dtype=pred_coordinate.dtype)
        n_lags = 0
        for lag in range(1, self.max_lag + 1):
            if N_frame <= lag:
                break
            pred_acf = self._normalized_acf(pred_flat, lag)          # [N_sample]
            true_acf = self._normalized_acf(true_flat, lag).detach()  # [N_sample]
            total_loss = total_loss + ((pred_acf - true_acf) ** 2).mean()
            n_lags += 1

        if n_lags == 0:
            return pred_coordinate.sum() * 0.0
        return total_loss / n_lags


def calculate_atom_bespoke_lddt(
    pred_coordinate: torch.Tensor,
    true_coordinate: torch.Tensor,
    is_nucleotide: torch.Tensor,
    is_polymer: torch.Tensor,
    rep_atom_mask: torch.Tensor,
    is_nucleotide_threshold: float = 30.0,
    is_not_nucleotide_threshold: float = 15.0,
) -> torch.Tensor:
    """calculate the bespoke lddt as described in Sec 4.3.1.
    Args:
        pred_coordinate (torch.Tensor):
            [..., N_sample, N_atom, 3]
        true_coordinate (torch.Tensor):
            [..., N_atom]
        is_nucleotide (torch.Tensor):
            [N_atom] or [..., N_atom]
        is_polymer (torch.Tensor):
            [N_atom]
        rep_atom_mask (torch.Tensor):
            [N_atom]
    Returns:
        torch.Tensor: per-atom lddt
            [..., N_sample, N_atom, 1]
        torch.Tensor: per-atom lddt weight
            [..., N_sample, N_atom, 1]
    """

    N_atom = true_coordinate.size(-2)
    atom_m_mask = (rep_atom_mask * is_polymer).bool()  # [N_atom]
    # Distance: d_lm
    pred_d_lm = torch.cdist(
        pred_coordinate, pred_coordinate[..., atom_m_mask, :]
    )  # [..., N_sample, N_atom, N_atom(m)]
    true_d_lm = torch.cdist(
        true_coordinate, true_coordinate[..., atom_m_mask, :]
    )  # [..., N_atom, N_atom(m)]
    delta_d_lm = torch.abs(
        pred_d_lm - true_d_lm.unsqueeze(dim=-3)
    )  # [..., N_sample, N_atom, N_atom(m)]
    # Pair-wise lddt
    thresholds = [0.5, 1, 2, 4]
    lddt_lm = (
        torch.stack([delta_d_lm < t for t in thresholds], dim=-1)
        .to(dtype=delta_d_lm.dtype)
        .mean(dim=-1)
    )  # [..., N_sample, N_atom, N_atom(m)]
    # Select atoms that are within certain threshold to l in ground truth
    # Restrict to bespoke inclusion radius
    is_nucleotide = is_nucleotide[
        ..., atom_m_mask
    ].bool()  # [N_atom(m)] or [..., N_atom(m)]
    locality_mask = (true_d_lm < is_nucleotide_threshold) * is_nucleotide.unsqueeze(
        dim=-2
    ) + (true_d_lm < is_not_nucleotide_threshold) * (
        ~is_nucleotide.unsqueeze(dim=-2)
    )  # [..., N_atom, N_atom(m)]
    # Remove self-distance computation
    diagonal_mask = ((1 - torch.eye(n=N_atom)).bool().to(true_d_lm.device))[
        ..., atom_m_mask
    ]  # [N_atom, N_atom(m)]
    pair_mask = (locality_mask * diagonal_mask).unsqueeze(
        dim=-3
    )  # [..., 1, N_atom, N_atom(m)]
    per_atom_lddt = torch.sum(
        lddt_lm * pair_mask, dim=-1, keepdim=True
    )  # [...,  N_sample, N_atom, 1]
    per_atom_weight = torch.sum(pair_mask.to(dtype=lddt_lm.dtype), dim=-1, keepdim=True)
    return per_atom_lddt, per_atom_weight


class PLDDTLoss(nn.Module):
    """
    Implements PLDDT Loss in AF3, different from the paper description.
    Main changes:
    1. use difference of distance instead of predicted distance when calculating plddt
    2. normalize each plddt score within 0-1
    """

    def __init__(
        self,
        min_bin: float = 0,
        max_bin: float = 1,
        no_bins: int = 50,
        is_nucleotide_threshold: float = 30.0,
        is_not_nucleotide_threshold: float = 15.0,
        eps: float = 1e-6,
        normalize: bool = True,
        reduction: str = "mean",
    ) -> None:
        """PLDDT loss
        This loss are between atoms l and m (has some filters) in the mini-rollout prediction

        Args:
            min_bin (float, optional): min boundary of bins. Defaults to 0.
            max_bin (float, optional): max boundary of bins. Defaults to 1.
            no_bins (int, optional): number of bins. Defaults to 50.
            is_nucleotide_threshold (float, optional): threshold for nucleotide atoms. Defaults 30.0.
            is_not_nucleotide_threshold (float, optional): threshold for non-nucleotide atoms. Defaults 15.0
            eps (float, optional): small number added to denominator. Defaults to 1e-6.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super(PLDDTLoss, self).__init__()
        self.normalize = normalize
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.eps = eps
        self.reduction = reduction
        self.is_nucleotide_threshold = is_nucleotide_threshold
        self.is_not_nucleotide_threshold = is_not_nucleotide_threshold

    def bins_from_lddt(
        self,
        per_atom_lddt: torch.Tensor,
        per_atom_weight: torch.Tensor,
    ):
        if self.normalize:
            per_atom_lddt = per_atom_lddt / (per_atom_weight + self.eps)
        # Distribute into bins
        boundaries = torch.linspace(
            start=self.min_bin,
            end=self.max_bin,
            steps=self.no_bins + 1,
            device=per_atom_lddt.device,
        )  # [N_bins]

        true_bins = torch.sum(
            per_atom_lddt > boundaries, dim=-1
        )  # [...,  N_sample, N_atom], range in [1, no_bins]
        true_bins = torch.clamp(
            true_bins, min=1, max=self.no_bins
        )  # just in case bin=0/no_bins+1 occurs
        true_bins = F.one_hot(
            true_bins - 1, self.no_bins
        )  # [...,  N_sample, N_atom, N_bins]

        return true_bins

    def forward_given_atom_lddt(
        self,
        logits: torch.Tensor,
        per_atom_lddt: torch.Tensor,
        per_atom_weight: torch.Tensor,
    ):
        """
        Args:
        per_atom_lddt
            [..., N_sample, N_atom, 1]
        per_atom_weight
            [..., N_sample, N_atom, 1]
        Returns:
            torch.Tensor: per-atom lddt bins
                [..., N_sample, N_atom, N_bins]
        """
        with torch.no_grad():
            true_bins = self.bins_from_lddt(per_atom_lddt, per_atom_weight).detach()
        plddt_loss = softmax_cross_entropy(
            logits=logits,
            labels=true_bins,
        )  # [..., N_sample, N_atom_with_coords]
        # Average over atoms
        plddt_loss = plddt_loss.mean(dim=-1)  # [..., N_sample]
        # Average over samples
        plddt_loss = plddt_loss.mean(dim=-1)  # [...]
        return loss_reduction(plddt_loss, method=self.reduction)

    def forward(
        self,
        logits: torch.Tensor,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        is_nucleotide: torch.Tensor,
        is_polymer: torch.Tensor,
        rep_atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """PLDDT loss

        Args:
            logits (torch.Tensor): logits
                [..., N_sample, N_atom, no_bins:=50]
            pred_coordinate (torch.Tensor): predicted coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): true coordinates
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom]
            is_nucleotide (torch.Tensor): "is_rna" or "is_dna"
                [N_atom]
            is_polymer (torch.Tensor): not "is_ligand"
                [N_atom]
            rep_atom_mask (torch.Tensor): representative atom of each token
                [N_atom]

        Returns:
            torch.Tensor: the return loss
                [...] if self.reduction is None else []
        """
        assert (
            is_nucleotide.shape
            == is_polymer.shape
            == rep_atom_mask.shape
            == coordinate_mask.shape
            == coordinate_mask.view(-1).shape
        )

        coordinate_mask = coordinate_mask.bool()
        rep_atom_mask = rep_atom_mask.bool()
        is_nucleotide = is_nucleotide.bool()
        is_polymer = is_polymer.bool()

        with torch.no_grad():
            per_atom_lddt, per_atom_weight = calculate_atom_bespoke_lddt(
                pred_coordinate=pred_coordinate[..., coordinate_mask, :],
                true_coordinate=true_coordinate[..., coordinate_mask, :],
                is_nucleotide=is_nucleotide[coordinate_mask],
                is_polymer=is_polymer[coordinate_mask],
                rep_atom_mask=rep_atom_mask[coordinate_mask],
                is_nucleotide_threshold=self.is_nucleotide_threshold,
                is_not_nucleotide_threshold=self.is_not_nucleotide_threshold,
            )

        loss = self.forward_given_atom_lddt(
            logits=logits[..., coordinate_mask, :],
            per_atom_lddt=per_atom_lddt,
            per_atom_weight=per_atom_weight,
        )

        return loss


class CollisionLoss(nn.Module):
    """
    Collision Loss: 惩罚配体原子与其他原子距离过近（碰撞）的情况
    使用动态阈值，参考 GT 距离避免误判
    """
    def __init__(
        self,
        pl_threshold: float = 3.0,   # 配体-蛋白碰撞阈值（埃）
        ll_threshold: float = 2.0,   # 配体-配体碰撞阈值（埃）
        gt_ratio: float = 0.9,       # GT 距离的缩放比例
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        """
        Args:
            pl_threshold: 配体-蛋白原子间最大允许距离（埃）
            ll_threshold: 配体-配体原子间最大允许距离（埃）
            gt_ratio: GT 距离的缩放比例，用于动态阈值
            weight_collision: 损失权重
            eps: 避免除零
            reduction: 损失聚合方式
        """
        super(CollisionLoss, self).__init__()
        self.pl_threshold = pl_threshold
        self.ll_threshold = ll_threshold
        self.gt_ratio = gt_ratio
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, pred_x1, x1, atom_is_ligand, bond_mask, per_sample_scale=None):
        # atoms between ligand and protein are not colliding
        pred_x1_ligs = pred_x1[..., atom_is_ligand, :] # [N_frame, N_sample, N_ligand, 3]
        pred_x1_prots = pred_x1[..., ~atom_is_ligand, :] # [N_frame, N_sample, N_protein, 3]
        pred_dist = torch.cdist(pred_x1_prots, pred_x1_ligs, p=2) # [N_frame, N_sample, N_protein, N_ligand]
        pred_dist_lig = torch.cdist(pred_x1_ligs, pred_x1_ligs, p=2)

        x1_ligs = x1[..., atom_is_ligand, :] # [N_frame, N_sample, N_ligand, 3]
        x1_prots = x1[..., ~atom_is_ligand, :] # [N_frame, N_sample, N_protein, 3]
        gt_dist = torch.cdist(x1_prots, x1_ligs, p=2) # [N_frame, N_sample, N_protein, N_ligand]
        gt_dist = gt_dist.min(dim=0)[0][0] # [N_protein, N_ligand]
        gt_dist_lig = torch.cdist(x1_ligs, x1_ligs, p=2) # [N_frame, N_sample, N_protein, N_ligand]
        gt_dist_lig = gt_dist_lig.min(dim=0)[0][0] # [N_protein, N_ligand]

        pl_thres = 3.0
        ll_thres = 2.0
        # collision: pred_dist < gt_dist and pred_dist < 3A

        threshold = torch.minimum(0.9 * gt_dist, torch.ones_like(gt_dist) * pl_thres/16.)[None, None, ...]
        collision_mask = (pred_dist < threshold).float() # [N_frame, N_sample, N_protein, N_ligand]
        
        threshold_ligand = torch.minimum(0.9 * gt_dist_lig, torch.ones_like(gt_dist_lig) * ll_thres/16.)[None, None, ...]
        collision_mask_lig = (pred_dist_lig < threshold_ligand).float()
        lig_bond = bond_mask[atom_is_ligand][:, atom_is_ligand].float()
        lig_num = lig_bond.shape[0]
        collision_mask_lig = collision_mask_lig * ( (1-lig_bond) * (1-torch.eye(lig_num).to(x1.device)) )[None, None, ...]
        # if collision_mask_lig.sum() > 0:
        #     print(collision_mask_lig.sum())
        #     breakpoint()

        # get collision loss
        collision_loss = collision_mask * (threshold - pred_dist) ** 2  # [N_frame, N_sample, N_protein, N_ligand]
        collision_loss = collision_loss.sum(dim=-1).sum(dim=-1) # [N_frame, N_sample]
        collision_loss_lig = collision_mask_lig * (threshold_ligand - pred_dist_lig) ** 2  # [N_frame, N_sample, N_protein, N_ligand]
        collision_loss_lig = collision_loss_lig.sum(dim=-1).sum(dim=-1) # [N_frame, N_sample]
        collision_loss = collision_loss + collision_loss_lig
        if per_sample_scale is not None:
            collision_loss = collision_loss * per_sample_scale
        
        # 7. 聚合
        collision_loss = collision_loss.mean()
        return collision_loss
    
    def forward1(
        self,
        pred_coordinate: torch.Tensor,  # [N_frame, N_sample, N_atom, 3]
        true_coordinate: torch.Tensor,  # [N_frame, N_atom, 3]
        is_ligand: torch.Tensor,        # [N_atom]
        bond_mask: torch.Tensor,        # [N_atom, N_atom]
        per_sample_scale: torch.Tensor = None,  # [N_frame, N_sample]
    ) -> torch.Tensor:
        """
        计算 collision loss
        
        Args:
            pred_coordinate: 预测的原子坐标 [N_frame, N_sample, N_atom, 3]
            true_coordinate: 真实的原子坐标 [N_frame, N_atom, 3]
            is_ligand: 配体原子标记 [N_atom]
            bond_mask: 化学键mask [N_atom, N_atom]
            per_sample_scale: 每个样本的缩放因子 [N_frame, N_sample]
            
        Returns:
            collision_loss: 标量损失
        """
        is_ligand = is_ligand.bool()
        
        # 1. 分离配体和蛋白原子
        pred_ligs = pred_coordinate[..., is_ligand, :]   # [N_frame, N_sample, N_lig, 3]
        pred_prots = pred_coordinate[..., ~is_ligand, :] # [N_frame, N_sample, N_prot, 3]
        true_ligs = true_coordinate[..., is_ligand, :]   # [N_frame, N_lig, 3]
        true_prots = true_coordinate[..., ~is_ligand, :] # [N_frame, N_prot, 3]
        
        # 2. 配体-蛋白碰撞
        # 预测距离
        pred_dist_pl = torch.cdist(pred_prots, pred_ligs, p=2)  # [N_frame, N_sample, N_prot, N_lig]
        # GT 距离（取时间维度最小值）
        gt_dist_pl = torch.cdist(true_prots, true_ligs, p=2)    # [N_frame, N_prot, N_lig]
        gt_dist_pl = gt_dist_pl.min(dim=0)[0]                   # [N_prot, N_lig]
        
        # 动态阈值：min(0.9 * gt_dist, 3.0Å / 16)
        threshold_pl = torch.minimum(
            self.gt_ratio * gt_dist_pl,
            torch.full_like(gt_dist_pl, self.pl_threshold / 16.0)
        )[None, None, ...]  # [1, 1, N_prot, N_lig]
        
        # 碰撞 mask：预测距离 < 阈值
        collision_mask_pl = (pred_dist_pl < threshold_pl).float()
        # 碰撞损失：(threshold - pred_dist)^2
        collision_loss_pl = (collision_mask_pl * (threshold_pl - pred_dist_pl) ** 2).sum(dim=(-1, -2))
        
        # 3. 配体-配体碰撞
        pred_dist_ll = torch.cdist(pred_ligs, pred_ligs, p=2)  # [N_frame, N_sample, N_lig, N_lig]
        gt_dist_ll = torch.cdist(true_ligs, true_ligs, p=2)    # [N_frame, N_lig, N_lig]
        gt_dist_ll = gt_dist_ll.min(dim=0)[0]                  # [N_lig, N_lig]
        
        threshold_ll = torch.minimum(
            self.gt_ratio * gt_dist_ll,
            torch.full_like(gt_dist_ll, self.ll_threshold / 16.0)
        )[None, None, ...]  # [1, 1, N_lig, N_lig]
        
        # 排除化学键和自身
        lig_bond = bond_mask[is_ligand][:, is_ligand].float()  # [N_lig, N_lig]
        lig_num = lig_bond.shape[0]
        lig_mask = (1 - lig_bond) * (1 - torch.eye(lig_num, device=bond_mask.device))
        
        collision_mask_ll = (pred_dist_ll < threshold_ll).float() * lig_mask[None, None, ...]
        collision_loss_ll = (collision_mask_ll * (threshold_ll - pred_dist_ll) ** 2).sum(dim=(-1, -2))
        
        # 4. 总损失
        collision_loss = collision_loss_pl + collision_loss_ll  # [N_frame, N_sample]
        
        # 5. 归一化（避免原子数影响）
        num_pairs = collision_mask_pl.sum(dim=(-1, -2)) + collision_mask_ll.sum(dim=(-1, -2)) + self.eps
        collision_loss = collision_loss / num_pairs
        
        # 6. 应用 per_sample_scale（如 t^2 缩放）
        if per_sample_scale is not None:
            collision_loss = collision_loss * per_sample_scale
        
        # 7. 聚合
        collision_loss = collision_loss.mean()
        
        return collision_loss

class CenterLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(CenterLoss, self).__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred_coordinate: torch.Tensor,  # [N_frame, N_sample, N_atom, 3]
        true_coordinate: torch.Tensor,  # [N_frame, N_sample, N_atom, 3]
        coordinate_mask: torch.Tensor,  # [N_atom]
        is_ligand: torch.Tensor,        # [N_atom]
        per_sample_scale: torch.Tensor = None,  # [N_frame, N_sample]
    ) -> torch.Tensor:
        is_ligand_valid = is_ligand.bool() & coordinate_mask.bool()

        if is_ligand_valid.sum() == 0:
            # 返回一个标量 0，必须确保它在正确的设备(GPU/CPU)上
            # requires_grad=True 是为了保持计算图的完整性（虽然梯度是0）
            return torch.tensor(0.0, device=pred_coordinate.device, requires_grad=True)


        is_ligand_ = is_ligand_valid[..., None]  # [N_atom, 1]
        lig_center_true = (true_coordinate * is_ligand_).sum(dim=-2) / is_ligand_.sum(dim=-2)
        lig_center_pred = (pred_coordinate * is_ligand_).sum(dim=-2) / is_ligand_.sum(dim=-2)
    
        center_loss = ((lig_center_true - lig_center_pred) ** 2).sum(dim=-1)

        if per_sample_scale is not None:
            center_loss = center_loss * per_sample_scale

        center_loss = center_loss.mean()
        return center_loss

class ProtenixLoss(nn.Module):
    """Aggregates all training losses for trajectory-generation training.

    Active terms (weights in configs_base.py `loss.weight`, all scaled by
    `alpha_diffusion`):
      • diffusion MSE + smooth-LDDT + bond  — per-frame structural fidelity (AF3).
      • RMSF / RelRMSF / LocalRMSF           — per-atom flexibility (magnitude, shape, local).
      • Velocity                             — frame-to-frame displacement scale.
      • ACF                                  — Cα lag-1 temporal autocorrelation.
      • ensemble / center / lig_bond         — ensemble-level + ligand geometry terms.
      • tica_dynamics                        — TICA-space transition/population/autocorrelation;
                                               only active when the dataset provides an MSM/TICA
                                               cache (has_msm=True), else skipped.
    Ensemble terms (RMSF/Velocity/ACF/TICA) require traj_len >= 3.
    """

    def __init__(self, configs) -> None:
        super(ProtenixLoss, self).__init__()
        self.configs = configs

        self.alpha_confidence = self.configs.loss.weight.alpha_confidence
        self.alpha_pae = self.configs.loss.weight.alpha_pae
        self.alpha_except_pae = self.configs.loss.weight.alpha_except_pae
        self.alpha_diffusion = self.configs.loss.weight.alpha_diffusion
        self.alpha_distogram = self.configs.loss.weight.alpha_distogram
        self.alpha_bond = self.configs.loss.weight.alpha_bond
        self.weight_smooth_lddt = self.configs.loss.weight.smooth_lddt
        self.alpha_collision = self.configs.loss.weight.alpha_collision
        self.alpha_center = self.configs.loss.weight.alpha_center
        self.alpha_rmsf = self.configs.loss.weight.alpha_rmsf
        self.alpha_velocity = self.configs.loss.weight.alpha_velocity
        self.alpha_rel_rmsf = self.configs.loss.weight.alpha_rel_rmsf
        self.alpha_local_rmsf = self.configs.loss.weight.alpha_local_rmsf
        self.alpha_ensemble = self.configs.loss.weight.alpha_ensemble
        self.alpha_lig_bond = self.configs.loss.weight.alpha_lig_bond
        self.alpha_acf = self.configs.loss.weight.alpha_acf


        self.lddt_radius = {
            "is_nucleotide_threshold": 30.0,
            "is_not_nucleotide_threshold": 15.0,
        }

        self.loss_weight = {
            # confidence
            "plddt_loss": self.alpha_confidence * self.alpha_except_pae,
            "pde_loss": self.alpha_confidence * self.alpha_except_pae,
            "resolved_loss": self.alpha_confidence * self.alpha_except_pae,
            "pae_loss": self.alpha_confidence * self.alpha_pae,
            # diffusion
            "mse_loss": self.alpha_diffusion,
            "bond_loss": self.alpha_diffusion * self.alpha_bond,
            "smooth_lddt_loss": self.alpha_diffusion
            * self.weight_smooth_lddt,  # Different from AF3 appendix eq(6), where smooth_lddt has no weight
            # distogram
            "distogram_loss": self.alpha_distogram,
            "center_loss": self.alpha_diffusion * self.alpha_center, # ligand center
            "lig_bond_loss": self.alpha_diffusion * self.alpha_lig_bond,
            
            "velocity_loss": self.alpha_diffusion * self.alpha_velocity * self.alpha_ensemble, # all atom velocity
            "rmsf_loss": self.alpha_diffusion * self.alpha_rmsf * self.alpha_ensemble, # 0.25(no scaling) # all atom rmsf
            "rel_rmsf_loss": self.alpha_diffusion * self.alpha_rel_rmsf * self.alpha_ensemble, # alpha C pairwise distance rmsf
            "local_rmsf_loss": self.alpha_diffusion * self.alpha_local_rmsf * self.alpha_ensemble, # atoms within one residue pairwise distance rmsf
            "acf_loss": self.alpha_diffusion * self.alpha_acf * self.alpha_ensemble,  # Cα lag-1 ACF
        }

        # Loss
        # self.plddt_loss = PLDDTLoss(**configs.loss.plddt, **self.lddt_radius)
        # self.pde_loss = PDELoss(**configs.loss.pde)
        # self.resolved_loss = ExperimentallyResolvedLoss(**configs.loss.resolved)
        # self.pae_loss = PAELoss(**configs.loss.pae)
        # self.distogram_loss = DistogramLoss(**configs.loss.distogram)
        
        self.mse_loss = MSELoss(
            **configs.loss.diffusion.mse,
            mse_align=getattr(configs.loss, "mse_align", False),
        )
        self.bond_loss = BondLoss(**configs.loss.diffusion.bond)
        self.smooth_lddt_loss = SmoothLDDTLoss(**configs.loss.diffusion.smooth_lddt)

        self.rmsf_loss = RMSFLoss(**configs.loss.diffusion.mse)
        self.rel_rmsf_loss = RelRMSFLoss(**configs.loss.diffusion.smooth_lddt)
        self.local_rmsf_loss = LocalRMSFLoss()
        self.velocity_loss = VelocityLoss(**configs.loss.diffusion.mse)
        self.lig_bond_loss = BondLoss(**configs.loss.diffusion.bond)
        self.center_loss = CenterLoss(reduction="mean")
        self.acf_loss = ACFLoss(max_lag=self.configs.loss.weight.max_acf_lag)

        # TICA-space dynamics losses (transition + population + autocorrelation)
        tica_cfg = getattr(configs.loss, "tica_dynamics", None)
        self.alpha_tica_dynamics = getattr(configs.loss.weight, "alpha_tica_dynamics", 0.0)
        if tica_cfg is not None and self.alpha_tica_dynamics > 0:
            self.tica_dynamics_loss = TICADynamicsLoss(
                weight_transition=getattr(tica_cfg, "weight_transition", 1.0),
                weight_population=getattr(tica_cfg, "weight_population", 0.1),
                weight_autocorrelation=getattr(tica_cfg, "weight_autocorrelation", 0.5),
                max_acf_lag=self.configs.loss.weight.max_acf_lag,
            )
            self.loss_weight["tica_dynamics_loss"] = (
                self.alpha_diffusion * self.alpha_tica_dynamics * self.alpha_ensemble
            )
        else:
            self.tica_dynamics_loss = None
        # Debug: log TICA loss init status
        import logging as _logging
        _logging.getLogger(__name__).info(
            f"[Loss.__init__] tica_dynamics_loss={self.tica_dynamics_loss is not None}, "
            f"alpha_tica_dynamics={self.alpha_tica_dynamics}, tica_cfg={tica_cfg}"
        )

    def calculate_label(
        self,
        feat_dict: dict[str, Any],
        label_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """calculate true distance, and atom pair mask

        Args:
            feat_dict (dict): Feature dictionary containing additional features.
            label_dict (dict): Label dictionary containing ground truth data.

        Returns:
            label_dict (dict): with the following updates:
                distance (torch.Tensor): true atom-atom distance.
                    [..., N_atom, N_atom]
                distance_mask (torch.Tensor): atom-atom mask indicating whether true distance exists.
                    [..., N_atom, N_atom]
        """
        # Distance mask
        distance_mask = (
            label_dict["coordinate_mask"][..., None]
            * label_dict["coordinate_mask"][..., None, :]
        )
        # Distances for all atom pairs
        # Note: we convert to bf16 for saving cuda memory, if performance drops, do not convert it
        traj_len = label_dict["traj_len"]
        traj = []
        for i in range(traj_len):
            traj.append(label_dict[f"coordinate_{i}"])
        x_original = torch.stack(traj, dim=0)

        distance = (
            cdist(x_original, x_original) * distance_mask
        ).to(torch.bfloat16)  # [..., N_atom, N_atom]

        lddt_mask = compute_lddt_mask(
            true_distance=distance,
            distance_mask=distance_mask,
            is_nucleotide=feat_dict["is_rna"].bool() + feat_dict["is_dna"].bool(),
            **self.lddt_radius,
        )

        label_dict["lddt_mask"] = lddt_mask
        label_dict["distance_mask"] = distance_mask
        # if not self.configs.loss_metrics_sparse_enable:
        #     label_dict["distance"] = distance
        del distance, distance_mask, lddt_mask
        return label_dict

    def calculate_prediction(
        self,
        pred_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """get more predictions used for calculating difference losses

        Args:
            pred_dict (dict[str, torch.Tensor]): raw prediction dict given by the model

        Returns:
            dict[str, torch.Tensor]: updated predictions
        """
        if not self.configs.loss_metrics_sparse_enable:
            pred_dict["distance"] = torch.cdist(
                pred_dict["coordinate"], pred_dict["coordinate"]
            ).to(
                pred_dict["coordinate"].dtype
            )  # [..., N_atom, N_atom]

        pred_dict["coordinate_inverse"] = pred_dict["coordinate"]
        pred_dict["coordinate_gt_original"] = pred_dict["coordinate_gt"]
        return pred_dict

    def aggregate_losses(
        self, loss_fns: dict, has_valid_resolution: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Aggregates multiple loss functions and their respective metrics.

        Args:
            loss_fns (dict): Dictionary of loss functions to be aggregated.
            has_valid_resolution (Optional[torch.Tensor]): Tensor indicating valid resolutions. Defaults to None.

        Returns:
            tuple[torch.Tensor, dict]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - all_metrics (dict): Dictionary containing all metrics.
        """
        cum_loss = 0.0
        all_metrics = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.loss_weight[loss_name]
            loss_outputs = loss_fn()
            if isinstance(loss_outputs, tuple):
                loss, metrics = loss_outputs
            else:
                assert isinstance(loss_outputs, torch.Tensor)
                loss, metrics = loss_outputs, {}

            all_metrics.update(
                {f"{loss_name}/{key}": val for key, val in metrics.items()}
            )
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"{loss_name} loss is NaN. Skipping...")
                loss = torch.zeros_like(loss)

            all_metrics[loss_name] = loss.detach().clone()
            all_metrics[f"weighted_{loss_name}"] = weight * loss.detach().clone()

            cum_loss = cum_loss + weight * loss
        all_metrics["loss"] = cum_loss.detach().clone()

        return cum_loss, all_metrics

    def calculate_losses(
        self,
        feat_dict: dict[str, Any],
        pred_dict: dict[str, torch.Tensor],
        label_dict: dict[str, Any],
        mode: str = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Calculate the cumulative loss and aggregated metrics for the given predictions and labels.

        Args:
            feat_dict (dict[str, Any]): Feature dictionary containing additional features.
            pred_dict (dict[str, torch.Tensor]): Prediction dictionary containing model outputs.
            label_dict (dict[str, Any]): Label dictionary containing ground truth data.
            mode (str): Mode of operation ('train', 'eval', 'inference'). Defaults to 'train'.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - metrics (dict[str, torch.Tensor]): Dictionary containing aggregated metrics.
        """
        assert mode in ["train", "eval", "inference"]
        if mode == "train":
            # Confidence Loss: use mini-rollout coordinates
            confidence_coordinate = "coordinate_mini"
            if not self.configs.train_confidence_only:
                # Scale diffusion loss with noise-level
                diffusion_per_sample_scale = (
                    pred_dict["noise_level"] ** 2 + self.configs.sigma_data**2
                ) / (self.configs.sigma_data * pred_dict["noise_level"]) ** 2

        else:
            # Confidence Loss: use diffusion coordinates
            confidence_coordinate = "coordinate"
            # No scale is required
            diffusion_per_sample_scale = None

        # if False: # diffusion_per_sample_scale is not None:
        #     loss_mask = (pred_dict["noise_level"] > 1e-3).float() # t_hat_noise_level [N_frame, N_sample]
        #     loss_mask[1:-1] = 1.0
        #     if diffusion_per_sample_scale is not None:
        #         diffusion_per_sample_scale = diffusion_per_sample_scale * loss_mask
    
        #     rmsf_per_sample_scale = diffusion_per_sample_scale[1]
        # else:
        # loss_mask = None
        if "noise_level" in pred_dict:
            loss_mask = (pred_dict["noise_level"] > 1e-3).float() # t_hat_noise_level [N_frame, N_sample]
        else:
            loss_mask = None
        rmsf_per_sample_scale = None
            
        atom_bond_mask = feat_dict["atom_bond_mask"]
        is_ligand = feat_dict["is_ligand"]
        lig_bond_mask = is_ligand.unsqueeze(1) * is_ligand.unsqueeze(0) * atom_bond_mask
        feat_dict["lig_bond_mask"] = lig_bond_mask

        if self.configs.train_confidence_only and mode == "train":
            # Skip Diffusion Loss and distogram loss
            loss_fns = {}
        else:
            # Diffusion Loss: SmoothLDDTLoss / BondLoss / MSELoss
            loss_fns = {}
            loss_fns.update(
                {
                    "smooth_lddt_loss": lambda: self.smooth_lddt_loss.dense_forward(
                        pred_coordinate=pred_dict["coordinate_inverse"],
                        true_coordinate=pred_dict["coordinate_gt_original"],
                        is_ligand=feat_dict["is_ligand"],
                        lddt_mask=label_dict["lddt_mask"],
                        loss_mask=loss_mask
                    )  # it's faster is not OOM
                }
            )

            loss_fns.update(
                {
                    "bond_loss": lambda: self.bond_loss.sparse_forward(
                            pred_coordinate=pred_dict["coordinate_inverse"],
                            true_coordinate=pred_dict["coordinate_gt_original"],
                            distance_mask=label_dict["distance_mask"],
                            bond_mask=feat_dict["bond_mask"],
                            per_sample_scale=diffusion_per_sample_scale,
                    ),
                    "mse_loss": lambda: self.mse_loss(
                        pred_coordinate=pred_dict["coordinate"],
                        true_coordinate=pred_dict["coordinate_gt"],
                        coordinate_mask=label_dict["coordinate_mask"],
                        is_rna=feat_dict["is_rna"],
                        is_dna=feat_dict["is_dna"],
                        is_ligand=feat_dict["is_ligand"],
                        per_sample_scale=diffusion_per_sample_scale,
                    ),
                }
            )
            loss_fns.update(
                    {
                        "lig_bond_loss": lambda: self.lig_bond_loss.sparse_forward(
                                pred_coordinate=pred_dict["coordinate_inverse"],
                                true_coordinate=pred_dict["coordinate_gt_original"],
                                distance_mask=label_dict["distance_mask"],
                                bond_mask=feat_dict["lig_bond_mask"],
                                per_sample_scale=diffusion_per_sample_scale,
                        ),
                        "center_loss": lambda: self.center_loss(
                            pred_coordinate=pred_dict["coordinate_inverse"],
                            true_coordinate=pred_dict["coordinate_gt_original"],
                            coordinate_mask=label_dict["coordinate_mask"],
                            is_ligand=feat_dict["is_ligand"],
                            per_sample_scale=diffusion_per_sample_scale,
                        ),
                    }
                )

            # Ensemble losses: only valid when traj_len >= 3 (RMSF/velocity/ACF
            # need variance over multiple frames). For shorter trajectories
            # (traj_len in {1, 2}) only the per-frame losses above apply.
            #
            # DyDiff note: true_coordinate for ensemble metrics uses the *original*
            # (non-Dynamics-transformed) GT, because these losses measure physical
            # properties of real trajectories. The denoiser output ("coordinate")
            # is in Dynamics space when DyDiff is on, but the Dynamics transform
            # preserves ensemble variance, so Dynamics-space predictions are an
            # acceptable approximation. The GT, however, must be real coordinates.
            _tl = label_dict["traj_len"]
            traj_len_int = int(_tl.item() if torch.is_tensor(_tl) else _tl)
            if traj_len_int >= 3:
                loss_fns.update(
                    {
                        "rmsf_loss": lambda: self.rmsf_loss(
                            pred_coordinate=pred_dict["coordinate_inverse"],
                            true_coordinate=pred_dict["coordinate_gt_original"],
                            coordinate_mask=label_dict["coordinate_mask"],
                            is_rna=feat_dict["is_rna"],
                            is_dna=feat_dict["is_dna"],
                            is_ligand=feat_dict["is_ligand"],
                            per_sample_scale=rmsf_per_sample_scale  # [N_sample]
                        ),
                        "rel_rmsf_loss": lambda: self.rel_rmsf_loss(
                            pred_coordinate=pred_dict["coordinate_inverse"],
                            true_coordinate=pred_dict["coordinate_gt_original"],
                            atom_to_tokatom_idx=feat_dict["atom_to_tokatom_idx"],
                            is_ligand=feat_dict["is_ligand"],
                            lddt_mask=label_dict["lddt_mask"],
                            per_sample_scale=rmsf_per_sample_scale # [N_sample]
                        ),
                        "local_rmsf_loss": lambda: self.local_rmsf_loss(
                            pred_coordinate=pred_dict["coordinate_inverse"],
                            true_coordinate=pred_dict["coordinate_gt_original"],
                            atom_to_token_idx=feat_dict["atom_to_token_idx"],
                            is_ligand=feat_dict["is_ligand"],
                            lddt_mask=label_dict["lddt_mask"],
                            per_sample_scale=rmsf_per_sample_scale # [N_sample]
                        ),
                        "velocity_loss": lambda: self.velocity_loss(
                            pred_coordinate=pred_dict["coordinate_inverse"],
                            true_coordinate=pred_dict["coordinate_gt_original"],
                            coordinate_mask=label_dict["coordinate_mask"],
                            is_rna=feat_dict["is_rna"],
                            is_dna=feat_dict["is_dna"],
                            is_ligand=feat_dict["is_ligand"],
                            per_sample_scale=rmsf_per_sample_scale # [N_sample]
                        ),
                        "acf_loss": lambda: self.acf_loss(
                            pred_coordinate=pred_dict["coordinate_inverse"],
                            true_coordinate=pred_dict["coordinate_gt_original"],
                            coordinate_mask=label_dict["coordinate_mask"],
                            atom_to_tokatom_idx=feat_dict["atom_to_tokatom_idx"],
                            is_ligand=feat_dict["is_ligand"],
                        ),
                    }
                )

            # TICA-space dynamics loss (transition + population + autocorrelation).
            # Also requires traj_len >= 3 since transition matrices and ACF on
            # 1-2 frames are undefined / degenerate.
            _has_msm = label_dict.get("has_msm", torch.tensor(False)).item()
            if not _has_msm and self.tica_dynamics_loss is not None:
                import logging as _log; _log.getLogger(__name__).debug("[TICA] has_msm=False, skipping tica_dynamics_loss")
            if (
                self.tica_dynamics_loss is not None
                and _has_msm
                and traj_len_int >= 3
            ):
                def _tica_dynamics_loss_fn():
                    _ftr = label_dict["msm_frame_time_gap_ratio"]
                    frame_time_gap_ratio = _ftr.item() if torch.is_tensor(_ftr) else float(_ftr)
                    loss, sub_metrics = self.tica_dynamics_loss(
                        pred_coordinate=pred_dict["coordinate_inverse"],
                        noise_level=pred_dict.get("noise_level"),
                        atom_to_tokatom_idx=feat_dict["atom_to_tokatom_idx"],
                        is_ligand=feat_dict["is_ligand"],
                        msm_state_labels=label_dict["msm_state_labels"],
                        msm_tica_mean=label_dict["msm_tica_mean"],
                        msm_tica_components=label_dict["msm_tica_components"],
                        msm_tica_eigenvalues=label_dict["msm_tica_eigenvalues"],
                        msm_cluster_centers=label_dict["msm_cluster_centers"],
                        msm_cluster_log_vars=label_dict["msm_cluster_log_vars"],
                        msm_transition_matrix=label_dict["msm_transition_matrix"],
                        msm_stationary_distribution=label_dict["msm_stationary_distribution"],
                        msm_traj_population=label_dict["msm_traj_population"],
                        msm_pair_indices_i=label_dict["msm_pair_indices_i"],
                        msm_pair_indices_j=label_dict["msm_pair_indices_j"],
                        frame_time_gap_ratio=frame_time_gap_ratio,
                        _diag_traj_name=(lambda _n: _n[0] if isinstance(_n, (list, tuple)) and _n else _n)(label_dict.get("msm_traj_name")),
                    )
                    return loss, sub_metrics

                loss_fns["tica_dynamics_loss"] = _tica_dynamics_loss_fn

        cum_loss, metrics = self.aggregate_losses(loss_fns)
        return cum_loss, metrics

    def forward(
        self,
        feat_dict: dict[str, Any],
        pred_dict: dict[str, torch.Tensor],
        label_dict: dict[str, Any],
        mode: str = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass for calculating the cumulative loss and aggregated metrics.

        Args:
            feat_dict (dict[str, Any]): Feature dictionary containing additional features.
            pred_dict (dict[str, torch.Tensor]): Prediction dictionary containing model outputs.
            label_dict (dict[str, Any]): Label dictionary containing ground truth data.
            mode (str): Mode of operation ('train', 'eval', 'inference'). Defaults to 'train'.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - losses (dict[str, torch.Tensor]): Dictionary containing aggregated metrics.
        """
        diffusion_chunk_size = self.configs.loss.diffusion_chunk_size_outer
        assert mode in ["train", "eval", "inference"]
        # Pre-computations
        with torch.no_grad():
            label_dict = self.calculate_label(feat_dict, label_dict)

        pred_dict = self.calculate_prediction(pred_dict)

        if diffusion_chunk_size <= 0:
            # Calculate losses
            cum_loss, losses = self.calculate_losses(
                feat_dict=feat_dict,
                pred_dict=pred_dict,
                label_dict=label_dict,
                mode=mode,
            )
        else:
            if "coordinate" in pred_dict:
                N_sample = pred_dict["coordinate"].shape[-3]
            elif self.configs.train_confidence_only:
                N_sample = pred_dict["coordinate_mini"].shape[-3]
            else:
                raise KeyError("Missing key: coordinate (in pred_dict).")
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            cum_loss = 0.0
            losses = {}
            for i in range(no_chunks):
                cur_sample_num = min(
                    diffusion_chunk_size, N_sample - i * diffusion_chunk_size
                )
                pred_dict_i = {}
                for key, value in pred_dict.items():
                    if key in ["coordinate"] and mode == "train":
                        pred_dict_i[key] = value[
                            i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                            :,
                            :,
                        ]
                    elif (
                        key in ["coordinate", "plddt", "pae", "pde", "resolved"]
                        and mode != "train"
                    ):
                        pred_dict_i[key] = value[
                            i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                            :,
                            :,
                        ]
                    elif key == "noise_level":
                        pred_dict_i[key] = value[
                            i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size
                        ]
                    else:
                        pred_dict_i[key] = value
                pred_dict_i = self.calculate_prediction(pred_dict_i)
                cum_loss_i, losses_i = self.calculate_losses(
                    feat_dict=feat_dict,
                    pred_dict=pred_dict_i,
                    label_dict=label_dict,
                    mode=mode,
                )
                cum_loss += cum_loss_i * cur_sample_num
                # Aggregate metrics
                for key, value in losses_i.items():
                    if key in losses:
                        losses[key] += value * cur_sample_num
                    else:
                        losses[key] = value * cur_sample_num
            cum_loss /= N_sample
            for key in losses.keys():
                losses[key] /= N_sample

        return cum_loss, losses
