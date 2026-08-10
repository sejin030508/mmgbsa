"""Convert one BioKinema molecular-glue prediction to an all-atom DCD."""

import glob
import os
import sys

import mdtraj as md
import numpy as np
from openmm import (
    Context,
    CustomExternalForce,
    LangevinMiddleIntegrator,
    LocalEnergyMinimizer,
)
from openmm.app import AmberPrmtopFile, NoCutoff, PDBFile
from openmm.unit import kelvin, kilojoules_per_mole, nanometer, picosecond, picoseconds


if len(sys.argv) != 3:
    print(
        "사용법: python npy_to_dcd_moleculeglue_biokinema.py "
        "[작업_폴더] [BioKinema_출력_폴더]"
    )
    sys.exit(1)

work_dir = os.path.expanduser(sys.argv[1])
prediction_dir = os.path.expanduser(sys.argv[2])
pdb_path = os.path.join(work_dir, "complex_reference.pdb")
prmtop_path = os.path.join(work_dir, "complex.prmtop")
output_dcd = os.path.join(work_dir, "biokinema_trajectory.dcd")

npy_files = sorted(
    glob.glob(
        os.path.join(prediction_dir, "**", "*_pred_coordinates.npy"),
        recursive=True,
    )
)
if not npy_files:
    raise FileNotFoundError(
        f"{prediction_dir} 내부에서 BioKinema 좌표 파일을 찾을 수 없습니다."
    )
if len(npy_files) != 1:
    raise RuntimeError(f"BioKinema 좌표 파일이 여러 개입니다: {npy_files}")

prmtop = AmberPrmtopFile(prmtop_path)
pdb = PDBFile(pdb_path)

heavy_indices = []
backbone_indices = []
for atom in prmtop.topology.atoms():
    if atom.element.symbol != "H":
        heavy_indices.append(atom.index)
    if atom.name in {"N", "CA", "C", "O"} and atom.residue.name not in {
        "MOL",
        "LIG",
    }:
        backbone_indices.append(atom.index)

heavy_idx_to_array_idx = {
    heavy_index: array_index
    for array_index, heavy_index in enumerate(heavy_indices)
}

coords_raw = np.load(npy_files[0])
if coords_raw.ndim != 4 or coords_raw.shape[-1] != 3:
    raise ValueError(
        "BioKinema coordinates must have shape [frame, sample, atom, 3], "
        f"got {coords_raw.shape}"
    )
if coords_raw.shape[0] < 1 or coords_raw.shape[1] < 1:
    raise ValueError("BioKinema coordinates contain no frames or samples")

coords = coords_raw[:, 0, :, :] / 10.0
ref_positions = np.array(pdb.positions.value_in_unit(nanometer))
if len(ref_positions) != prmtop.topology.getNumAtoms():
    raise ValueError("complex_reference.pdb and complex.prmtop have different atom counts")
if coords.shape[1] != len(heavy_indices):
    raise ValueError(
        "BioKinema and AMBER heavy-atom counts differ: "
        f"{coords.shape[1]} != {len(heavy_indices)}"
    )

ref_heavy = ref_positions[heavy_indices]
pred_center = np.mean(coords[0], axis=0)
ref_center = np.mean(ref_heavy, axis=0)
centered_pred = coords[0] - pred_center
centered_ref = ref_heavy - ref_center
u_matrix, _, vt_matrix = np.linalg.svd(centered_pred.T @ centered_ref)
rotation = vt_matrix.T @ u_matrix.T
if np.linalg.det(rotation) < 0:
    vt_matrix[2, :] *= -1
    rotation = vt_matrix.T @ u_matrix.T

aligned_coords = np.empty_like(coords)
for frame_index in range(coords.shape[0]):
    aligned_coords[frame_index] = (
        (coords[frame_index] - pred_center) @ rotation.T + ref_center
    )

system = prmtop.createSystem(nonbondedMethod=NoCutoff, constraints=None)
force = CustomExternalForce("k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
force.addGlobalParameter("k", 1000.0 * kilojoules_per_mole / nanometer**2)
for parameter_name in ("x0", "y0", "z0"):
    force.addPerParticleParameter(parameter_name)
for atom_index in backbone_indices:
    force.addParticle(atom_index, [0, 0, 0])
system.addForce(force)

integrator = LangevinMiddleIntegrator(
    300 * kelvin,
    1 / picosecond,
    0.002 * picoseconds,
)
context = Context(system, integrator)
trajectory = np.zeros((coords.shape[0], prmtop.topology.getNumAtoms(), 3))
current_positions = ref_positions.copy()

for frame_index, current_heavy_coords in enumerate(aligned_coords):
    for force_index, backbone_index in enumerate(backbone_indices):
        array_index = heavy_idx_to_array_idx[backbone_index]
        force.setParticleParameters(
            force_index,
            backbone_index,
            current_heavy_coords[array_index],
        )
    force.updateParametersInContext(context)
    current_positions[heavy_indices] = current_heavy_coords
    context.setPositions(current_positions)
    LocalEnergyMinimizer.minimize(context, maxIterations=100)
    current_positions = context.getState(positions=True).getPositions(
        asNumpy=True
    ).value_in_unit(nanometer)
    trajectory[frame_index] = current_positions

md.Trajectory(
    xyz=trajectory,
    topology=md.load_prmtop(prmtop_path),
).save_dcd(output_dcd)
print(f"BioKinema all-atom DCD 생성 완료: {output_dcd}")
