import numpy as np
import mdtraj as md
from openmm.app import *
from openmm import *
from openmm.unit import *
from scipy.spatial import KDTree
import sys
import os

if len(sys.argv) < 2:
    print("사용법: python npy_to_dcd.py [작업_폴더_경로]")
    sys.exit(1)

work_dir = os.path.expanduser(sys.argv[1])
pdb_path = os.path.join(work_dir, "complex_reference.pdb")
prmtop_path = os.path.join(work_dir, "complex.prmtop")
npy_path = os.path.join(work_dir, "bk_output", "bk_output_0", "bk_output_0_pred_coordinates.npy")
output_dcd = os.path.join(work_dir, "biokinema_trajectory.dcd")

print("\n1. 위상 파일 로드 및 원자 분류 중...")
prmtop = AmberPrmtopFile(prmtop_path)
pdb = PDBFile(pdb_path)

heavy_indices = []
for atom in prmtop.topology.atoms():
    if atom.element.symbol != 'H':
        heavy_indices.append(atom.index)

print("\n2. BioKinema 궤적 데이터 로드 중...")
coords_raw = np.load(npy_path)
coords_s0 = coords_raw[:, 0, :, :] / 10.0
n_frames = coords_s0.shape[0]

print("\n3. [중요] Kabsch 알고리즘으로 구조 1:1 3D 정렬 중...")
ref_positions = np.array(pdb.positions.value_in_unit(nanometers))
ref_heavy = ref_positions[heavy_indices]
bk_f0 = coords_s0[0]

ref_center = np.mean(ref_heavy, axis=0)
bk_center = np.mean(bk_f0, axis=0)

P = bk_f0 - bk_center
Q = ref_heavy - ref_center

H_mat = P.T @ Q
U, S, Vt = np.linalg.svd(H_mat)
R = Vt.T @ U.T

if np.linalg.det(R) < 0:
    Vt[2, :] *= -1
    R = Vt.T @ U.T

max_err = np.max(np.linalg.norm((P @ R.T + ref_center) - ref_heavy, axis=1))
print(f" -> 정렬 완료! (최대 오차 거리: {max_err:.6f} nm)")

aligned_coords_s0 = np.zeros_like(coords_s0)
for i in range(n_frames):
    aligned_coords_s0[i] = (coords_s0[i] - bk_center) @ R.T + ref_center

# =====================================================================
# 🌟 [해결책] 못(Mass=0) 대신 강력한 스프링(Positional Restraint) 장착 🌟
print("\n4. 원자 충돌(Steric Clash) 해소를 위한 OpenMM 시스템 세팅 중...")
system = prmtop.createSystem(nonbondedMethod=NoCutoff, constraints=None)

# 1000 kJ/mol/nm^2 의 강력한 스프링 생성
force = CustomExternalForce("k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
force.addGlobalParameter("k", 1000.0 * kilojoules_per_mole/nanometer**2)
force.addPerParticleParameter("x0")
force.addPerParticleParameter("y0")
force.addPerParticleParameter("z0")

# 무거운 원자들에 스프링 매달기
for idx in heavy_indices:
    force.addParticle(idx, [0, 0, 0])
system.addForce(force)

integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
context = Context(system, integrator)
# =====================================================================

all_atom_trajectory = np.zeros((n_frames, prmtop.topology.getNumAtoms(), 3))
current_positions = ref_positions.copy()

print("\n5. 프레임별 AI 뼈대 유지 및 충돌 완화(Relaxation) 진행 중...")
for i in range(n_frames):
    current_heavy_coords = aligned_coords_s0[i]
    
    # 각 프레임마다 AI가 예측한 위치로 스프링의 기준점(x0, y0, z0) 이동
    for j, idx in enumerate(heavy_indices):
        force.setParticleParameters(j, idx, current_heavy_coords[j])
    force.updateParametersInContext(context)
    
    # 좌표 덮어씌우기 및 에너지 최소화 (이제 뼈대가 살짝 움직이며 16억 kcal 에러를 해소함!)
    current_positions[heavy_indices] = current_heavy_coords
    context.setPositions(current_positions)
    LocalEnergyMinimizer.minimize(context, maxIterations=100)
    
    state = context.getState(positions=True)
    minimized_coords = state.getPositions(asNumpy=True).value_in_unit(nanometers)
    
    all_atom_trajectory[i] = minimized_coords
    current_positions = minimized_coords
    
    if (i + 1) % 20 == 0:
        print(f" -> 처리 완료: {i + 1} / {n_frames} 프레임")

print("\n6. 최종 DCD 궤적 파일 저장 중...")
md_top = md.load_prmtop(prmtop_path)
traj = md.Trajectory(xyz=all_atom_trajectory, topology=md_top)
traj.save_dcd(output_dcd)

print(f"\n🎉 완벽하게 안정화된 All-atom 궤적 파일이 생성되었습니다: {output_dcd}")