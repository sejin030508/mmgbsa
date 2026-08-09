import os
import sys
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout

## PARAMETER
save_interval = 25000 # 50 ps (25,000 steps) 마다 정보 및 궤적 저장 (총 2000 프레임)[cite: 1]
production_steps = 50000000 # 100 ns (2 fs * 50,000,000)[cite: 1]

# ========================================================
# 1. 초기 설정 및 파일 로드
# ========================================================
if len(sys.argv) < 2:
    print("사용법: python run_md.py [데이터 폴더 경로]")
    sys.exit(1)

data_dir = sys.argv[1]

prmtop_file = os.path.join(data_dir, 'complex_solv.prmtop')
inpcrd_file = os.path.join(data_dir, 'complex_solv.inpcrd')
dcd_file = os.path.join(data_dir, 'production.dcd')
log_file = os.path.join(data_dir, 'md_log.txt')

print(f"[*] 파일 로드 중... ({data_dir})")
if not os.path.exists(prmtop_file):
    print(f"에러: {prmtop_file} 파일을 찾을 수 없습니다.")
    sys.exit(1)

prmtop = AmberPrmtopFile(prmtop_file)
inpcrd = AmberInpcrdFile(inpcrd_file)

# ========================================================
# 2. 시스템 및 제한(Restraint) 세팅
# ========================================================
print("[*] 시스템 세팅 중...")
system = prmtop.createSystem(nonbondedMethod=PME, 
                             nonbondedCutoff=1*nanometer, 
                             constraints=HBonds,
                             rigidWater=True,
                             removeCMMotion=True)

# 단위 변환: 1 kcal/mol/A^2 = 418.4 kJ/mol/nm^2
FC_UNIT = 418.4 

# CustomExternalForce를 이용한 위치 제한 설정
# K는 글로벌 상수(힘의 세기), weight는 원자별 가중치(0 또는 1)
restraint = CustomExternalForce('K * weight * periodicdistance(x, y, z, x0, y0, z0)^2')
restraint.addGlobalParameter('K', 0.0)
restraint.addPerParticleParameter('weight')
restraint.addPerParticleParameter('x0')
restraint.addPerParticleParameter('y0')
restraint.addPerParticleParameter('z0')
system.addForce(restraint)

# 원자 분류 (용매/이온 제외 = 단백질 및 리간드)
solvent_res = ['WAT', 'HOH', 'NA', 'CL', 'Na+', 'Cl-', 'K+', 'MG']
heavy_atoms = []
prot_lig_heavy_atoms = []
backbone_atoms = []

for atom in prmtop.topology.atoms():
    is_heavy = (atom.element.symbol != 'H')
    is_solvent = (atom.residue.name in solvent_res)
    is_backbone = (atom.name in ['CA', 'C', 'N', 'O'])
    
    if is_heavy:
        heavy_atoms.append(atom.index)
        if not is_solvent:
            prot_lig_heavy_atoms.append(atom.index)
            if is_backbone:
                backbone_atoms.append(atom.index)
                
    # 모든 원자를 Force 객체에 추가하되, 초기 weight는 0으로 설정
    restraint.addParticle(atom.index, [0.0, inpcrd.positions[atom.index][0], 
                                       inpcrd.positions[atom.index][1], 
                                       inpcrd.positions[atom.index][2]])

# 온도 및 적분기 설정
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
platform = Platform.getPlatformByName('CUDA')
simulation = Simulation(prmtop.topology, system, integrator, platform)
simulation.context.setPositions(inpcrd.positions)
if inpcrd.boxVectors is not None:
    simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

# weight 업데이트 함수
def update_restraints(active_indices):
    for i in range(system.getNumParticles()):
        pos = inpcrd.positions[i]
        weight = 1.0 if i in active_indices else 0.0
        restraint.setParticleParameters(i, i, [weight, pos[0], pos[1], pos[2]])
    restraint.updateParametersInContext(simulation.context)

# ========================================================
# 3. 4단계 에너지 최소화 (Minimization)
# ========================================================
print("[*] 4단계 에너지 최소화 진행 중...")

# Step 1: 모든 중원자 제한 (5 kcal/mol/A^2)[cite: 1]
print("  -> Step 1: 모든 중원자 고정")
update_restraints(heavy_atoms)
simulation.context.setParameter('K', 5.0 * FC_UNIT)
simulation.minimizeEnergy(maxIterations=5000)

# Step 2: 단백질 및 리간드 중원자 제한 (5 kcal/mol/A^2)[cite: 1]
print("  -> Step 2: 단백질 및 리간드 중원자 고정")
update_restraints(prot_lig_heavy_atoms)
simulation.minimizeEnergy(maxIterations=5000)

# Step 3: 단백질 Backbone 제한 (5 kcal/mol/A^2)[cite: 1]
print("  -> Step 3: 단백질 뼈대(Backbone) 고정")
update_restraints(backbone_atoms)
simulation.minimizeEnergy(maxIterations=5000)

# Step 4: 제한 해제 후 전체 시스템 최소화[cite: 1]
print("  -> Step 4: 모든 제한 해제 및 전체 최적화")
update_restraints([])
simulation.context.setParameter('K', 0.0)
simulation.minimizeEnergy(maxIterations=10000)

# ========================================================
# 4. 점진적 가열 (Heating) - NVT 앙상블[cite: 1]
# ========================================================
print("[*] 점진적 가열 시작 (0K -> 300K, 50 ps, NVT)...")
# Backbone에 2 kcal/mol/A^2 제한 적용[cite: 1]
update_restraints(backbone_atoms)
simulation.context.setParameter('K', 2.0 * FC_UNIT)

simulation.context.setVelocitiesToTemperature(10*kelvin)
steps_per_temp = 500  # 1 ps 마다 온도 상승
total_heating_steps = 25000  # 50 ps

for i in range(1, 51):
    temp = (300.0 / 50) * i
    integrator.setTemperature(temp * kelvin)
    simulation.step(steps_per_temp)

# ========================================================
# 5. 밀도 평형화 (Equilibration) - NPT 앙상블[cite: 1]
# ========================================================
print("[*] 밀도 평형화 시작 (300K, 1 atm, 50 ps, NPT)...")
# NPT 앙상블로 전환하기 위해 Barostat 추가[cite: 1]
barostat = MonteCarloBarostat(1*atmosphere, 300*kelvin)
system.addForce(barostat)
simulation.context.reinitialize(preserveState=True) # 변경된 시스템 반영

# Backbone 제한은 그대로 유지 (2 kcal/mol/A^2)[cite: 1]
simulation.step(25000) # 50 ps 수행

# ========================================================
# 6. 본 시뮬레이션 (Production) - 100 ns[cite: 1]
# ========================================================
print("[*] 본 시뮬레이션 시작 (100 ns)...")
# 모든 제한 해제[cite: 1]
simulation.context.setParameter('K', 0.0)



simulation.reporters.append(StateDataReporter(stdout, save_interval, step=True, 
                                              potentialEnergy=True, temperature=True, 
                                              volume=True, speed=True))
simulation.reporters.append(StateDataReporter(log_file, save_interval, step=True, 
                                              potentialEnergy=True, temperature=True, 
                                              volume=True))
simulation.reporters.append(DCDReporter(dcd_file, save_interval))

simulation.step(production_steps)
print(f"[*] MD 시뮬레이션 완료! 궤적이 {dcd_file} 에 저장되었습니다.")