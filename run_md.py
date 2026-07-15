import os
import sys
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout

# 터미널에서 데이터 폴더 경로를 입력받음
if len(sys.argv) < 2:
    print("사용법: python run_md.py [데이터 폴더 경로]")
    sys.exit(1)

data_dir = sys.argv[1]

# 입력 파일 경로 설정
prmtop_file = os.path.join(data_dir, 'complex_solv.prmtop')
inpcrd_file = os.path.join(data_dir, 'complex_solv.inpcrd')
# 출력 파일 경로 설정
dcd_file = os.path.join(data_dir, 'production.dcd')

print(f"1. 파일 로드 중... ({data_dir})")
if not os.path.exists(prmtop_file):
    print(f"에러: {prmtop_file} 파일을 찾을 수 없습니다.")
    sys.exit(1)

prmtop = AmberPrmtopFile(prmtop_file)
inpcrd = AmberInpcrdFile(inpcrd_file)

print("2. 시스템 세팅 중...")
system = prmtop.createSystem(nonbondedMethod=PME, 
                             nonbondedCutoff=1*nanometer, 
                             constraints=HBonds,
                             rigidWater=True,
                             removeCMMotion=True)

integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
system.addForce(MonteCarloBarostat(1*atmosphere, 300*kelvin))

platform = Platform.getPlatformByName('CUDA')
properties = {} 

# Simulation 객체 생성 시 platform과 properties를 넘겨줍니다.
simulation = Simulation(prmtop.topology, system, integrator, platform, properties)

simulation.context.setPositions(inpcrd.positions)
if inpcrd.boxVectors is not None:
    simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

print("3. 에너지 최소화(Minimization) 진행 중...")
simulation.minimizeEnergy()

print("4. 시뮬레이션 시작...")
simulation.context.setVelocitiesToTemperature(300*kelvin)

simulation.reporters.append(StateDataReporter(stdout, 5000, step=True, potentialEnergy=True, temperature=True, volume=True))
# 궤적 파일을 데이터 폴더에 저장
simulation.reporters.append(DCDReporter(dcd_file, 15000))

# 100,000 스텝 (0.2 ns) 수행
simulation.step(1500000)
print(f"MD 시뮬레이션 완료! 결과가 {dcd_file} 에 저장되었습니다.")