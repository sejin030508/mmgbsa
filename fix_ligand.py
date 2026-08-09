# fix_ligand.py
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
import sys

def fix_ligand(raw_pdb, smiles, out_sdf):
    # 1. SMILES로부터 올바른 템플릿(결합, 전하 정보) 생성
    template_mol = Chem.MolFromSmiles(smiles)
    if template_mol is None:
        print("❌ 에러: 유효하지 않은 SMILES입니다.")
        sys.exit(1)
        
    # 2. 3D 좌표만 담겨있는 깡통 PDB 로드
    raw_mol = Chem.MolFromPDBFile(raw_pdb, sanitize=False)
    if raw_mol is None:
        print("❌ 에러: PDB 파일을 RDKit으로 로드할 수 없습니다.")
        sys.exit(1)

    # 3. 템플릿의 결합(Bond Order) 정보를 PDB 좌표 분자에 강제 할당
    try:
        fixed_mol = AllChem.AssignBondOrdersFromTemplate(template_mol, raw_mol)
        
        # 4. 수소(H) 추가 및 수소 원자들만의 3D 좌표 자동 생성 (중원자 좌표는 고정)
        fixed_mol = Chem.AddHs(fixed_mol, addCoords=True)
        
        # 5. SDF로 저장
        writer = Chem.SDWriter(out_sdf)
        writer.write(fixed_mol)
        writer.close()
        print(f"✅ 리간드 3D 좌표 맵핑 및 보정 완료: {out_sdf}")
        
    except Exception as e:
        print(f"❌ 에러: PDB 좌표와 SMILES 간 매칭에 실패했습니다. (원자 수 불일치 등)")
        print(str(e))
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True, help="Raw ligand PDB file (from Step 1)")
    parser.add_argument("--smiles", required=True, help="Canonical SMILES of the ligand")
    parser.add_argument("--out", required=True, help="Output SDF file path")
    args = parser.parse_args()
    
    fix_ligand(args.pdb, args.smiles, args.out)