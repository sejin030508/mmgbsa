# extract_chains.py
import argparse
import os

def extract_chains(pdb_in, out_dir, chain_a, chain_b, chain_l):
    os.makedirs(out_dir, exist_ok=True)
    
    out_a = os.path.join(out_dir, "Prot_A.pdb")
    out_b = os.path.join(out_dir, "Prot_B.pdb")
    out_l = os.path.join(out_dir, "Ligand_raw.pdb")
    
    with open(pdb_in, 'r') as f_in, \
         open(out_a, 'w') as f_a, \
         open(out_b, 'w') as f_b, \
         open(out_l, 'w') as f_l:
             
        for line in f_in:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21]
                if chain_id == chain_a:
                    f_a.write(line)
                elif chain_id == chain_b:
                    f_b.write(line)
                elif chain_id == chain_l:
                    f_l.write(line)
            elif line.startswith("TER"):
                # TER 레코드는 놔두거나 적절히 분배할 수 있지만, 
                # Amber tleap이 알아서 처리하므로 생략해도 무방합니다.
                pass

    print(f"✅ 추출 완료: {out_a}, {out_b}, {out_l}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract chains from a PDB file.")
    parser.add_argument("input_pdb", help="Input PDB file path")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("-A", required=True, help="Chain ID for Protein A")
    parser.add_argument("-B", required=True, help="Chain ID for Protein B")
    parser.add_argument("-L", required=True, help="Chain ID for Ligand")
    args = parser.parse_args()
    
    extract_chains(args.input_pdb, args.out_dir, args.A, args.B, args.L)