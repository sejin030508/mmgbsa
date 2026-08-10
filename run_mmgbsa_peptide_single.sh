#!/bin/bash

set -euo pipefail

usage() {
    echo "Usage: $0 protein.pdb output_dir [receptor_chain peptide_chain]"
    echo ""
    echo "Examples:"
    echo "  $0 complex.pdb peptide_mmgbsa_run"
    echo "  $0 complex.pdb peptide_mmgbsa_run A B"
    echo ""
    echo "output_dir is required. The script refuses to run if output_dir already"
    echo "exists and is not empty, so previous results are not overwritten."
    echo ""
    echo "If receptor_chain and peptide_chain are omitted, the script expects exactly"
    echo "two chains and assigns the larger chain as receptor and the smaller chain"
    echo "as peptide."
}

if [ $# -ne 2 ] && [ $# -ne 4 ]; then
    usage
    exit 1
fi

INPUT_PDB=$(python -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$1")
OUTPUT_DIR=$2
RECEPTOR_CHAIN=${3:-}
PEPTIDE_CHAIN=${4:-}

if [ ! -f "$INPUT_PDB" ]; then
    echo "Error: input PDB not found: $INPUT_PDB"
    exit 1
fi

if [ -n "$RECEPTOR_CHAIN" ] && [ -z "$PEPTIDE_CHAIN" ]; then
    echo "Error: peptide_chain is required when receptor_chain is provided."
    usage
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=$(python -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$OUTPUT_DIR")
RESULT_FILE="$OUTPUT_DIR/peptide_MMPBSA_results.dat"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-md_mmgbsa}"
fi

if [ -e "$OUTPUT_DIR" ] && [ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then
    echo "Error: output directory already exists and is not empty:"
    echo "  $OUTPUT_DIR"
    echo "Choose a new output_dir so previous MM/GBSA results are not overwritten."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "=========================================================="
echo "Protein-peptide MM/GBSA single-system pipeline"
echo "Input PDB: $INPUT_PDB"
echo "Work dir : $OUTPUT_DIR"
echo "=========================================================="

cp "$INPUT_PDB" input_complex.pdb

echo "[1/5] Cleaning input PDB with pdb4amber..."
pdb4amber -i input_complex.pdb -o complex_clean.pdb --nohyd > pdb4amber.log 2>&1

echo "[2/5] Splitting two chains into receptor.pdb and peptide.pdb..."
python - "$RECEPTOR_CHAIN" "$PEPTIDE_CHAIN" <<'PY'
import sys
from collections import defaultdict

receptor_chain = sys.argv[1] or None
peptide_chain = sys.argv[2] or None

input_pdb = "complex_clean.pdb"
water_and_ions = {
    "HOH", "WAT", "SOL", "TIP3",
    "NA", "Na+", "CL", "Cl-", "K", "MG", "CA", "ZN",
}

records = []
chain_atoms = defaultdict(int)
chain_residues = defaultdict(set)

with open(input_pdb) as handle:
    for line in handle:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        resname = line[17:20].strip()
        if resname in water_and_ions:
            continue
        chain = line[21].strip()
        if not chain:
            raise SystemExit("Error: found atoms without chain ID. Please assign chain IDs first.")
        resid = (line[22:26].strip(), line[26].strip(), resname)
        records.append((chain, line))
        chain_atoms[chain] += 1
        chain_residues[chain].add(resid)

if not records:
    raise SystemExit("Error: no protein/peptide atoms found in complex_clean.pdb.")

chains = sorted(chain_atoms)

if receptor_chain and peptide_chain:
    if receptor_chain not in chain_atoms:
        raise SystemExit(f"Error: receptor chain '{receptor_chain}' not found. Found chains: {', '.join(chains)}")
    if peptide_chain not in chain_atoms:
        raise SystemExit(f"Error: peptide chain '{peptide_chain}' not found. Found chains: {', '.join(chains)}")
    if receptor_chain == peptide_chain:
        raise SystemExit("Error: receptor_chain and peptide_chain must be different.")
else:
    if len(chains) != 2:
        raise SystemExit(
            "Error: automatic mode requires exactly two chains. "
            f"Found chains: {', '.join(chains)}. "
            "Run again with receptor_chain and peptide_chain arguments."
        )
    ranked = sorted(
        chains,
        key=lambda chain: (len(chain_residues[chain]), chain_atoms[chain]),
        reverse=True,
    )
    receptor_chain, peptide_chain = ranked[0], ranked[1]

def write_chain(path, selected_chain):
    with open(path, "w") as out:
        for chain, line in records:
            if chain == selected_chain:
                out.write(line)
        out.write("TER\nEND\n")

def write_complex(path, first_chain, second_chain):
    with open(path, "w") as out:
        for selected_chain in (first_chain, second_chain):
            for chain, line in records:
                if chain == selected_chain:
                    out.write(line)
            out.write("TER\n")
        out.write("END\n")

write_chain("receptor.pdb", receptor_chain)
write_chain("peptide.pdb", peptide_chain)
write_complex("complex_ordered.pdb", receptor_chain, peptide_chain)

with open("chain_assignment.txt", "w") as out:
    out.write(f"receptor_chain={receptor_chain}\n")
    out.write(f"peptide_chain={peptide_chain}\n")
    for chain in chains:
        out.write(
            f"chain_{chain}_residues={len(chain_residues[chain])} "
            f"chain_{chain}_atoms={chain_atoms[chain]}\n"
        )

print(f"Receptor chain: {receptor_chain} ({len(chain_residues[receptor_chain])} residues)")
print(f"Peptide chain : {peptide_chain} ({len(chain_residues[peptide_chain])} residues)")
PY

echo "[3/5] Building AMBER topology files with tleap..."
cat > tleap.in <<'EOF'
source leaprc.protein.ff14SB
source leaprc.water.tip3p
source leaprc.phosaa10

rec = loadpdb receptor.pdb
pep = loadpdb peptide.pdb
comp = combine {rec pep}

saveamberparm comp complex.prmtop complex.inpcrd
saveamberparm rec receptor.prmtop receptor.inpcrd
saveamberparm pep peptide.prmtop peptide.inpcrd

solvatebox comp TIP3PBOX 10.0
addions comp Na+ 0
addions comp Cl- 0
addions comp Na+ 40
addions comp Cl- 40
saveamberparm comp complex_solv.prmtop complex_solv.inpcrd
quit
EOF
tleap -f tleap.in > tleap.log 2>&1

echo "[4/5] Running OpenMM MD..."
python "$SCRIPT_DIR/run_md.py" "$OUTPUT_DIR" > openmm.log 2>&1

echo "[5/5] Running MM/GBSA..."
MMPBSA.py -O -i "$SCRIPT_DIR/mmpbsa.in" \
    -sp complex_solv.prmtop \
    -cp complex.prmtop \
    -rp receptor.prmtop \
    -lp peptide.prmtop \
    -y production.dcd > mmpbsa.log 2>&1

if [ -f "FINAL_RESULTS_MMPBSA.dat" ]; then
    cp FINAL_RESULTS_MMPBSA.dat "$RESULT_FILE"
    echo "Done: $RESULT_FILE"
else
    echo "Error: FINAL_RESULTS_MMPBSA.dat was not generated. Check mmpbsa.log."
    exit 1
fi
