from pdbfixer import PDBFixer
from openmm.app import PDBFile

# 1. Load file
print("[1/7] Loading PDB file...")
fixer = PDBFixer(filename='../data/coreset_classified/3/3qgy/3qgy_protein.pdb')

# 2. Find missing residues
print("[2/7] Finding missing residues...")
fixer.findMissingResidues()

# 3. Replace non-standard residues
print("[3/7] Replacing non-standard residues...")
fixer.findNonstandardResidues()
fixer.replaceNonstandardResidues()

# 4. Remove heterogens
print("[4/7] Removing heterogens (water, etc.)...")
fixer.removeHeterogens(True)

# 5. Add missing atoms
print("[5/7] Adding missing atoms...")
fixer.findMissingAtoms()
fixer.addMissingAtoms()

# 6. Add hydrogens
print("[6/7] Adding hydrogens at pH 7.0...")
fixer.addMissingHydrogens(7.0)

# 7. Save file
print("[7/7] Saving output file...")
PDBFile.writeFile(fixer.topology, fixer.positions, open('3qgy_protein_fixed.pdb', 'w'))

print("Done!")