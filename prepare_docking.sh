#!/bin/bash
set -e
cd /home/enriquefbrt/TFG
source .venv/bin/activate
mkdir -p docking/receptor docking/ligands

echo "=== Receptor preparation (4YLL) ==="
/usr/bin/python3.10 - <<'PYEOF'
from pdbfixer import PDBFixer
from openmm.app import PDBFile

fixer = PDBFixer(filename='docking/receptor/4YLL.pdb')
print("Chains:", [c.id for c in fixer.topology.chains()])

fixer.removeHeterogens(keepWater=False)
fixer.findMissingResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(7.4)
print("Heterogens removed, H added at pH 7.4")

with open('docking/receptor/4YLL_clean.pdb', 'w') as f:
    PDBFile.writeFile(fixer.topology, fixer.positions, f)
print("Saved: docking/receptor/4YLL_clean.pdb")
PYEOF

echo ""
echo "=== Extract reference ligand from 4YLL ==="
/usr/bin/python3.10 - <<'PYEOF'
lines = open('docking/receptor/4YLL.pdb').readlines()
exclude = {'HOH', 'WAT', 'NA', 'CL', 'MG', 'ZN', 'CA', 'K', 'SO4'}
resnames = set()
for l in lines:
    if l.startswith('HETATM'):
        rn = l[17:20].strip()
        resnames.add(rn)
print("HETATM residues:", resnames)

lig_resnames = resnames - exclude
print("Ligand candidates:", lig_resnames)

with open('docking/receptor/4YLL_ligand_ref.pdb', 'w') as f:
    for l in lines:
        if l.startswith('HETATM') and l[17:20].strip() in lig_resnames:
            f.write(l)
    f.write('END\n')
print("Saved: docking/receptor/4YLL_ligand_ref.pdb")

# Print centroid for GNINA box
import re
coords = []
for l in lines:
    if l.startswith('HETATM') and l[17:20].strip() in lig_resnames:
        x, y, z = float(l[30:38]), float(l[38:46]), float(l[46:54])
        coords.append((x, y, z))
cx = sum(c[0] for c in coords) / len(coords)
cy = sum(c[1] for c in coords) / len(coords)
cz = sum(c[2] for c in coords) / len(coords)
print(f"Ligand centroid (binding site): x={cx:.2f}, y={cy:.2f}, z={cz:.2f}")
PYEOF

echo ""
echo "=== Prepare compound 1 (IC50=41nM) as 3D SDF ==="
python - <<'PYEOF'  # uses venv rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

smiles_1 = "CN1CCC(n2cc(-c3cnc4[nH]c(-c5cccc(F)c5)cc4c3)cn2)CC1"
mol = Chem.MolFromSmiles(smiles_1)
mol.SetProp("_Name", "compound1_IC50_41nM")
mol = Chem.AddHs(mol)
params = AllChem.ETKDGv3()
params.randomSeed = 42
AllChem.EmbedMolecule(mol, params)
AllChem.MMFFOptimizeMolecule(mol)

writer = Chem.SDWriter('docking/ligands/compound1_ref.sdf')
writer.write(mol)
writer.close()
print("Saved: docking/ligands/compound1_ref.sdf")

from rdkit.Chem import Descriptors, rdMolDescriptors
print(f"MW: {Descriptors.MolWt(mol):.1f}")
print(f"Formula: {rdMolDescriptors.CalcMolFormula(mol)}")
PYEOF

echo ""
echo "Files ready:"
ls -lh docking/receptor/
ls -lh docking/ligands/
