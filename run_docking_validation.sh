#!/bin/bash
set -e
cd /home/enriquefbrt/TFG
source .venv/bin/activate

GNINA=docking/bin/gnina

echo "=== Extract only 4E3 ligand for autobox ==="
/usr/bin/python3.10 - <<'PYEOF'
lines = open('docking/receptor/4YLL_ligand_ref.pdb').readlines()
with open('docking/receptor/4E3_autobox.pdb', 'w') as f:
    for l in lines:
        if l.startswith('HETATM') and l[17:20].strip() == '4E3':
            f.write(l)
    f.write('END\n')
n = sum(1 for l in open('docking/receptor/4E3_autobox.pdb') if l.startswith('HETATM'))
print(f"4E3 atoms extracted: {n}")
print(f"Centroid will define the binding box (ATP pocket of DYRK1A)")
PYEOF

echo ""
echo "=== Run GNINA: dock compound 1 (IC50=41nM) into DYRK1A (4YLL) ==="
echo "Using autobox around crystallographic ligand 4E3..."

$GNINA \
    --receptor docking/receptor/4YLL_clean.pdb \
    --ligand docking/ligands/compound1_ref.sdf \
    --autobox_ligand docking/receptor/4E3_autobox.pdb \
    --autobox_add 4 \
    --out docking/ligands/compound1_docked.sdf \
    --exhaustiveness 16 \
    --num_modes 9 \
    --cpu 8 \
    --no_gpu \
    2>&1

echo ""
echo "=== Parse docking results ==="
/usr/bin/python3.10 - <<'PYEOF'
# Read docked SDF and extract scores
with open('docking/ligands/compound1_docked.sdf') as f:
    content = f.read()

poses = content.split('$$$$')
print(f"Poses generated: {len([p for p in poses if p.strip()])}")
print()
print(f"{'Pose':>5}  {'Vina score':>12}  {'CNNaffinity':>12}  {'CNNscore':>10}")
print("-" * 50)
for i, pose in enumerate(poses, 1):
    if not pose.strip():
        continue
    lines = pose.strip().split('\n')
    vina, cnn_aff, cnn_score = None, None, None
    for j, line in enumerate(lines):
        if '>  <minimizedAffinity>' in line and j+1 < len(lines):
            vina = lines[j+1].strip()
        if '>  <CNNaffinity>' in line and j+1 < len(lines):
            cnn_aff = lines[j+1].strip()
        if '>  <CNNscore>' in line and j+1 < len(lines):
            cnn_score = lines[j+1].strip()
    if vina:
        print(f"  {i:>3}  {vina:>12}  {cnn_aff or 'N/A':>12}  {cnn_score or 'N/A':>10}")

print()
print("Reference: González García 2025 (Glide XP on 4YLL): -13.99 kcal/mol")
print("Note: Vina/GNINA and Glide XP scores are not directly comparable.")
PYEOF
