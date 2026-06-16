#!/bin/bash
set -e
cd /home/enriquefbrt/TFG
source .venv/bin/activate

echo "=== Step 1: Install Python deps ==="
pip install pdbfixer meeko -q
echo "pdbfixer + meeko installed"

echo ""
echo "=== Step 2: Download GNINA binary (GPU-enabled) ==="
mkdir -p docking/bin
GNINA_URL="https://github.com/gnina/gnina/releases/download/v1.1/gnina"
if [ ! -f docking/bin/gnina ]; then
    wget -q --show-progress -O docking/bin/gnina "$GNINA_URL"
    chmod +x docking/bin/gnina
    echo "GNINA downloaded: $(docking/bin/gnina --version 2>&1 | head -1)"
else
    echo "GNINA already present"
fi

echo ""
echo "=== Step 3: Download PDB 4YLL (DYRK1A + ligand 4E3) ==="
mkdir -p docking/receptor
if [ ! -f docking/receptor/4YLL.pdb ]; then
    wget -q -O docking/receptor/4YLL.pdb "https://files.rcsb.org/download/4YLL.pdb"
    echo "PDB 4YLL downloaded ($(wc -l < docking/receptor/4YLL.pdb) lines)"
else
    echo "4YLL.pdb already present"
fi

echo ""
echo "All downloads done."
