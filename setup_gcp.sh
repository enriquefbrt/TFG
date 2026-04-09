#!/bin/bash
# -----------------------------------------------------
# Script de preparación para instancia de Google Cloud (GCP)
# -----------------------------------------------------
set -e # Detiene el script si cualquier comando falla

echo "=========================================="
echo "    🧪 TFG Molecular Generation Setup     "
echo "=========================================="

echo "[1/4] Instalando dependencias del proyecto..."
# Si no está instalado uv en la máquina, lo instala globalmente
if ! command -v uv &> /dev/null; then
    echo "Instalador 'uv' no encontrado, instalándolo por pip..."
    pip install uv
fi

# Crea el entorno virtual .venv e instala todo lo del pyproject.toml
uv sync

echo ""
if [ ! -f "./data/guacamol_completo_1.6M.csv" ]; then
    echo "[2/4] Descargando el Dataset Oficial GuacaMol v1..."
    mkdir -p data

    # Descargamos los 3 cortes desde Figshare
    echo "Descargando Train (1.27 M)..."
    wget https://ndownloader.figshare.com/files/13612760 -O data/guacamol_v1_train.smiles
    echo "Descargando Valid (~80 k)..."
    wget https://ndownloader.figshare.com/files/13612766 -O data/guacamol_v1_valid.smiles
    echo "Descargando Test (~238 k)..."
    wget https://ndownloader.figshare.com/files/13612757 -O data/guacamol_v1_test.smiles

    echo ""
    echo "[3/4] Fusionando particiones y formateando a CSV..."
    # Usamos el python del entorno virtual que ya tiene Pandas instalado
    .venv/bin/python -c "
import pandas as pd
print('Leyendo y convirtiendo a DataFrames...')
df_train = pd.read_csv('data/guacamol_v1_train.smiles', header=None, names=['smiles'])
df_valid = pd.read_csv('data/guacamol_v1_valid.smiles', header=None, names=['smiles'])
df_test = pd.read_csv('data/guacamol_v1_test.smiles', header=None, names=['smiles'])

print('Fusionando las tres particiones...')
df_all = pd.concat([df_train, df_valid, df_test], ignore_index=True)

output_path = 'data/guacamol_completo_1.6M.csv'
df_all.to_csv(output_path, index=False)
print(f'Guardado con éxito en: {output_path} ({len(df_all)} SMILES exactos obtenidos).')
"

    echo ""
    echo "[4/4] Limpiando archivos temporales txt..."
    rm data/guacamol_v1_train.smiles data/guacamol_v1_valid.smiles data/guacamol_v1_test.smiles
else
    echo "[2,3,4/4] ¡Dataset 'guacamol_completo_1.6M.csv' ya existe! Omitiendo descarga."
fi

echo "=========================================="
echo "  INSTALACIÓN Y DATOS COMPLETADOS AL 100%"
echo "=========================================="
echo "Siguiente paso, activa tu entorno e inicia el tokenizador:"
echo "  $ source .venv/bin/activate"
echo "  $ python src/tfg_molecular_generation/train_tokenizer.py --input_csv data/guacamol_completo_1.6M.csv"
