#!/bin/bash
# -----------------------------------------------------
# Script de preparación para instancia de Google Cloud (GCP)
# -----------------------------------------------------
set -e # Detiene el script si cualquier comando falla

echo "=========================================="
echo "       TFG Molecular Generation Setup     "
echo "=========================================="

echo "Instalando dependencias del proyecto..."
# Si no está instalado uv en la máquina, lo instala globalmente
if ! command -v uv &> /dev/null; then
    echo "Instalador 'uv' no encontrado, instalándolo por pip..."
    pip install uv
fi

# Crea el entorno virtual .venv e instala todo lo del pyproject.toml
uv sync

echo "=========================================="
echo "  INSTALACIÓN Y DATOS COMPLETADOS AL 100%"
echo "=========================================="
echo "Siguiente paso, activa tu entorno e inicia el tokenizador:"
echo "  $ source .venv/bin/activate"
echo "  $ python src/tfg_molecular_generation/train_tokenizer.py --input_csv data/guacamol_completo_1.6M.csv"
