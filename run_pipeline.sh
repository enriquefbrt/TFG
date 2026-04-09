#!/bin/bash
set -e # Detiene el script si cualquier comando falla

# 1. Configurar Entorno
echo "=========================================="
echo "🚀 INICIANDO PIPELINE DE PRE-ENTRENAMIENTO"
echo "=========================================="
echo ">>> Activando entorno virtual..."
source .venv/bin/activate
export PYTHONPATH=src

# 2. Generar Scaffolds (Solo si no existen)
if [ ! -f "data/pretrain_t5.csv" ]; then
    echo ">>> Generando Scaffolds de Murcko (esto puede tardar unos 15 min)..."
    nohup python -c "from tfg_molecular_generation.data_prep import prepare_pretraining_dataset; prepare_pretraining_dataset('data/guacamol_completo_1.6M.csv', 'data/pretrain_t5.csv')" > nohup_scaffolds.out 2>&1 &
    wait $!
else
    echo ">>> El archivo de Scaffolds maestro ya existe. Saltando cálculo..."
fi

# 3. Particionar Dataset (Train, Val, Test)
if [ ! -f "data/pretrain_t5_train.csv" ]; then
    echo ">>> Particionando dataset en Train, Val (10K) y Test (10K)..."
    python -c "
import pandas as pd
print('Leyendo CSV maestro...')
df = pd.read_csv('data/pretrain_t5.csv')

print('Extrayendo partición TEST (10.000)...')
test_df = df.sample(n=10000, random_state=42)
df = df.drop(test_df.index)

print('Extrayendo partición VAL (10.000)...')
val_df = df.sample(n=10000, random_state=42)
train_df = df.drop(val_df.index)

print('Guardando archivos en Disco...')
test_df.to_csv('data/pretrain_t5_test.csv', index=False)
val_df.to_csv('data/pretrain_t5_val.csv', index=False)
train_df.to_csv('data/pretrain_t5_train.csv', index=False)
print('✅ Particiones guardadas con éxito!')
" > nohup_split.out 2>&1 &
    wait $!
else
    echo ">>> Las particiones de Train/Val/Test ya existen. Saltando división..."
fi

# 4. Entrenamiento del Transformador
echo "=========================================="
echo "🧠 ENTRENANDO RED NEURONAL T5             "
echo "=========================================="

# Añadimos nohup al entrenamiento final y lo dejamos en 2º plano libre
echo ">>> (Puedes ver el progreso del T5 con: tail -f nohup_pretrain.out)"
nohup python -u src/tfg_molecular_generation/pretrain.py \
    --train_data data/pretrain_t5_train.csv \
    --val_data data/pretrain_t5_val.csv \
    --tokenizer_dir ./models/ape_tokenizer \
    --output_dir ./models/t5_base_pretrain \
    --epochs 10 \
    --batch_size 128 > nohup_pretrain.out 2>&1 &

echo "=========================================="
echo "🎉 PIPELINE FINALIZADO CON ÉXITO"
echo "=========================================="
