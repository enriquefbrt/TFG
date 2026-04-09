import os
import argparse
import pandas as pd
import re
import json
import time

# Prevenir advertencias de paralelismo en tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# Regex pura extraída del APE original
SMILES_REGEX_PATTERN = r"\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"

def main():
    parser = argparse.ArgumentParser(description="Tren Tokenizador APE acelerado por Unicode & Rust")
    parser.add_argument("--input_csv", type=str, required=True, help="Path al CSV de las moléculas.")
    parser.add_argument("--smiles_col", type=str, default="smiles", help="Columna SMILES.")
    parser.add_argument("--output_dir", type=str, default="./models/ape_tokenizer", help="Directorio.)")
    parser.add_argument("--max_vocab_size", type=int, default=5000, help="Tamaño max vocab")
    parser.add_argument("--min_freq_for_merge", type=int, default=1500, help="Frecuencia mínima para unir")
    
    args = parser.parse_args()

    # 1. Cargar Data
    print(f"Cargando dataset {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    smiles_list = df[args.smiles_col].dropna().astype(str).tolist()
    print(f"[{len(smiles_list)} Moléculas Cargadas]")

    # 2. Extraer Átomos del Espacio Químico
    print(">> Analizando moléculas y detectando átomos APE únicos...")
    pattern = re.compile(SMILES_REGEX_PATTERN)
    unique_tokens = set()
    for sm in smiles_list:
        unique_tokens.update(pattern.findall(sm))

    multi_char_tokens = {t for t in unique_tokens if len(t) > 1}
    single_char_tokens = {t for t in unique_tokens if len(t) == 1}

    print(f">> Átomos complejos descubiertos: {len(multi_char_tokens)}")

    # 3. Construir Biyección PUA (Private Use Area Unicode)
    token_to_unicode = {}
    unicode_to_token = {}
    START_CODE = 0xE000 
    for i, token in enumerate(multi_char_tokens):
        char = chr(START_CODE + i)
        token_to_unicode[token] = char
        unicode_to_token[char] = token

    def translate_to_unicode(sm):
        return "".join(token_to_unicode.get(p, p) for p in pattern.findall(sm))

    # 4. Traducir dataset
    print(f">> Traduciendo las {len(smiles_list)} moléculas al Espacio Unicode 1D...")
    t0 = time.time()
    mapped_smiles = [translate_to_unicode(sm) for sm in smiles_list]
    print(f"Traducción lista en {time.time()-t0:.2f}s")

    # 5. Entrenar Tokenizador en BPE Nativo Rust
    initial_alphabet = list(single_char_tokens) + list(unicode_to_token.keys())
    
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(
        vocab_size=args.max_vocab_size,
        min_frequency=args.min_freq_for_merge,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
        initial_alphabet=initial_alphabet,
        show_progress=True
    )

    print(f">> 💪 Pasando dataset a RUST T5 Tokenizer (freq>={args.min_freq_for_merge})...")
    t1 = time.time()
    tokenizer.train_from_iterator(mapped_smiles, trainer=trainer)
    print(f">> 🏎️ Entrenamiento BPE Rust finalizado en {time.time()-t1:.2f}s!")

    # 6. Guardar Todo (Modelo y Mapeos)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Modelo HuggingFace
    hf_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(hf_path)
    
    # Diccionarios de Traducción
    mapping_path = os.path.join(args.output_dir, "unicode_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump({
            "token_to_unicode": token_to_unicode,
            "unicode_to_token": unicode_to_token
        }, f, indent=2)

    print(f"✅ ¡Todo guardado con éxito en {args.output_dir}!")
    
    # Decodificar log para humano
    hf_vocab = tokenizer.get_vocab()
    print(f"\n📊 RESULTADOS FINALES:")
    print(f"Tokens Generados Reales: {len(hf_vocab)}")

if __name__ == "__main__":
    main()
