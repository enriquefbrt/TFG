import os
import argparse
import pandas as pd
import re
import json
import time

# Prevent tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# Pure Regex extracted from the original APE
SMILES_REGEX_PATTERN = r"\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"

def main():
    parser = argparse.ArgumentParser(description="Accelerated APE Tokenizer Training via Unicode & Rust")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the molecules CSV.")
    parser.add_argument("--smiles_col", type=str, default="smiles", help="SMILES column name.")
    parser.add_argument("--output_dir", type=str, default="./models/ape_tokenizer", help="Output directory.")
    parser.add_argument("--max_vocab_size", type=int, default=5000, help="Max vocab size")
    parser.add_argument("--min_freq_for_merge", type=int, default=1500, help="Minimum frequency for merge")
    
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading dataset {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    smiles_list = df[args.smiles_col].dropna().astype(str).tolist()
    print(f"[{len(smiles_list)} Molecules Loaded]")

    # 2. Extract Chemical Space Atoms
    print(">> Analyzing molecules and detecting unique APE atoms...")
    pattern = re.compile(SMILES_REGEX_PATTERN)
    unique_tokens = set()
    for sm in smiles_list:
        unique_tokens.update(pattern.findall(sm))

    multi_char_tokens = {t for t in unique_tokens if len(t) > 1}
    single_char_tokens = {t for t in unique_tokens if len(t) == 1}

    print(f">> Complex atoms discovered: {len(multi_char_tokens)}")

    # 3. Build PUA Bijection (Private Use Area Unicode)
    token_to_unicode = {}
    unicode_to_token = {}
    START_CODE = 0xE000 
    for i, token in enumerate(multi_char_tokens):
        char = chr(START_CODE + i)
        token_to_unicode[token] = char
        unicode_to_token[char] = token

    def translate_to_unicode(sm):
        return "".join(token_to_unicode.get(p, p) for p in pattern.findall(sm))

    # 4. Translate dataset
    print(f">> Translating {len(smiles_list)} molecules to 1D Unicode Space...")
    t0 = time.time()
    mapped_smiles = [translate_to_unicode(sm) for sm in smiles_list]
    print(f"Translation completed in {time.time()-t0:.2f}s")

    # 5. Train Tokenizer on Native Rust BPE
    initial_alphabet = list(single_char_tokens) + list(unicode_to_token.keys())
    
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(
        vocab_size=args.max_vocab_size,
        min_frequency=args.min_freq_for_merge,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
        initial_alphabet=initial_alphabet,
        show_progress=True
    )

    print(f">> Passing dataset to RUST T5 Tokenizer (freq>={args.min_freq_for_merge})...")
    t1 = time.time()
    tokenizer.train_from_iterator(mapped_smiles, trainer=trainer)
    print(f">> BPE Rust training finished in {time.time()-t1:.2f}s!")

    # 6. Save Everything (Model and Mappings)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # HuggingFace Model
    hf_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(hf_path)
    
    # Translation Dictionaries
    mapping_path = os.path.join(args.output_dir, "unicode_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump({
            "token_to_unicode": token_to_unicode,
            "unicode_to_token": unicode_to_token
        }, f, indent=2)

    print(f"Successfully saved everything to {args.output_dir}!")
    
    # Decode log for human reading
    hf_vocab = tokenizer.get_vocab()
    print(f"\nFINAL RESULTS:")
    print(f"Generated True Tokens: {len(hf_vocab)}")

if __name__ == "__main__":
    main()
