import os
import argparse
import pandas as pd
from tfg_molecular_generation.ape_tokenizer import APETokenizer

def main():
    parser = argparse.ArgumentParser(description="Train custom APETokenizer on a SMILES dataset.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the CSV file containing SMILES.")
    parser.add_argument("--smiles_col", type=str, default="smiles", help="Name of the column containing SMILES strings.")
    parser.add_argument("--output_dir", type=str, default="./models/ape_tokenizer", help="Directory to save the trained tokenizer.")
    parser.add_argument("--max_vocab_size", type=int, default=5000, help="Maximum vocabulary size.")
    parser.add_argument("--min_freq_for_merge", type=int, default=2000, help="Minimum frequency required to merge a pair.")
    parser.add_argument("--save_checkpoint", action="store_true", help="Whether to save checkpoint models during training.")
    
    args = parser.parse_args()

    # 1. Load data
    print(f"Loading dataset from {args.input_csv}...")
    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    if args.smiles_col not in df.columns:
        raise ValueError(f"Column '{args.smiles_col}' not found in the CSV. Available columns: {list(df.columns)}")

    # Extract SMILES as a list of strings
    smiles_list = df[args.smiles_col].dropna().astype(str).tolist()
    print(f"Loaded {len(smiles_list)} valid SMILES strings.")

    # 2. Initialize Tokenizer
    print("Initializing original APETokenizer (Python)...")
    tokenizer = APETokenizer()

    # 3. Train Tokenizer
    print(f"Starting APE training with max_vocab_size={args.max_vocab_size} and min_freq_for_merge={args.min_freq_for_merge}...")
    print("WARNING: This pure-Python script may take hours for millions of molecules. Consider taking a coffee break.")
    
    # We pass the loaded SMILES list as the corpus
    tokenizer.train(
        corpus=smiles_list,
        type="smiles", 
        max_vocab_size=args.max_vocab_size,
        min_freq_for_merge=args.min_freq_for_merge,
        save_checkpoint=args.save_checkpoint,
        checkpoint_path=os.path.join(args.output_dir, "checkpoints")
    )

    # 4. Save Final Tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Tokenizer vocabulary and training state saved successfully to {args.output_dir}")

if __name__ == "__main__":
    main()
