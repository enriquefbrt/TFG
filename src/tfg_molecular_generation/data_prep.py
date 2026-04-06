import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import random

def extract_scaffold(smiles: str) -> str:
    """
    Extracts the Bemis-Murcko scaffold from a given SMILES string.
    Returns an empty string if it fails or if the molecule has no scaffold.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        # Get the Bemis-Murcko scaffold (the core rings and connecting linkers)
        scaffold_smi = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles)
        return scaffold_smi
    except Exception:
        return ""

def generate_random_smiles(smiles: str, num_random: int = 10) -> list[str]:
    """
    Generates multiple random (non-canonical) SMILES strings representing 
    the same molecule, useful for data augmentation in Low-Data Regimes.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
        
    random_smiles = set()
    max_attempts = num_random * 5
    attempts = 0
    
    # Try to generate unique random smiles
    while len(random_smiles) < num_random and attempts < max_attempts:
        rs = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        random_smiles.add(rs)
        attempts += 1
        
    return list(random_smiles)

def prepare_pretraining_dataset(input_csv: str, output_csv: str, smiles_col: str = "smiles") -> None:
    """
    Prepares a dataset for the T5 Pre-training phase.
    Input format: T5 Encoder: [Scaffold SMILES] -> T5 Decoder: [Original SMILES]
    """
    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in the CSV.")
        
    print("Extracting scaffolds... (This may take a while depending on dataset size)")
    df["scaffold"] = df[smiles_col].apply(extract_scaffold)
    
    # Drop rows where scaffold couldn't be extracted or is empty (acyclic molecules)
    df_clean = df[df["scaffold"] != ""].copy()
    
    # For T5 text-to-text, we rename the columns clearly
    df_clean = df_clean[["scaffold", smiles_col]].rename(
        columns={"scaffold": "input_text", smiles_col: "target_text"}
    )
    
    print(f"Retained {len(df_clean)} molecules with valid scaffolds out of {len(df)}.")
    df_clean.to_csv(output_csv, index=False)
    print(f"Pre-training dataset saved to {output_csv}")

def prepare_finetuning_dataset(input_csv: str, output_csv: str, smiles_col: str = "smiles", augment_factor: int = 10) -> None:
    """
    Prepares a dataset for the T5 Fine-tuning (Transfer Learning) phase.
    Applies data augmentation through SMILES randomization for the target molecules (e.g., DYRK1A inhibitors).
    Input format: T5 Encoder: [Scaffold SMILES] -> T5 Decoder: [Randomized Target SMILES]
    """
    print(f"Loading finetuning dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    augmented_data = []
    
    print(f"Applying SMILES randomization (Factor: {augment_factor}x)...")
    for _, row in df.iterrows():
        original_smiles = row[smiles_col]
        
        # 1. Extract scaffold for the input
        scaffold = extract_scaffold(original_smiles)
        if not scaffold:
            continue
            
        # 2. Get multiple random representations of the active molecule
        random_smiles_list = generate_random_smiles(original_smiles, num_random=augment_factor)
        
        # 3. Create a pair mapping Scaffold -> Random SMILES
        for rs in random_smiles_list:
            augmented_data.append({
                "input_text": scaffold,
                "target_text": rs
            })
            
    df_augmented = pd.DataFrame(augmented_data)
    print(f"Augmented dataset size: {len(df_augmented)} molecules from original {len(df)}.")
    df_augmented.to_csv(output_csv, index=False)
    print(f"Fine-tuning dataset saved to {output_csv}")

if __name__ == "__main__":
    # Example usage (uncomment and modify paths when datasets are available):
    
    # 1. Prepare Pre-training data
    # prepare_pretraining_dataset("data/guacamol_train.csv", "data/pretrain_t5.csv", "smiles")
    
    # 2. Prepare Fine-tuning data (DYRK1A targets)
    prepare_finetuning_dataset("data/finetuning/datos_crudos.csv", "data/finetuning/postprocessed_finetunning.csv", "Smiles", augment_factor=10)
