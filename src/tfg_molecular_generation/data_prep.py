import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from tfg_molecular_generation.decorator_utils import smiles_to_scaffold_and_decorators

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


def build_decorator_pair_from_smiles(smiles: str):
    """
    Returns (input_text, target_text) in decorator format:
    input_text  = scaffold with labeled attachment points [*:i]
    target_text = "<R1> [*:1]... </R1> <R2> [*:2]... </R2> ..."
    """
    return smiles_to_scaffold_and_decorators(smiles)

def prepare_pretraining_dataset(input_csv: str, output_csv: str, smiles_col: str = "smiles") -> None:
    """
    Prepares a dataset for the T5 Pre-training phase.
    Input format:
    T5 Encoder: [Scaffold with [*:i]]
    T5 Decoder: [Decorator sequence with <Ri> [*:i]... </Ri>]
    """
    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in the CSV.")
        
    print("Converting molecules to scaffold+decorator pairs...")
    rows = []
    skipped = 0
    for smiles in df[smiles_col].dropna().astype(str):
        pair = build_decorator_pair_from_smiles(smiles)
        if pair is None:
            skipped += 1
            continue
        scaffold, decorators = pair
        rows.append(
            {
                "source_smiles": smiles,
                "input_text": scaffold,
                "target_text": decorators,
            }
        )

    df_clean = pd.DataFrame(rows)
    print(
        f"Retained {len(df_clean)} decorator pairs out of {len(df)} molecules. "
        f"Skipped {skipped} molecules that could not be decomposed."
    )
    df_clean.to_csv(output_csv, index=False)
    print(f"Pre-training dataset saved to {output_csv}")

def prepare_finetuning_dataset(input_csv: str, output_csv: str, smiles_col: str = "smiles", augment_factor: int = 10) -> None:
    """
    Prepares a dataset for the T5 Fine-tuning (Transfer Learning) phase.
    Stores source molecules and fallback decorator pairs. During training, if source_smiles
    is present, randomization and scaffold/decorator recomputation can be done on-the-fly each epoch.
    augment_factor controls row repetition (sampling weight), not static randomization.
    Input format:
    T5 Encoder: [Scaffold with [*:i]]
    T5 Decoder: [Decorator sequence with <Ri> [*:i]... </Ri>]
    """
    print(f"Loading finetuning dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    augmented_data = []
    
    repeat_factor = max(int(augment_factor), 1)
    print(
        f"Converting to scaffold+decorators with dynamic augmentation-ready rows "
        f"(repeat factor: {repeat_factor}x)..."
    )
    skipped = 0
    for _, row in df.iterrows():
        original_smiles = str(row[smiles_col]).strip()
        if not original_smiles:
            skipped += 1
            continue

        pair = build_decorator_pair_from_smiles(original_smiles)
        if pair is None:
            skipped += 1
            continue
        scaffold, decorators = pair

        for _ in range(repeat_factor):
            augmented_data.append(
                {
                    "source_smiles": original_smiles,
                    "input_text": scaffold,
                    "target_text": decorators,
                }
            )
            
    df_augmented = pd.DataFrame(augmented_data)
    print(
        f"Augmented dataset size: {len(df_augmented)} pairs from original {len(df)} molecules. "
        f"Skipped {skipped} failed decompositions."
    )
    df_augmented.to_csv(output_csv, index=False)
    print(f"Fine-tuning dataset saved to {output_csv}")

if __name__ == "__main__":
    # Example usage (uncomment and modify paths when datasets are available):
    
    # 1. Prepare Pre-training data
    # prepare_pretraining_dataset("data/guacamol_train.csv", "data/pretrain_t5.csv", "smiles")
    
    # 2. Prepare Fine-tuning data (DYRK1A targets)
    prepare_finetuning_dataset("data/finetuning/datos_crudos.csv", "data/finetuning/postprocessed_finetunning.csv", "Smiles", augment_factor=10)
