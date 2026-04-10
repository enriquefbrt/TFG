import os
import argparse
import torch
import pandas as pd
from transformers import (
    T5ForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

from tfg_molecular_generation.ape_hf_wrapper import APEHuggingFaceTokenizer

MODEL_NAME = "google/t5-v1_1-base"

def _ensure_list(values):
    """Normalizes Dataset transform inputs to list format."""
    if isinstance(values, list):
        return values
    return [values]

def load_and_tokenize_data(csv_path: str, tokenizer, max_input_length=128, max_target_length=128):
    """
    Loads pre-training data (scaffold -> original_smiles) and sets up dynamic tokenization.
    """
    from tfg_molecular_generation.data_prep import generate_random_smiles

    df = pd.read_csv(csv_path)
    required_columns = {"input_text", "target_text"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {csv_path}: {sorted(missing_columns)}. "
            "Expected columns: ['input_text', 'target_text']"
        )

    # Ensure there are no NaNs/empty strings in the required columns
    rows_before_clean = len(df)
    df = df.dropna(subset=["input_text", "target_text"]).copy()
    df["input_text"] = df["input_text"].astype(str).str.strip()
    df["target_text"] = df["target_text"].astype(str).str.strip()
    df = df[(df["input_text"] != "") & (df["target_text"] != "")]

    rows_dropped = rows_before_clean - len(df)
    if rows_dropped > 0:
        print(f"[Data Quality] Dropped {rows_dropped} rows with empty/invalid input_text or target_text.")
    if df.empty:
        raise ValueError(f"No valid rows left in {csv_path} after cleaning.")
    
    # Convert pandas dataframe to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    
    # We use a transform function which is applied ON-THE-FLY dynamically during dataloading per epoch
    def preprocess_transform(examples):
        input_column = _ensure_list(examples["input_text"])
        target_column = _ensure_list(examples["target_text"])

        # inputs: Scaffold (static)
        inputs = [str(ex) for ex in input_column]
        
        # targets: Original Molecule (DYNAMICALY AUGMENTED)
        original_targets = [str(ex) for ex in target_column]
        targets = []
        for t in original_targets:
            # On-the-fly SMILES randomization 
            # We ask for 1 random variant per molecule just-in-time
            r_smiles = generate_random_smiles(t, num_random=1)
            # If RDKit somehow fails to randomize, fallback to the original SMILES
            targets.append(r_smiles[0] if r_smiles else t)
        
        # Tokenize Inputs (Encoder)
        model_inputs = tokenizer(
            inputs, 
            max_length=max_input_length, 
            padding="max_length", 
            truncation=True
        )
        
        # Tokenize Targets (Decoder)
        labels = tokenizer(
            text_target=targets, 
            max_length=max_target_length, 
            padding="max_length", 
            truncation=True
        )

        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer pad_token_id is None. Please define a valid pad token.")
        safe_label_fallback_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.unk_token_id
        if safe_label_fallback_id is None:
            raise ValueError("Tokenizer has neither eos_token_id nor unk_token_id defined.")

        # If we are padding, replace pad token id's of the labels by -100 so it's ignored by the loss.
        # Guard against all--100 rows, which would produce NaN loss (empty CE denominator).
        cleaned_labels = []
        for label in labels["input_ids"]:
            masked_label = [(l if l != tokenizer.pad_token_id else -100) for l in label]
            if all(l == -100 for l in masked_label):
                masked_label[0] = safe_label_fallback_id
            cleaned_labels.append(masked_label)
        labels["input_ids"] = cleaned_labels
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # `set_transform` replaces static `map` to perform preprocessing in RAM iteratively
    dataset.set_transform(preprocess_transform)
    return dataset

def resolve_precision_mode(requested_precision: str):
    """
    Returns (bf16, fp16, resolved_name) based on requested precision and hardware support.
    """
    requested = requested_precision.lower()
    cuda_available = torch.cuda.is_available()
    bf16_supported = (
        cuda_available
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )

    if requested == "auto":
        if bf16_supported:
            return True, False, "bf16"
        if cuda_available:
            return False, True, "fp16"
        return False, False, "fp32"

    if requested == "bf16":
        if not bf16_supported:
            raise ValueError("bf16 was requested, but this machine/GPU does not support bf16.")
        return True, False, "bf16"

    if requested == "fp16":
        if not cuda_available:
            raise ValueError("fp16 was requested, but CUDA is not available.")
        return False, True, "fp16"

    if requested == "fp32":
        return False, False, "fp32"

    raise ValueError(f"Unsupported precision mode: {requested_precision}")

def main():
    parser = argparse.ArgumentParser(description="TFG Molecular Generation Pre-training")
    parser.add_argument("--train_data", type=str, default="data/pretrain_t5_train.csv", help="Path to training CSV")
    parser.add_argument("--val_data", type=str, default="data/pretrain_t5_val.csv", help="Path to validation CSV")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Directory of the trained APETokenizer")
    parser.add_argument("--output_dir", type=str, default="./models/t5_pretrain_scaffolds", help="Directory where to save the model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Training precision mode. 'auto' prefers bf16 on supported GPUs.",
    )
    
    args = parser.parse_args()
    use_bf16, use_fp16, resolved_precision = resolve_precision_mode(args.precision)
    print(f"Using precision mode: {resolved_precision}")

    # 1. Initialize Custom Tokenizer and Model
    print("Loading Custom APETokenizer and Model...")
    
    # Check if we should use the custom tokenizer
    if not os.path.isdir(args.tokenizer_dir):
        raise ValueError(f"Custom tokenizer directory '{args.tokenizer_dir}' not found. Please train APETokenizer first.")
        
    tokenizer = APEHuggingFaceTokenizer(ape_tokenizer_path=args.tokenizer_dir)
    
    # We load the t5-small architecture base (parameters optimized for NLP, ready to unlearn and learn chemistry)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Ensure the model's vocabulary size matches the tokenizer 
    model.resize_token_embeddings(len(tokenizer))
    
    # 2. Setup Data
    if not os.path.exists(args.train_data):
        print(f"Warning: {args.train_data} not found. Running a mock training check...")
        # Create a tiny mock dataset for validation to ensure compilation works
        df_mock = pd.DataFrame({
            "input_text": ["O=C(NCc1ccccc1)c1cccnc1", "c1ccccc1"],
            "target_text": ["CC1=CN=C(C(=C1O)C(=O)NCC2=CC=CC=C2)C3=CC=C(C=C3)Cl", "Cc1ccccc1C(C)N"]
        })
        os.makedirs(os.path.dirname(args.train_data) or ".", exist_ok=True)
        df_mock.iloc[:1].to_csv(args.train_data, index=False)
        df_mock.iloc[1:].to_csv(args.val_data, index=False)
        
    print("Tokenizing datasets...")
    train_dataset = load_and_tokenize_data(args.train_data, tokenizer)
    eval_dataset = load_and_tokenize_data(args.val_data, tokenizer)
    
    # 3. Training Arguments (Optimized for NVIDIA T4 - 16GB VRAM)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=4,        
        push_to_hub=False,
        logging_steps=100,
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        optim="adafactor",
        logging_nan_inf_filter=False,
        remove_unused_columns=False,
    )
    
    # Data collator manages the dynamic padding of the batches
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # 4. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    # 5. Handle Spot Instance Preemptions (Resume from Checkpoint)
    from transformers.trainer_utils import get_last_checkpoint
    
    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        try:
            last_checkpoint = get_last_checkpoint(args.output_dir)
        except Exception:
            pass
            
        if last_checkpoint is not None:
            print(f"Spot Preemption Alert! Resuming training from checkpoint: {last_checkpoint}")
        else:
            print("Starting Training from scratch...")

    print("Starting Training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    print("Training finished!")
    
    # 6. Save the final model and tokenizer state
    final_output_path = f"{args.output_dir}_FINAL"
    print(f"Saving final model and tokenizer to {final_output_path}")
    trainer.save_model(final_output_path)
    tokenizer.save_pretrained(final_output_path)
    print("All done!")

if __name__ == "__main__":
    main()
