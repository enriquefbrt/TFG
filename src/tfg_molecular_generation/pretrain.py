import os
import argparse
import torch
import pandas as pd
from transformers import (
    T5ForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import Dataset

from tfg_molecular_generation.ape_hf_wrapper import APEHuggingFaceTokenizer
from tfg_molecular_generation.decorator_utils import (
    attach_decorators_to_scaffold,
    is_decorator_sequence,
    smiles_to_scaffold_and_decorators,
)
from tfg_molecular_generation.data_prep import generate_random_smiles

MODEL_NAME = "google/t5-v1_1-base"

def _ensure_list(values):
    """Normalizes Dataset transform inputs to list format."""
    if isinstance(values, list):
        return values
    return [values]


def _build_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes supported CSV formats into rows with:
    - source_smiles (may be empty for static-only rows)
    - input_text    (fallback scaffold with [*:i])
    - target_text   (fallback decorator sequence)

    Supported inputs:
    - New format: source_smiles + input_text + target_text
    - Decorator static format: input_text + target_text (target already decorated)
    - Legacy format: input_text + target_text where target_text is full SMILES
    """
    normalized_rows = []
    skipped = 0

    for _, row in df.iterrows():
        source_smiles = ""
        input_text = ""
        target_text = ""

        if "source_smiles" in df.columns:
            val = row.get("source_smiles")
            source_smiles = "" if pd.isna(val) else str(val).strip()
        if "input_text" in df.columns:
            val = row.get("input_text")
            input_text = "" if pd.isna(val) else str(val).strip()
        if "target_text" in df.columns:
            val = row.get("target_text")
            target_text = "" if pd.isna(val) else str(val).strip()

        fallback_pair = None

        # If static decorator pair is provided, keep it as fallback.
        if input_text and target_text and is_decorator_sequence(target_text):
            fallback_pair = (input_text, target_text)

        # Legacy path: target_text may be a full molecule SMILES.
        if not source_smiles and target_text and not is_decorator_sequence(target_text):
            source_smiles = target_text

        # If fallback pair is still missing, derive it from source_smiles.
        if fallback_pair is None and source_smiles:
            fallback_pair = smiles_to_scaffold_and_decorators(source_smiles)

        if fallback_pair is None:
            skipped += 1
            continue

        fallback_input, fallback_target = fallback_pair
        if not source_smiles:
            try:
                source_smiles = attach_decorators_to_scaffold(fallback_input, fallback_target)
            except Exception:
                source_smiles = ""
        normalized_rows.append(
            {
                "source_smiles": source_smiles,
                "input_text": fallback_input,
                "target_text": fallback_target,
            }
        )

    if skipped:
        print(
            f"[Data Format] Skipped {skipped} rows that could not be converted to scaffold+decorators."
        )
    return pd.DataFrame(normalized_rows)

def load_and_tokenize_data(csv_path: str, tokenizer, max_input_length=128, max_target_length=128):
    """
    Loads data and sets up on-the-fly dynamic preprocessing.
    If source_smiles is present, each batch sample is randomized and then decomposed
    to scaffold+decorators at runtime (epoch-dependent augmentation).
    """
    df = pd.read_csv(csv_path)
    supported = {"source_smiles", "input_text", "target_text"} & set(df.columns)
    if not supported:
        raise ValueError(
            f"{csv_path} has no supported columns. Expected at least one of "
            "['source_smiles', 'input_text', 'target_text']."
        )

    rows_before_clean = len(df)
    df = _build_training_rows(df)
    rows_after_clean = len(df)
    if rows_after_clean == 0:
        raise ValueError(f"No valid rows left in {csv_path} after conversion/cleaning.")
    if rows_after_clean < rows_before_clean:
        print(
            f"[Data Quality] Dropped {rows_before_clean - rows_after_clean} invalid rows during normalization."
        )

    dynamic_rows = int((df["source_smiles"].astype(str).str.strip() != "").sum())
    print(
        f"[Data Format] Loaded {rows_after_clean} rows from {csv_path}. "
        f"Dynamic source_smiles rows: {dynamic_rows}."
    )
    
    # Convert pandas dataframe to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    
    # We use a transform function applied on-the-fly during dataloading.
    def preprocess_transform(examples):
        input_column = [str(x) for x in _ensure_list(examples["input_text"])]
        target_column = [str(x) for x in _ensure_list(examples["target_text"])]
        source_column = (
            [str(x) for x in _ensure_list(examples["source_smiles"])]
            if "source_smiles" in examples
            else [""] * len(input_column)
        )

        # If source_smiles is available, randomize molecule and recompute scaffold+decorators
        # online so each epoch sees fresh randomized writing and its corresponding scaffold.
        inputs = []
        targets = []
        for fallback_input, fallback_target, source_smiles in zip(
            input_column, target_column, source_column
        ):
            source_smiles = source_smiles.strip()
            if source_smiles:
                randomized = generate_random_smiles(source_smiles, num_random=1)
                sampled_smiles = randomized[0] if randomized else source_smiles
                pair = smiles_to_scaffold_and_decorators(sampled_smiles)
                if pair is None and sampled_smiles != source_smiles:
                    pair = smiles_to_scaffold_and_decorators(source_smiles)
                if pair is not None:
                    in_text, out_text = pair
                    inputs.append(in_text)
                    targets.append(out_text)
                    continue

            # Static fallback if dynamic conversion fails.
            inputs.append(fallback_input)
            targets.append(fallback_target)
        
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
        "--logging_steps",
        type=int,
        default=20,
        help="Log training metrics every N optimizer steps (lower gives denser curves).",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=1,
        help="Early stopping patience in evaluation rounds (epoch-level with current config). Set <=0 to disable.",
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.0,
        help="Minimum eval_loss improvement to reset early stopping patience.",
    )
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

    # Align generation-related IDs with tokenizer, so EOS is consistently learned/used.
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.bos_token_id
        )
    
    # 2. Setup Data
    if not os.path.exists(args.train_data):
        print(f"Warning: {args.train_data} not found. Running a mock training check...")
        # Create a tiny mock dataset for validation to ensure compilation works
        df_mock = pd.DataFrame(
            {
            "source_smiles": [
                "NCCc1ccccc1",
                "NCCc1ccc(F)cc1",
            ],
            "input_text": [
                "c1ccc([*:1])cc1",
                "c1cc([*:1])ccc1[*:2]",
            ],
            "target_text": [
                "<R1> [*:1]CCN </R1>",
                "<R1> [*:1]F </R1> <R2> [*:2]CCN </R2>",
            ],
            }
        )
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
        logging_strategy="steps",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        # We optimize pretraining speed by evaluating with loss only.
        predict_with_generate=False,
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=4,        
        push_to_hub=False,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        optim="adafactor",
        logging_nan_inf_filter=False,
        remove_unused_columns=False,
        report_to="none",
        load_best_model_at_end=args.early_stopping_patience > 0,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Data collator manages the dynamic padding of the batches
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    # 4. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
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
