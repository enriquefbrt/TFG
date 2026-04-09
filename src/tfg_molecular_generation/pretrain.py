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

MODEL_NAME = "google/t5-v1_1-small"

def load_and_tokenize_data(csv_path: str, tokenizer, max_input_length=128, max_target_length=128):
    """
    Loads pre-training data (scaffold -> original_smiles) and sets up dynamic tokenization.
    """
    from tfg_molecular_generation.data_prep import generate_random_smiles

    df = pd.read_csv(csv_path)
    # Ensure there are no NaNs in the required columns
    df = df.dropna(subset=["input_text", "target_text"])
    
    # Convert pandas dataframe to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    
    # We use a transform function which is applied ON-THE-FLY dynamically during dataloading per epoch
    def preprocess_transform(examples):
        # inputs: Scaffold (static)
        inputs = [str(ex) for ex in examples["input_text"]]
        
        # targets: Original Molecule (DYNAMICALY AUGMENTED)
        original_targets = [str(ex) for ex in examples["target_text"]]
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
            
        # If we are padding, replace pad token id's of the labels by -100 so it's ignored by the loss
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # `set_transform` replaces static `map` to perform preprocessing in RAM iteratively
    dataset.set_transform(preprocess_transform)
    return dataset

def main():
    parser = argparse.ArgumentParser(description="TFG Molecular Generation Pre-training")
    parser.add_argument("--train_data", type=str, default="data/pretrain_t5_train.csv", help="Path to training CSV")
    parser.add_argument("--val_data", type=str, default="data/pretrain_t5_val.csv", help="Path to validation CSV")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Directory of the trained APETokenizer")
    parser.add_argument("--output_dir", type=str, default="./models/t5_pretrain_scaffolds", help="Directory where to save the model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()

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
        predict_with_generate=True,      # Needed to compute generation metrics later
        fp16=True,                       # Critical for NVIDIA T4: Mixed Precision halves VRAM and doubles speed
        dataloader_num_workers=4,        
        push_to_hub=False,
        logging_steps=100,
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
