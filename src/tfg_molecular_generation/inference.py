import argparse
import csv
import os
import random

import torch
from transformers import T5ForConditionalGeneration

from tfg_molecular_generation.ape_hf_wrapper import APEHuggingFaceTokenizer
from tfg_molecular_generation.inference_utils import (
    build_first_token_distribution,
    load_distribution_cache,
    load_scaffolds,
    mix_distributions,
    resolve_decoder_start_id,
    sample_token_id,
    save_distribution_cache,
)


def main():
    parser = argparse.ArgumentParser(
        description="Scaffold-conditioned inference with weighted first-token sampling."
    )
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model folder.")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Path to APE tokenizer folder.")
    parser.add_argument("--scaffold", type=str, default=None, help="Single scaffold SMILES string.")
    parser.add_argument("--scaffold_file", type=str, default=None, help="TXT/CSV file containing scaffolds.")
    parser.add_argument("--scaffold_col", type=str, default=None, help="Scaffold column if scaffold_file is CSV.")
    parser.add_argument("--output_csv", type=str, default="generated_scaffold_samples.csv", help="Output CSV path.")
    parser.add_argument("--num_samples_per_scaffold", type=int, default=20, help="Samples generated per scaffold.")
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--finetune_csv", type=str, required=True, help="Finetuning CSV path.")
    parser.add_argument("--finetune_smiles_col", type=str, default=None, help="SMILES column in finetune CSV.")
    parser.add_argument("--affinity_col", type=str, default=None, help="Affinity column in finetune CSV.")
    parser.add_argument(
        "--affinity_mode",
        type=str,
        default="auto",
        choices=["auto", "higher_better", "lower_better"],
        help="Whether larger affinity values are better.",
    )
    parser.add_argument("--tau", type=float, default=None, help="Sigmoid temperature. Default: IQR(scores).")

    parser.add_argument("--pretrain_csv", type=str, required=True, help="Pretraining CSV path.")
    parser.add_argument("--pretrain_smiles_col", type=str, default=None, help="SMILES column in pretrain CSV.")

    parser.add_argument("--mix_w_active", type=float, default=0.6)
    parser.add_argument("--mix_w_finetune", type=float, default=0.3)
    parser.add_argument("--mix_w_pretrain", type=float, default=0.1)
    parser.add_argument(
        "--distribution_cache_json",
        type=str,
        default=None,
        help="Optional JSON cache to save/load first-token distributions.",
    )
    parser.add_argument("--max_rows_finetune", type=int, default=None)
    parser.add_argument("--max_rows_pretrain", type=int, default=None)

    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        raise ValueError(f"model_dir not found: {args.model_dir}")
    if not os.path.isdir(args.tokenizer_dir):
        raise ValueError(f"tokenizer_dir not found: {args.tokenizer_dir}")

    print("Loading tokenizer and model...")
    tokenizer = APEHuggingFaceTokenizer(ape_tokenizer_path=args.tokenizer_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Using device: {device}")

    if args.distribution_cache_json and os.path.exists(args.distribution_cache_json):
        print(f"Loading cached distributions from {args.distribution_cache_json}")
        active_dist, finetune_dist, pretrain_dist = load_distribution_cache(args.distribution_cache_json)
    else:
        print("Building affinity-weighted finetune first-token distribution...")
        active_dist, active_meta = build_first_token_distribution(
            csv_path=args.finetune_csv,
            tokenizer=tokenizer,
            smiles_col=args.finetune_smiles_col,
            affinity_col=args.affinity_col,
            affinity_mode=args.affinity_mode,
            tau=args.tau,
            weighted_by_affinity=True,
            max_rows=args.max_rows_finetune,
        )
        print(f"Active distribution built: {active_meta}")

        print("Building full finetune first-token distribution...")
        finetune_dist, finetune_meta = build_first_token_distribution(
            csv_path=args.finetune_csv,
            tokenizer=tokenizer,
            smiles_col=args.finetune_smiles_col,
            weighted_by_affinity=False,
            max_rows=args.max_rows_finetune,
        )
        print(f"Finetune distribution built: {finetune_meta}")

        print("Building pretrain first-token distribution...")
        pretrain_dist, pretrain_meta = build_first_token_distribution(
            csv_path=args.pretrain_csv,
            tokenizer=tokenizer,
            smiles_col=args.pretrain_smiles_col,
            weighted_by_affinity=False,
            max_rows=args.max_rows_pretrain,
        )
        print(f"Pretrain distribution built: {pretrain_meta}")

        if args.distribution_cache_json:
            save_distribution_cache(
                args.distribution_cache_json,
                active_dist=active_dist,
                finetune_dist=finetune_dist,
                pretrain_dist=pretrain_dist,
            )
            print(f"Saved distribution cache: {args.distribution_cache_json}")

    mixed_dist = mix_distributions(
        active_dist,
        finetune_dist,
        pretrain_dist,
        w_active=args.mix_w_active,
        w_ft=args.mix_w_finetune,
        w_pre=args.mix_w_pretrain,
    )
    print(f"Mixed distribution size: {len(mixed_dist)} token ids")

    scaffolds = load_scaffolds(args.scaffold, args.scaffold_file, args.scaffold_col)
    print(f"Loaded {len(scaffolds)} scaffold(s).")

    decoder_start_token_id = resolve_decoder_start_id(model, tokenizer)
    rng = random.Random(args.seed)
    rows = []

    print("Generating molecules...")
    with torch.no_grad():
        for scaffold in scaffolds:
            encoder_inputs = tokenizer(
                scaffold,
                max_length=args.max_input_length,
                truncation=True,
                padding=False,
                return_tensors="pt",
            )
            input_ids = encoder_inputs["input_ids"].to(device)
            attention_mask = encoder_inputs["attention_mask"].to(device)

            for sample_idx in range(args.num_samples_per_scaffold):
                first_token_id = sample_token_id(mixed_dist, rng)
                decoder_input_ids = torch.tensor(
                    [[decoder_start_token_id, first_token_id]],
                    dtype=torch.long,
                    device=device,
                )

                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
                first_token_text = tokenizer.decode([first_token_id], skip_special_tokens=True)

                rows.append(
                    {
                        "scaffold": scaffold,
                        "sample_idx": sample_idx,
                        "sampled_first_token_id": int(first_token_id),
                        "sampled_first_token_text": first_token_text,
                        "generated_smiles": decoded,
                    }
                )

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scaffold",
                "sample_idx",
                "sampled_first_token_id",
                "sampled_first_token_text",
                "generated_smiles",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Saved {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
