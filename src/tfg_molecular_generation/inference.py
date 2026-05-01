import argparse
import csv
import os

import torch
from transformers import T5ForConditionalGeneration

from tfg_molecular_generation.ape_hf_wrapper import APEHuggingFaceTokenizer
from tfg_molecular_generation.decorator_utils import attach_decorators_to_scaffold
from tfg_molecular_generation.inference_utils import load_scaffolds, resolve_decoder_start_id


def clean_decoded_text(text: str) -> str:
    # Keep decorator tags; only remove tokenizer control tokens.
    cleaned = (
        (text or "")
        .replace("<pad>", " ")
        .replace("<s>", " ")
        .replace("</s>", " ")
        .replace("<unk>", " ")
        .replace("<mask>", " ")
    )
    return " ".join(cleaned.split())


def main():
    parser = argparse.ArgumentParser(
        description="Scaffold-conditioned decorator generation and deterministic scaffold assembly."
    )
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model folder.")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Path to APE tokenizer folder.")
    parser.add_argument(
        "--scaffold",
        type=str,
        default=None,
        help="Single scaffold SMILES with labeled attachment points (e.g. c1cc([*:1])ccc1[*:2]).",
    )
    parser.add_argument("--scaffold_file", type=str, default=None, help="TXT/CSV file containing scaffolds.")
    parser.add_argument("--scaffold_col", type=str, default=None, help="Scaffold column if scaffold_file is CSV.")
    parser.add_argument("--output_csv", type=str, default="generated_scaffold_samples.csv", help="Output CSV path.")
    parser.add_argument("--num_samples_per_scaffold", type=int, default=20, help="Samples generated per scaffold.")
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_beams", type=int, default=1, help="Beam size. Keep 1 when sampling.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--skip_assembly",
        action="store_true",
        help="If set, only outputs generated decorator text without assembling final molecules.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        raise ValueError(f"model_dir not found: {args.model_dir}")
    if not os.path.isdir(args.tokenizer_dir):
        raise ValueError(f"tokenizer_dir not found: {args.tokenizer_dir}")

    print("Loading tokenizer and model...")
    tokenizer = APEHuggingFaceTokenizer(ape_tokenizer_path=args.tokenizer_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.bos_token_id
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Using device: {device}")

    scaffolds = load_scaffolds(args.scaffold, args.scaffold_file, args.scaffold_col)
    print(f"Loaded {len(scaffolds)} scaffold(s).")

    decoder_start_token_id = resolve_decoder_start_id(model, tokenizer)
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
                decoder_input_ids = torch.tensor([[decoder_start_token_id]], dtype=torch.long, device=device)

                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    repetition_penalty=args.repetition_penalty,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                decoded_raw = tokenizer.decode(generated[0], skip_special_tokens=False)
                decorator_text = clean_decoded_text(decoded_raw)

                assembled_smiles = ""
                assembly_error = ""
                valid_assembly = False
                if not args.skip_assembly:
                    try:
                        assembled_smiles = attach_decorators_to_scaffold(scaffold, decorator_text)
                        valid_assembly = True
                    except Exception as exc:
                        assembly_error = str(exc)

                rows.append(
                    {
                        "scaffold": scaffold,
                        "sample_idx": sample_idx,
                        "generated_decorators": decorator_text,
                        "assembled_smiles": assembled_smiles,
                        "assembled_valid": int(valid_assembly),
                        "assembly_error": assembly_error,
                    }
                )

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scaffold",
                "sample_idx",
                "generated_decorators",
                "assembled_smiles",
                "assembled_valid",
                "assembly_error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Saved {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
