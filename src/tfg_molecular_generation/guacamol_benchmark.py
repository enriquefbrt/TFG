import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
from rdkit import Chem

from tfg_molecular_generation.ape_hf_wrapper import APEHuggingFaceTokenizer
from tfg_molecular_generation.decorator_utils import (
    attach_decorators_to_scaffold,
    smiles_to_scaffold_and_decorators,
)
from tfg_molecular_generation.inference import clean_decoded_text, load_model_for_inference
from tfg_molecular_generation.inference_utils import resolve_decoder_start_id


def _load_smiles_lines(path: str) -> List[str]:
    smiles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value:
                smiles.append(value)
    return smiles


def _load_scaffold_lines(path: str) -> List[str]:
    scaffolds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value:
                scaffolds.append(value)
    return scaffolds


def _build_scaffold_pool_from_smiles(smiles_file: str, max_scaffolds: int) -> List[str]:
    pool: List[str] = []
    seen = set()
    for smiles in _load_smiles_lines(smiles_file):
        pair = smiles_to_scaffold_and_decorators(smiles)
        if pair is None:
            continue
        scaffold, _ = pair
        if scaffold in seen:
            continue
        seen.add(scaffold)
        pool.append(scaffold)
        if len(pool) >= max_scaffolds:
            break
    return pool


@dataclass
class GenerationStats:
    attempts: int = 0
    successes: int = 0
    assembly_failures: int = 0
    invalid_smiles: int = 0


class ScaffoldConditionedGuacaMolGenerator:
    """
    Adapter that behaves like guacamol.distribution_matching_generator.DistributionMatchingGenerator.
    """

    def __init__(
        self,
        model_dir: str,
        tokenizer_dir: str,
        scaffold_pool: List[str],
        temperature: float,
        top_p: float,
        num_beams: int,
        repetition_penalty: float,
        max_new_tokens: int,
        seed: int,
        attempts_multiplier: int,
        max_input_length: int = 128,
    ):
        if not scaffold_pool:
            raise ValueError("Scaffold pool is empty. Cannot run benchmark.")

        self.scaffold_pool = scaffold_pool
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.max_input_length = max_input_length
        self.attempts_multiplier = max(1, int(attempts_multiplier))
        self.rng = random.Random(seed)
        self.stats = GenerationStats()

        self.tokenizer = APEHuggingFaceTokenizer(ape_tokenizer_path=tokenizer_dir)
        self.model = load_model_for_inference(model_dir)

        if self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if self.tokenizer.eos_token_id is not None:
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
        if self.model.config.decoder_start_token_id is None:
            self.model.config.decoder_start_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.bos_token_id
            )

        self.decoder_start_token_id = resolve_decoder_start_id(self.model, self.tokenizer)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[GuacaMol] Device: {self.device}")
        print(f"[GuacaMol] Scaffolds in pool: {len(self.scaffold_pool)}")

    def _sample_scaffold(self) -> str:
        return self.rng.choice(self.scaffold_pool)

    def _generate_one_from_scaffold(self, scaffold: str) -> Optional[str]:
        encoder_inputs = self.tokenizer(
            scaffold,
            max_length=self.max_input_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids = encoder_inputs["input_ids"].to(self.device)
        attention_mask = encoder_inputs["attention_mask"].to(self.device)

        decoder_input_ids = torch.tensor(
            [[self.decoder_start_token_id]], dtype=torch.long, device=self.device
        )

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        decoded_raw = self.tokenizer.decode(generated[0], skip_special_tokens=False)
        decorators = clean_decoded_text(decoded_raw)

        try:
            assembled = attach_decorators_to_scaffold(scaffold, decorators)
        except Exception:
            self.stats.assembly_failures += 1
            return None

        mol = Chem.MolFromSmiles(assembled)
        if mol is None:
            self.stats.invalid_smiles += 1
            return None
        return Chem.MolToSmiles(mol, canonical=True)

    def generate(self, number_samples: int) -> List[str]:
        target = int(number_samples)
        if target <= 0:
            return []

        generated: List[str] = []
        max_attempts = target * self.attempts_multiplier

        with torch.no_grad():
            while len(generated) < target and self.stats.attempts < max_attempts:
                scaffold = self._sample_scaffold()
                self.stats.attempts += 1
                smiles = self._generate_one_from_scaffold(scaffold)
                if smiles is None:
                    continue
                generated.append(smiles)
                self.stats.successes += 1

        if len(generated) < target:
            if not generated:
                raise RuntimeError(
                    "Could not generate any valid molecules. "
                    "Increase --attempts_multiplier or check model quality."
                )
            while len(generated) < target:
                generated.append(self.rng.choice(generated))

        return generated


def _benchmark_results_to_json(
    results,
    output_json: str,
    benchmark_version: str,
    number_samples: int,
    stats: GenerationStats,
    scaffold_pool_size: int,
):
    payload = {
        "benchmark_suite_version": benchmark_version,
        "number_samples": number_samples,
        "timestamp_epoch": int(time.time()),
        "results": [vars(r) for r in results],
        "generation_stats": {
            "attempts": stats.attempts,
            "successes": stats.successes,
            "assembly_failures": stats.assembly_failures,
            "invalid_smiles": stats.invalid_smiles,
            "scaffold_pool_size": scaffold_pool_size,
        },
    }
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run GuacaMol distribution-learning benchmarks with this scaffold-conditioned model. "
            "Note: this is an adapted setting (prompted generation), not directly leaderboard-comparable."
        )
    )
    parser.add_argument("--model_dir", required=True, help="Trained model dir (full model or LoRA adapter dir).")
    parser.add_argument("--tokenizer_dir", required=True, help="APE tokenizer dir.")
    parser.add_argument(
        "--chembl_training_file",
        default="data/guacamol_v1_train.smiles",
        help="Reference training set used by GuacaMol for novelty/KL/FCD.",
    )
    parser.add_argument(
        "--scaffold_pool_file",
        default=None,
        help=(
            "Optional prebuilt scaffold pool (one scaffold per line). "
            "If provided, this is used directly and scaffold_source_smiles_file is ignored."
        ),
    )
    parser.add_argument(
        "--scaffold_source_smiles_file",
        default="data/guacamol_v1_test.smiles",
        help="SMILES file used to derive scaffold prompts.",
    )
    parser.add_argument(
        "--max_scaffolds",
        type=int,
        default=2000,
        help="Maximum unique scaffolds to keep in prompt pool.",
    )
    parser.add_argument("--benchmark_version", default="v1", choices=["v1", "v2"])
    parser.add_argument(
        "--number_samples",
        type=int,
        default=10000,
        help="Number of molecules requested by each benchmark metric.",
    )
    parser.add_argument("--json_output_file", default="outputs/guacamol_distribution_learning.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attempts_multiplier", type=int, default=8)
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    args = parser.parse_args()

    try:
        from guacamol.benchmark_suites import distribution_learning_benchmark_suite
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "guacamol is not installed. Install in your training/eval env first: pip install guacamol==0.5.5"
        ) from exc

    if not os.path.isfile(args.chembl_training_file):
        raise FileNotFoundError(f"chembl_training_file not found: {args.chembl_training_file}")
    if args.scaffold_pool_file:
        if not os.path.isfile(args.scaffold_pool_file):
            raise FileNotFoundError(f"scaffold_pool_file not found: {args.scaffold_pool_file}")
        scaffold_pool = _load_scaffold_lines(args.scaffold_pool_file)
        if args.max_scaffolds and len(scaffold_pool) > args.max_scaffolds:
            scaffold_pool = scaffold_pool[: args.max_scaffolds]
        print(
            f"[GuacaMol] Loaded scaffold pool from file: {args.scaffold_pool_file} "
            f"(n={len(scaffold_pool)})"
        )
    else:
        if not os.path.isfile(args.scaffold_source_smiles_file):
            raise FileNotFoundError(
                f"scaffold_source_smiles_file not found: {args.scaffold_source_smiles_file}"
            )
        scaffold_pool = _build_scaffold_pool_from_smiles(
            args.scaffold_source_smiles_file,
            max_scaffolds=max(1, args.max_scaffolds),
        )
    if not scaffold_pool:
        raise ValueError(
            "No scaffolds available. Check scaffold_pool_file or scaffold_source_smiles_file."
        )

    generator = ScaffoldConditionedGuacaMolGenerator(
        model_dir=args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        scaffold_pool=scaffold_pool,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        max_input_length=args.max_input_length,
        seed=args.seed,
        attempts_multiplier=args.attempts_multiplier,
    )

    print("[GuacaMol] Building benchmark suite...")
    benchmarks = distribution_learning_benchmark_suite(
        chembl_file_path=args.chembl_training_file,
        version_name=args.benchmark_version,
        number_samples=args.number_samples,
    )

    results = []
    print(f"[GuacaMol] Running {len(benchmarks)} benchmarks...")
    for idx, benchmark in enumerate(benchmarks, start=1):
        print(f"[GuacaMol] {idx}/{len(benchmarks)} - {benchmark.name}")
        result = benchmark.assess_model(generator)
        print(
            f"[GuacaMol] score={result.score:.6f} | "
            f"sampling_time={result.sampling_time:.2f}s"
        )
        results.append(result)

    _benchmark_results_to_json(
        results=results,
        output_json=args.json_output_file,
        benchmark_version=args.benchmark_version,
        number_samples=args.number_samples,
        stats=generator.stats,
        scaffold_pool_size=len(scaffold_pool),
    )
    print(f"[GuacaMol] Saved results to {args.json_output_file}")
    print(
        "[GuacaMol] Generation stats: "
        f"attempts={generator.stats.attempts}, "
        f"successes={generator.stats.successes}, "
        f"assembly_failures={generator.stats.assembly_failures}, "
        f"invalid_smiles={generator.stats.invalid_smiles}"
    )


if __name__ == "__main__":
    main()
