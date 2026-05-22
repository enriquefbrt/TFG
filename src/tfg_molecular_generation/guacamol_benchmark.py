import argparse
import csv
import json
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm.auto import tqdm

from tfg_molecular_generation.ape_hf_wrapper import APEHuggingFaceTokenizer
from tfg_molecular_generation.decorator_utils import (
    attach_decorators_to_scaffold,
    parse_decorator_sequence,
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


TAG_OPEN_RE = re.compile(r"<\s*R\s*(\d+)\s*>")
TAG_CLOSE_RE = re.compile(r"<\s*/\s*R\s*(\d+)\s*>")
LEGACY_DUMMY_RE = re.compile(r"\[\s*:\s*(\d+)\s*\]")


def _normalize_decorators_text(text: str) -> str:
    """
    Normalize decoder outputs to parser-friendly format:
    - < R 1 > ... < / R 1 > -> <R1> ... </R1>
    - [:1] -> [*:1]
    - If a block starts with plain label token ("1 C..."), prepend [*:1].
    """
    normalized = text or ""
    normalized = TAG_OPEN_RE.sub(r"<R\1>", normalized)
    normalized = TAG_CLOSE_RE.sub(r"</R\1>", normalized)
    normalized = LEGACY_DUMMY_RE.sub(r"[*:\1]", normalized)
    normalized = " ".join(normalized.split())

    def _fix_block(match: re.Match) -> str:
        label = int(match.group(1))
        content = (match.group(2) or "").strip()
        if not content:
            return f"<R{label}> </R{label}>"
        content = content.replace(" ", "")
        if "[*:" not in content:
            content = re.sub(rf"^\s*{label}\s*", "", content).strip()
            content = f"[*:{label}]{content}".strip()
        return f"<R{label}> {content} </R{label}>"

    normalized = re.sub(r"<R(\d+)>\s*(.*?)\s*</R\1>", _fix_block, normalized, flags=re.DOTALL)
    normalized = " ".join(normalized.split())

    # Rebuild from parsed blocks so duplicates/truncated tails are dropped and
    # all blocks are returned in canonical parser-friendly form.
    parsed = parse_decorator_sequence(normalized)
    if not parsed:
        return ""
    rebuilt = [f"<R{label}> {parsed[label]} </R{label}>" for label in sorted(parsed.keys())]
    return " ".join(rebuilt)


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
        eval_batch_size: int = 1,
        num_return_sequences: int = 1,
        max_input_length: int = 128,
        show_progress: bool = True,
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
        self.eval_batch_size = max(1, int(eval_batch_size))
        self.num_return_sequences = max(1, int(num_return_sequences))
        self.show_progress = bool(show_progress)
        self.rng = random.Random(seed)
        self.stats = GenerationStats()
        self._warned_unexpected_generate_shape = False
        self._warned_shape_fix = False
        self._stable_decode_chunk_size = self.eval_batch_size
        if num_beams > 1 and self.num_return_sequences > num_beams:
            raise ValueError(
                "num_return_sequences must be <= num_beams when num_beams > 1. "
                f"Got num_return_sequences={self.num_return_sequences}, num_beams={num_beams}."
            )

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

    def _run_generate_decoded(
        self,
        scaffolds: List[str],
        num_return_sequences: int,
    ) -> List[str]:
        if not scaffolds:
            return []

        encoder_inputs = self.tokenizer(
            scaffolds,
            max_length=self.max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoder_inputs["input_ids"].to(self.device)
        attention_mask = encoder_inputs["attention_mask"].to(self.device)

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_start_token_id=self.decoder_start_token_id,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=False,
        )
        generated = self._normalize_generate_output(generated)
        if isinstance(generated, torch.Tensor):
            generated = generated.detach().cpu()
        return self.tokenizer.batch_decode(generated, skip_special_tokens=False)

    def _normalize_generate_output(self, generated: torch.Tensor) -> torch.Tensor:
        if hasattr(generated, "sequences"):
            generated = generated.sequences
        if not isinstance(generated, torch.Tensor):
            generated = torch.as_tensor(generated)

        if generated.ndim == 1:
            generated = generated.unsqueeze(0)
        elif generated.ndim > 2:
            if not self._warned_shape_fix:
                print(
                    f"[GuacaMol] Warning: generate output ndim={generated.ndim}; "
                    "flattening to 2D [num_sequences, seq_len]."
                )
                self._warned_shape_fix = True
            generated = generated.reshape(-1, generated.shape[-1])
        return generated

    def _decode_scaffolds_as_pairs(
        self, scaffolds: List[str], chunk_size: int
    ) -> Optional[List[tuple]]:
        pairs: List[tuple] = []
        for start in range(0, len(scaffolds), chunk_size):
            chunk = scaffolds[start : start + chunk_size]
            decoded = self._run_generate_decoded(
                chunk, num_return_sequences=self.num_return_sequences
            )
            expected = len(chunk) * self.num_return_sequences
            if len(decoded) != expected:
                return None
            for i, scaffold in enumerate(chunk):
                for j in range(self.num_return_sequences):
                    pairs.append((scaffold, decoded[i * self.num_return_sequences + j]))
        return pairs

    def _generate_batch_from_scaffolds(self, scaffolds: List[str]) -> List[Optional[str]]:
        if not scaffolds:
            return []
        scaffold_and_raw: List[tuple] = []

        start_chunk = min(max(1, self._stable_decode_chunk_size), len(scaffolds))
        chunk_size = start_chunk
        recovered_pairs: Optional[List[tuple]] = None

        while chunk_size >= 1:
            recovered_pairs = self._decode_scaffolds_as_pairs(scaffolds, chunk_size)
            if recovered_pairs is not None:
                if chunk_size < self._stable_decode_chunk_size:
                    self._stable_decode_chunk_size = chunk_size
                    print(
                        "[GuacaMol] Adjusted stable decode chunk size to "
                        f"{self._stable_decode_chunk_size} due to generate-shape instability."
                    )
                break
            if not self._warned_unexpected_generate_shape:
                expected_total = len(scaffolds) * self.num_return_sequences
                print(
                    "[GuacaMol] Warning: Unexpected generate output count "
                    f"(expected={expected_total}). "
                    "Retrying with smaller decode chunks."
                )
                self._warned_unexpected_generate_shape = True
            if chunk_size == 1:
                break
            chunk_size = max(1, chunk_size // 2)

        if recovered_pairs is not None:
            scaffold_and_raw = recovered_pairs
        else:
            # Last-resort fallback: keep attempt accounting stable even if decode output shape
            # is unusable in this batch.
            return [None] * (len(scaffolds) * self.num_return_sequences)

        out: List[Optional[str]] = []
        for scaffold, decoded_raw in scaffold_and_raw:
            decorators = _normalize_decorators_text(clean_decoded_text(decoded_raw))
            try:
                assembled = attach_decorators_to_scaffold(scaffold, decorators)
            except Exception:
                self.stats.assembly_failures += 1
                out.append(None)
                continue

            mol = Chem.MolFromSmiles(assembled)
            if mol is None:
                self.stats.invalid_smiles += 1
                out.append(None)
                continue
            out.append(Chem.MolToSmiles(mol, canonical=True))
        return out

    def _generate_attempts_for_scaffold(
        self,
        scaffold: str,
        n_attempts: int,
    ) -> List[Optional[str]]:
        """Generate exactly n_attempts candidates for one fixed scaffold."""
        n_attempts = max(0, int(n_attempts))
        if n_attempts == 0:
            return []
        results: List[Optional[str]] = []
        produced = 0
        consecutive_empty_batches = 0
        while produced < n_attempts:
            remaining = n_attempts - produced
            scaffold_batch_size = min(self.eval_batch_size, remaining)
            batch_scaffolds = [scaffold] * scaffold_batch_size
            batch = self._generate_batch_from_scaffolds(batch_scaffolds)
            if not batch:
                consecutive_empty_batches += 1
                if consecutive_empty_batches >= 10:
                    break
                continue
            consecutive_empty_batches = 0
            take = min(remaining, len(batch))
            results.extend(batch[:take])
            produced += take
        return results

    def generate(self, number_samples: int) -> List[str]:
        target = int(number_samples)
        if target <= 0:
            return []

        generated: List[str] = []
        max_attempts = target * self.attempts_multiplier

        with torch.no_grad():
            consecutive_empty_batches = 0
            with tqdm(
                total=max_attempts,
                desc="[GuacaMol] Sampling attempts",
                unit="try",
                dynamic_ncols=True,
                disable=not self.show_progress,
                leave=False,
            ) as pbar:
                while len(generated) < target and self.stats.attempts < max_attempts:
                    remaining_attempts = max_attempts - self.stats.attempts
                    if remaining_attempts <= 0:
                        break

                    scaffold_batch_size = min(self.eval_batch_size, remaining_attempts)
                    scaffolds = [self._sample_scaffold() for _ in range(scaffold_batch_size)]
                    smiles_batch = self._generate_batch_from_scaffolds(scaffolds)
                    if not smiles_batch:
                        # No real sequences returned by generate() for this batch; skip safely.
                        consecutive_empty_batches += 1
                        if consecutive_empty_batches >= 10:
                            print(
                                "[GuacaMol] Stopping: generate() returned empty batches repeatedly."
                            )
                            break
                        continue
                    consecutive_empty_batches = 0
                    if len(smiles_batch) > remaining_attempts:
                        smiles_batch = smiles_batch[:remaining_attempts]

                    for smiles in smiles_batch:
                        self.stats.attempts += 1
                        pbar.update(1)
                        if smiles is not None:
                            self.stats.successes += 1
                            if len(generated) < target:
                                generated.append(smiles)

                        if (
                            self.stats.attempts == 1
                            or self.stats.attempts % 25 == 0
                            or smiles is not None
                        ):
                            success_rate = self.stats.successes / max(1, self.stats.attempts)
                            pbar.set_postfix(
                                valid=f"{len(generated)}/{target}",
                                succ_rate=f"{success_rate:.2%}",
                            )

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


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _conditioned_scaffold_core_smiles(scaffold_smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(scaffold_smiles)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    dummy_idxs = [atom.GetIdx() for atom in rw.GetAtoms() if atom.GetAtomicNum() == 0]
    for idx in sorted(dummy_idxs, reverse=True):
        rw.RemoveAtom(idx)
    core = rw.GetMol()
    if core.GetNumAtoms() == 0:
        return None
    try:
        Chem.SanitizeMol(core)
    except Exception:
        return None
    return Chem.MolToSmiles(core, canonical=True)


def _murcko_scaffold_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold_mol is None or scaffold_mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaffold_mol, canonical=True)


def _load_canonical_smiles_set(path: str) -> set:
    canonical = set()
    for line in _load_smiles_lines(path):
        smi = _canonicalize_smiles(line)
        if smi:
            canonical.add(smi)
    return canonical


def _save_scaffold_metrics_csv(rows: List[Dict[str, object]], output_csv: str) -> None:
    if not rows:
        print("[GuacaMol][ScaffoldEval] No rows to save in CSV.")
        return
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fieldnames = [
        "scaffold_index",
        "scaffold",
        "scaffold_core",
        "n_attempts",
        "n_valid",
        "n_unique",
        "n_novel",
        "n_similar",
        "validity",
        "uniqueness",
        "novelty",
        "similarity",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[GuacaMol][ScaffoldEval] Saved CSV: {output_csv} (rows={len(rows)})")


def _build_scaffold_boxplot(rows: List[Dict[str, object]], output_png: str) -> None:
    if not rows:
        print("[GuacaMol][ScaffoldEval] No rows available for boxplot.")
        return

    metrics = {
        "Validity": [float(r["validity"]) for r in rows],
        "Uniqueness": [float(r["uniqueness"]) for r in rows],
        "Novelty": [float(r["novelty"]) for r in rows],
        "Similarity": [float(r["similarity"]) for r in rows],
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(metrics.keys())
    data = [metrics[k] for k in labels]
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True)
    palette = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Per-Scaffold Metric Distributions (Paper-Style)")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    print(f"[GuacaMol][ScaffoldEval] Saved boxplot: {output_png}")


def _save_scaffold_eval_json(rows: List[Dict[str, object]], output_json: str) -> None:
    if not rows:
        print("[GuacaMol][ScaffoldEval] No rows available for JSON summary.")
        return

    def summarize(key: str) -> Dict[str, float]:
        vals = [float(r[key]) for r in rows]
        return {
            "mean": float(statistics.fmean(vals)),
            "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
            "min": float(min(vals)),
            "max": float(max(vals)),
            "median": float(statistics.median(vals)),
        }

    payload = {
        "n_scaffolds": len(rows),
        "summary": {
            "validity": summarize("validity"),
            "uniqueness": summarize("uniqueness"),
            "novelty": summarize("novelty"),
            "similarity": summarize("similarity"),
        },
        "rows": rows,
    }
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[GuacaMol][ScaffoldEval] Saved JSON: {output_json}")


def _evaluate_scaffold_distributions(
    generator: "ScaffoldConditionedGuacaMolGenerator",
    scaffold_pool: List[str],
    canonical_training_smiles: set,
    n_scaffolds: int,
    n_samples_per_scaffold: int,
    seed: int,
    show_progress: bool,
) -> List[Dict[str, object]]:
    n_scaffolds = max(1, int(n_scaffolds))
    n_samples_per_scaffold = max(1, int(n_samples_per_scaffold))

    rng = random.Random(seed)
    unique_pool = list(dict.fromkeys(scaffold_pool))
    if len(unique_pool) <= n_scaffolds:
        selected_scaffolds = unique_pool
    else:
        selected_scaffolds = rng.sample(unique_pool, n_scaffolds)

    rows: List[Dict[str, object]] = []
    print(
        "[GuacaMol][ScaffoldEval] Running paper-style evaluation with "
        f"{len(selected_scaffolds)} scaffolds x {n_samples_per_scaffold} molecules/scaffold"
    )

    with torch.no_grad():
        scaffold_iter = enumerate(selected_scaffolds, start=1)
        if show_progress:
            scaffold_iter = enumerate(
                tqdm(
                    selected_scaffolds,
                    total=len(selected_scaffolds),
                    desc="[GuacaMol][ScaffoldEval] Scaffolds",
                    unit="scf",
                    dynamic_ncols=True,
                ),
                start=1,
            )
        for scaffold_idx, scaffold in scaffold_iter:
            target_core = _conditioned_scaffold_core_smiles(scaffold)
            attempts = generator._generate_attempts_for_scaffold(scaffold, n_samples_per_scaffold)
            valid_smiles: List[str] = []
            novel_count = 0
            similar_count = 0

            for generated in attempts:
                if generated is None:
                    continue

                valid_smiles.append(generated)
                if generated not in canonical_training_smiles:
                    novel_count += 1

                gen_core = _murcko_scaffold_smiles(generated)
                if target_core and gen_core and gen_core == target_core:
                    similar_count += 1

            valid_count = len(valid_smiles)
            unique_count = len(set(valid_smiles))
            attempts_count = len(attempts)
            if attempts_count == 0:
                validity = 0.0
                uniqueness = 0.0
                novelty = 0.0
                similarity = 0.0
            else:
                validity = valid_count / float(attempts_count)
                uniqueness = (unique_count / float(valid_count)) if valid_count else 0.0
                novelty = (novel_count / float(valid_count)) if valid_count else 0.0
                similarity = (similar_count / float(valid_count)) if valid_count else 0.0

            rows.append(
                {
                    "scaffold_index": scaffold_idx,
                    "scaffold": scaffold,
                    "scaffold_core": target_core or "",
                    "n_attempts": attempts_count,
                    "n_valid": valid_count,
                    "n_unique": unique_count,
                    "n_novel": novel_count,
                    "n_similar": similar_count,
                    "validity": validity,
                    "uniqueness": uniqueness,
                    "novelty": novelty,
                    "similarity": similarity,
                }
            )
    return rows


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
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Number of scaffolds decoded per generate() call (higher = faster, uses more VRAM).",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help=(
            "Number of sampled sequences per scaffold prompt in each decode call "
            "(higher = faster, uses more VRAM)."
        ),
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--no_progress_bar",
        action="store_true",
        help="Disable tqdm progress bars (useful for very clean nohup logs).",
    )
    parser.add_argument(
        "--paper_style_eval",
        action="store_true",
        help="Run additional per-scaffold evaluation (paper-style Fig.3 protocol).",
    )
    parser.add_argument(
        "--paper_num_scaffolds",
        type=int,
        default=100,
        help="Number of scaffolds sampled for paper-style per-scaffold evaluation.",
    )
    parser.add_argument(
        "--paper_samples_per_scaffold",
        type=int,
        default=100,
        help="Number of generated molecules per scaffold in paper-style evaluation.",
    )
    parser.add_argument(
        "--paper_metrics_csv_output",
        default=None,
        help="Optional CSV output for per-scaffold metrics (Validity/Uniqueness/Novelty/Similarity).",
    )
    parser.add_argument(
        "--paper_boxplot_output",
        default=None,
        help="Optional PNG output for paper-style per-scaffold boxplot.",
    )
    parser.add_argument(
        "--paper_json_output",
        default=None,
        help="Optional JSON output with per-scaffold rows plus aggregated summary statistics.",
    )
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
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
        eval_batch_size=args.eval_batch_size,
        num_return_sequences=args.num_return_sequences,
        show_progress=not args.no_progress_bar,
    )

    print("[GuacaMol] Building benchmark suite...")
    benchmarks = distribution_learning_benchmark_suite(
        chembl_file_path=args.chembl_training_file,
        version_name=args.benchmark_version,
        number_samples=args.number_samples,
    )

    results = []
    print(f"[GuacaMol] Running {len(benchmarks)} benchmarks...")
    benchmark_iter = enumerate(benchmarks, start=1)
    if not args.no_progress_bar:
        benchmark_iter = enumerate(
            tqdm(
                benchmarks,
                total=len(benchmarks),
                desc="[GuacaMol] Benchmarks",
                unit="task",
                dynamic_ncols=True,
            ),
            start=1,
        )
    for idx, benchmark in benchmark_iter:
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

    run_paper_eval = (
        bool(args.paper_style_eval)
        or bool(args.paper_metrics_csv_output)
        or bool(args.paper_boxplot_output)
        or bool(args.paper_json_output)
    )
    if run_paper_eval:
        canonical_training_smiles = _load_canonical_smiles_set(args.chembl_training_file)
        rows = _evaluate_scaffold_distributions(
            generator=generator,
            scaffold_pool=scaffold_pool,
            canonical_training_smiles=canonical_training_smiles,
            n_scaffolds=args.paper_num_scaffolds,
            n_samples_per_scaffold=args.paper_samples_per_scaffold,
            seed=args.seed,
            show_progress=not args.no_progress_bar,
        )
        if args.paper_metrics_csv_output:
            _save_scaffold_metrics_csv(rows, args.paper_metrics_csv_output)
        if args.paper_boxplot_output:
            _build_scaffold_boxplot(rows, args.paper_boxplot_output)
        if args.paper_json_output:
            _save_scaffold_eval_json(rows, args.paper_json_output)

        if rows:
            print(
                "[GuacaMol][ScaffoldEval] Means: "
                f"Validity={statistics.fmean([r['validity'] for r in rows]):.4f}, "
                f"Uniqueness={statistics.fmean([r['uniqueness'] for r in rows]):.4f}, "
                f"Novelty={statistics.fmean([r['novelty'] for r in rows]):.4f}, "
                f"Similarity={statistics.fmean([r['similarity'] for r in rows]):.4f}"
            )


if __name__ == "__main__":
    main()
