import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from rdkit import Chem

from tfg_molecular_generation.ape_hf_wrapper import APEHuggingFaceTokenizer
from tfg_molecular_generation.decorator_utils import (
    attach_decorators_to_scaffold,
    smiles_to_scaffold_and_decorators,
)
from tfg_molecular_generation.inference import clean_decoded_text, load_model_for_inference
from tfg_molecular_generation.inference_utils import resolve_decoder_start_id


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_smiles_file(path: str) -> List[str]:
    values = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            v = line.strip()
            if v:
                values.append(v)
    return values


def load_scaffold_file(path: str) -> List[str]:
    scaffolds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value:
                scaffolds.append(value)
    return scaffolds


def build_scaffold_pool_from_smiles(smiles_file: str, max_scaffolds: int) -> List[str]:
    pool: List[str] = []
    seen = set()
    for smiles in load_smiles_file(smiles_file):
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
class ScaffoldStats:
    pulls: int = 0
    score_sum: float = 0.0
    best_score: float = 0.0
    generated: int = 0
    valid_generated: int = 0

    @property
    def mean_score(self) -> float:
        if self.pulls == 0:
            return 0.0
        return self.score_sum / float(self.pulls)

    @property
    def validity_rate(self) -> float:
        if self.generated == 0:
            return 0.0
        return self.valid_generated / float(self.generated)


class ScaffoldBanditGoalDirectedGenerator:
    """
    Goal-directed optimizer for scaffold-conditioned generators.

    The search loop uses:
    - scaffold-level UCB-like selection,
    - stochastic decoding around selected scaffolds,
    - batch scoring and elite archive.
    """

    def __init__(
        self,
        model_dir: str,
        tokenizer_dir: str,
        scaffold_pool: List[str],
        seed: int = 42,
        eval_budget_multiplier: int = 40,
        min_eval_budget: int = 2000,
        rounds_patience: int = 20,
        min_improvement: float = 1e-4,
        scaffolds_per_round: int = 24,
        candidates_per_scaffold: int = 4,
        attempts_per_candidate: int = 3,
        exploration_weight: float = 0.35,
        prior_mean_score: float = 0.10,
        top_archive_multiplier: int = 8,
        max_input_length: int = 128,
        max_new_tokens: int = 128,
        temperature_start: float = 1.20,
        temperature_end: float = 0.85,
        top_p_start: float = 0.98,
        top_p_end: float = 0.90,
        repetition_penalty: float = 1.05,
        num_beams: int = 1,
        verbose_every: int = 5,
    ):
        if not scaffold_pool:
            raise ValueError("Scaffold pool is empty.")

        self.scaffold_pool = list(dict.fromkeys(scaffold_pool))
        self.seed = seed
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.eval_budget_multiplier = max(1, int(eval_budget_multiplier))
        self.min_eval_budget = max(1, int(min_eval_budget))
        self.rounds_patience = max(1, int(rounds_patience))
        self.min_improvement = float(min_improvement)
        self.scaffolds_per_round = max(1, int(scaffolds_per_round))
        self.candidates_per_scaffold = max(1, int(candidates_per_scaffold))
        self.attempts_per_candidate = max(1, int(attempts_per_candidate))
        self.exploration_weight = float(exploration_weight)
        self.prior_mean_score = float(prior_mean_score)
        self.top_archive_multiplier = max(2, int(top_archive_multiplier))
        self.max_input_length = int(max_input_length)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature_start = float(temperature_start)
        self.temperature_end = float(temperature_end)
        self.top_p_start = float(top_p_start)
        self.top_p_end = float(top_p_end)
        self.repetition_penalty = float(repetition_penalty)
        self.num_beams = int(num_beams)
        self.verbose_every = max(1, int(verbose_every))

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

        print(f"[GoalDirected] Device: {self.device}")
        print(f"[GoalDirected] Scaffold pool size: {len(self.scaffold_pool)}")

    def _derive_scaffold(self, smiles: str) -> Optional[str]:
        pair = smiles_to_scaffold_and_decorators(smiles)
        if pair is None:
            return None
        return pair[0]

    def _schedule(self, round_idx: int, total_rounds: int) -> Tuple[float, float]:
        if total_rounds <= 1:
            progress = 1.0
        else:
            progress = min(max(round_idx / float(total_rounds - 1), 0.0), 1.0)
        temperature = self.temperature_start + (self.temperature_end - self.temperature_start) * progress
        top_p = self.top_p_start + (self.top_p_end - self.top_p_start) * progress
        return max(0.05, temperature), min(max(top_p, 0.50), 0.999)

    def _scaffold_utility(
        self,
        scaffold: str,
        stats: Dict[str, ScaffoldStats],
        total_pulls: int,
        starting_scaffolds: set,
    ) -> float:
        st = stats[scaffold]
        mean = self.prior_mean_score if st.pulls == 0 else st.mean_score
        best = st.best_score
        validity = st.validity_rate
        explore = self.exploration_weight * math.sqrt(
            math.log(total_pulls + 2.0) / float(st.pulls + 1)
        )
        start_bonus = 0.03 if scaffold in starting_scaffolds else 0.0
        return mean + 0.20 * best + 0.05 * validity + explore + start_bonus

    def _select_scaffolds(
        self,
        stats: Dict[str, ScaffoldStats],
        n_select: int,
        starting_scaffolds: set,
    ) -> List[str]:
        total_pulls = sum(s.pulls for s in stats.values())
        utilities = []
        for scaffold in self.scaffold_pool:
            utilities.append(self._scaffold_utility(scaffold, stats, total_pulls, starting_scaffolds))

        max_u = max(utilities)
        exp_weights = [math.exp(min(30.0, u - max_u)) for u in utilities]
        selected = self.rng.choices(self.scaffold_pool, weights=exp_weights, k=n_select)

        # Force a bit of pure exploration each round.
        n_explore = max(1, n_select // 6)
        for i in range(n_explore):
            selected[i] = self.rng.choice(self.scaffold_pool)
        return selected

    def _decode_one(self, scaffold: str, temperature: float, top_p: float) -> Optional[str]:
        encoded = self.tokenizer(
            scaffold,
            max_length=self.max_input_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        decoder_input_ids = torch.tensor(
            [[self.decoder_start_token_id]], dtype=torch.long, device=self.device
        )

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_beams=self.num_beams,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        decoded_raw = self.tokenizer.decode(generated[0], skip_special_tokens=False)
        decorators = clean_decoded_text(decoded_raw)
        try:
            molecule = attach_decorators_to_scaffold(scaffold, decorators)
        except Exception:
            return None
        return canonicalize_smiles(molecule)

    def _propose_candidates(
        self,
        selected_scaffolds: List[str],
        stats: Dict[str, ScaffoldStats],
        temperature: float,
        top_p: float,
        seen_smiles: set,
    ) -> Tuple[List[str], List[str]]:
        proposals: List[str] = []
        proposal_scaffolds: List[str] = []
        with torch.no_grad():
            for scaffold in selected_scaffolds:
                for _ in range(self.candidates_per_scaffold):
                    stats[scaffold].generated += 1
                    candidate = None
                    for _attempt in range(self.attempts_per_candidate):
                        candidate = self._decode_one(scaffold, temperature=temperature, top_p=top_p)
                        if candidate is not None:
                            break
                    if candidate is None:
                        continue
                    stats[scaffold].valid_generated += 1
                    if candidate in seen_smiles:
                        continue
                    seen_smiles.add(candidate)
                    proposals.append(candidate)
                    proposal_scaffolds.append(scaffold)
        return proposals, proposal_scaffolds

    def generate_optimized_molecules(self, scoring_function, number_molecules: int, starting_population=None):
        target = int(number_molecules)
        if target <= 0:
            return []

        eval_budget = max(self.min_eval_budget, target * self.eval_budget_multiplier)
        max_rounds = max(20, eval_budget // max(1, self.scaffolds_per_round * self.candidates_per_scaffold))
        archive_limit = target * self.top_archive_multiplier

        scaffold_stats = {s: ScaffoldStats() for s in self.scaffold_pool}
        starting_scaffolds = set()
        archive_scores: Dict[str, float] = {}
        seen_smiles = set()
        eval_calls = 0

        # Warm-start from benchmark-provided starting population.
        initial_smiles = []
        if starting_population:
            for sm in starting_population:
                can = canonicalize_smiles(sm)
                if can is None:
                    continue
                if can in seen_smiles:
                    continue
                seen_smiles.add(can)
                initial_smiles.append(can)
                scaffold = self._derive_scaffold(can)
                if scaffold and scaffold in scaffold_stats:
                    starting_scaffolds.add(scaffold)

        if initial_smiles:
            init_scores = scoring_function.score_list(initial_smiles)
            eval_calls += len(initial_smiles)
            for sm, score in zip(initial_smiles, init_scores):
                archive_scores[sm] = float(score)

        best_score = max(archive_scores.values()) if archive_scores else -1e9
        stagnant_rounds = 0

        for round_idx in range(max_rounds):
            if eval_calls >= eval_budget or stagnant_rounds >= self.rounds_patience:
                break

            temperature, top_p = self._schedule(round_idx=round_idx, total_rounds=max_rounds)
            selected_scaffolds = self._select_scaffolds(
                stats=scaffold_stats,
                n_select=self.scaffolds_per_round,
                starting_scaffolds=starting_scaffolds,
            )
            proposals, proposal_scaffolds = self._propose_candidates(
                selected_scaffolds=selected_scaffolds,
                stats=scaffold_stats,
                temperature=temperature,
                top_p=top_p,
                seen_smiles=seen_smiles,
            )
            if not proposals:
                stagnant_rounds += 1
                continue

            remaining_budget = max(0, eval_budget - eval_calls)
            if remaining_budget == 0:
                break
            if len(proposals) > remaining_budget:
                proposals = proposals[:remaining_budget]
                proposal_scaffolds = proposal_scaffolds[:remaining_budget]

            scores = scoring_function.score_list(proposals)
            eval_calls += len(proposals)
            improved = False

            for sm, score, scaffold in zip(proposals, scores, proposal_scaffolds):
                s = float(score)
                archive_scores[sm] = s

                st = scaffold_stats[scaffold]
                st.pulls += 1
                st.score_sum += s
                if s > st.best_score:
                    st.best_score = s

                if s > best_score + self.min_improvement:
                    best_score = s
                    improved = True

            if len(archive_scores) > archive_limit:
                top_items = sorted(archive_scores.items(), key=lambda kv: kv[1], reverse=True)[:archive_limit]
                archive_scores = dict(top_items)
                seen_smiles = set(archive_scores.keys())

            if improved:
                stagnant_rounds = 0
            else:
                stagnant_rounds += 1

            if (round_idx + 1) % self.verbose_every == 0:
                current_top = sorted(archive_scores.values(), reverse=True)[:5]
                top_mean = sum(current_top) / len(current_top) if current_top else float("nan")
                print(
                    "[GoalDirected] "
                    f"round={round_idx + 1}/{max_rounds} "
                    f"eval_calls={eval_calls}/{eval_budget} "
                    f"archive={len(archive_scores)} "
                    f"best={best_score:.4f} top5_mean={top_mean:.4f} "
                    f"temp={temperature:.3f} top_p={top_p:.3f} "
                    f"stagnant_rounds={stagnant_rounds}"
                )

        ranked = sorted(archive_scores.items(), key=lambda kv: kv[1], reverse=True)
        molecules = [sm for sm, _ in ranked[:target]]
        if not molecules:
            raise RuntimeError("Search produced zero valid molecules.")
        if len(molecules) < target:
            # Padding with top molecules keeps deterministic behavior; benchmark handles duplicates.
            pad_source = molecules.copy()
            while len(molecules) < target:
                molecules.append(self.rng.choice(pad_source))
        return molecules


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run GuacaMol goal-directed benchmark (20 tasks) with a scaffold-conditioned model "
            "using bandit-guided search on top of the decoder."
        )
    )
    parser.add_argument("--model_dir", required=True, help="Trained model dir (full model or LoRA adapter dir).")
    parser.add_argument("--tokenizer_dir", required=True, help="APE tokenizer dir.")
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
        help="SMILES file to derive scaffold prompt pool.",
    )
    parser.add_argument("--max_scaffolds", type=int, default=3000)
    parser.add_argument("--benchmark_version", default="v1", choices=["v1", "v2", "trivial"])
    parser.add_argument("--json_output_file", default="outputs/guacamol_goal_directed.json")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_budget_multiplier", type=int, default=40)
    parser.add_argument("--min_eval_budget", type=int, default=2000)
    parser.add_argument("--rounds_patience", type=int, default=20)
    parser.add_argument("--min_improvement", type=float, default=1e-4)
    parser.add_argument("--scaffolds_per_round", type=int, default=24)
    parser.add_argument("--candidates_per_scaffold", type=int, default=4)
    parser.add_argument("--attempts_per_candidate", type=int, default=3)
    parser.add_argument("--exploration_weight", type=float, default=0.35)
    parser.add_argument("--prior_mean_score", type=float, default=0.10)
    parser.add_argument("--top_archive_multiplier", type=int, default=8)

    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature_start", type=float, default=1.20)
    parser.add_argument("--temperature_end", type=float, default=0.85)
    parser.add_argument("--top_p_start", type=float, default=0.98)
    parser.add_argument("--top_p_end", type=float, default=0.90)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--verbose_every", type=int, default=5)
    args = parser.parse_args()

    try:
        from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "guacamol is not installed. Install first: pip install guacamol==0.5.5"
        ) from exc

    if args.scaffold_pool_file:
        if not os.path.isfile(args.scaffold_pool_file):
            raise FileNotFoundError(f"scaffold_pool_file not found: {args.scaffold_pool_file}")
        scaffold_pool = load_scaffold_file(args.scaffold_pool_file)
        if args.max_scaffolds and len(scaffold_pool) > args.max_scaffolds:
            scaffold_pool = scaffold_pool[: args.max_scaffolds]
        print(
            f"[GoalDirected] Loaded scaffold pool from file: {args.scaffold_pool_file} "
            f"(n={len(scaffold_pool)})"
        )
    else:
        if not os.path.isfile(args.scaffold_source_smiles_file):
            raise FileNotFoundError(
                f"scaffold_source_smiles_file not found: {args.scaffold_source_smiles_file}"
            )
        scaffold_pool = build_scaffold_pool_from_smiles(
            smiles_file=args.scaffold_source_smiles_file,
            max_scaffolds=max(1, int(args.max_scaffolds)),
        )
    if not scaffold_pool:
        raise ValueError("No scaffolds available. Check scaffold_pool_file or scaffold_source_smiles_file.")

    generator = ScaffoldBanditGoalDirectedGenerator(
        model_dir=args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        scaffold_pool=scaffold_pool,
        seed=args.seed,
        eval_budget_multiplier=args.eval_budget_multiplier,
        min_eval_budget=args.min_eval_budget,
        rounds_patience=args.rounds_patience,
        min_improvement=args.min_improvement,
        scaffolds_per_round=args.scaffolds_per_round,
        candidates_per_scaffold=args.candidates_per_scaffold,
        attempts_per_candidate=args.attempts_per_candidate,
        exploration_weight=args.exploration_weight,
        prior_mean_score=args.prior_mean_score,
        top_archive_multiplier=args.top_archive_multiplier,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        top_p_start=args.top_p_start,
        top_p_end=args.top_p_end,
        repetition_penalty=args.repetition_penalty,
        num_beams=args.num_beams,
        verbose_every=args.verbose_every,
    )

    os.makedirs(os.path.dirname(args.json_output_file) or ".", exist_ok=True)
    print(
        f"[GoalDirected] Running benchmark suite={args.benchmark_version} "
        f"with output={args.json_output_file}"
    )
    assess_goal_directed_generation(
        goal_directed_molecule_generator=generator,
        json_output_file=args.json_output_file,
        benchmark_version=args.benchmark_version,
    )
    print(f"[GoalDirected] Done. Results saved to: {args.json_output_file}")


if __name__ == "__main__":
    main()
