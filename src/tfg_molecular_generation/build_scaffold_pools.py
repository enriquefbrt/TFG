import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors

from tfg_molecular_generation.decorator_utils import smiles_to_scaffold_and_decorators

RDLogger.DisableLog("rdApp.error")


@dataclass
class CandidateScaffold:
    scaffold: str
    attach_points: int
    heavy_atoms: int
    rings: int
    hetero_atoms: int
    stratum: Tuple[str, str, str, str]


def _size_bin(heavy_atoms: int) -> str:
    if heavy_atoms <= 12:
        return "size_s"
    if heavy_atoms <= 20:
        return "size_m"
    if heavy_atoms <= 30:
        return "size_l"
    return "size_xl"


def _cap_bin(prefix: str, value: int, cap: int) -> str:
    if value >= cap:
        return f"{prefix}{cap}+"
    return f"{prefix}{value}"


def _stratum_key(attach_points: int, heavy_atoms: int, rings: int, hetero_atoms: int):
    return (
        _cap_bin("ap_", attach_points, 4),
        _size_bin(heavy_atoms),
        _cap_bin("rg_", rings, 3),
        _cap_bin("ht_", hetero_atoms, 3),
    )


def _count_attach_points(scaffold_mol: Chem.Mol) -> int:
    count = 0
    for atom in scaffold_mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() > 0:
            count += 1
    return count


def _count_hetero_atoms(scaffold_mol: Chem.Mol) -> int:
    hetero = 0
    for atom in scaffold_mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num in (0, 1, 6):
            continue
        hetero += 1
    return hetero


def collect_unique_candidates(
    input_smiles: str,
    max_source_smiles: Optional[int],
    min_attach_points: int,
    max_attach_points: int,
    min_heavy_atoms: int,
    max_heavy_atoms: int,
    progress_every: int,
) -> Tuple[List[CandidateScaffold], Dict[str, int]]:
    stats = Counter()
    seen_scaffolds = set()
    candidates: List[CandidateScaffold] = []

    with open(input_smiles, "r", encoding="utf-8") as f:
        for line_idx, raw in enumerate(f, start=1):
            if max_source_smiles is not None and line_idx > max_source_smiles:
                break
            smiles = raw.strip()
            if not smiles:
                stats["source_empty"] += 1
                continue

            stats["source_read"] += 1
            pair = smiles_to_scaffold_and_decorators(smiles)
            if pair is None:
                stats["skip_no_decomposition"] += 1
                continue
            scaffold = pair[0]
            if scaffold in seen_scaffolds:
                stats["skip_duplicate_scaffold"] += 1
                continue

            scaffold_mol = Chem.MolFromSmiles(scaffold)
            if scaffold_mol is None:
                stats["skip_invalid_scaffold"] += 1
                continue

            attach_points = _count_attach_points(scaffold_mol)
            if attach_points < min_attach_points or attach_points > max_attach_points:
                stats["skip_attach_filter"] += 1
                continue

            heavy_atoms = scaffold_mol.GetNumHeavyAtoms()
            if heavy_atoms < min_heavy_atoms or heavy_atoms > max_heavy_atoms:
                stats["skip_size_filter"] += 1
                continue

            rings = int(rdMolDescriptors.CalcNumRings(scaffold_mol))
            hetero_atoms = _count_hetero_atoms(scaffold_mol)
            stratum = _stratum_key(attach_points, heavy_atoms, rings, hetero_atoms)

            seen_scaffolds.add(scaffold)
            candidates.append(
                CandidateScaffold(
                    scaffold=scaffold,
                    attach_points=attach_points,
                    heavy_atoms=heavy_atoms,
                    rings=rings,
                    hetero_atoms=hetero_atoms,
                    stratum=stratum,
                )
            )
            stats["candidate_kept"] += 1

            if progress_every > 0 and line_idx % progress_every == 0:
                print(
                    f"[Collect] processed={line_idx:,} kept={stats['candidate_kept']:,} "
                    f"unique_seen={len(seen_scaffolds):,}"
                )

    return candidates, dict(stats)


def stratified_sample_indices(
    strata_to_indices: Dict[Tuple[str, str, str, str], List[int]],
    target_n: int,
    seed: int,
) -> List[int]:
    rng = random.Random(seed)
    all_indices = [idx for idxs in strata_to_indices.values() for idx in idxs]
    total = len(all_indices)
    if target_n >= total:
        return sorted(all_indices)
    if target_n <= 0:
        return []

    quota = {}
    fractional = []
    assigned = 0
    for key, idxs in strata_to_indices.items():
        exact = (len(idxs) / float(total)) * target_n
        q = min(len(idxs), int(math.floor(exact)))
        quota[key] = q
        assigned += q
        fractional.append((exact - q, key))

    remainder = target_n - assigned
    fractional.sort(reverse=True, key=lambda x: x[0])
    for _, key in fractional:
        if remainder <= 0:
            break
        if quota[key] < len(strata_to_indices[key]):
            quota[key] += 1
            remainder -= 1

    selected = []
    leftovers = []
    for key, idxs in strata_to_indices.items():
        q = quota[key]
        if q > 0:
            picked = rng.sample(idxs, q) if q < len(idxs) else list(idxs)
            selected.extend(picked)
            picked_set = set(picked)
            leftovers.extend([idx for idx in idxs if idx not in picked_set])
        else:
            leftovers.extend(idxs)

    if len(selected) < target_n:
        need = target_n - len(selected)
        if need > len(leftovers):
            selected.extend(leftovers)
        else:
            selected.extend(rng.sample(leftovers, need))
    elif len(selected) > target_n:
        selected = rng.sample(selected, target_n)

    return sorted(selected)


def make_morgan_fingerprints(scaffolds: Sequence[str], n_bits: int = 2048, radius: int = 2):
    valid_scaffolds = []
    fps = []
    for scaffold in scaffolds:
        mol = Chem.MolFromSmiles(scaffold)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        valid_scaffolds.append(scaffold)
        fps.append(fp)
    return valid_scaffolds, fps


def maxmin_select_indices(fps: Sequence, target_n: int, seed: int, progress_every: int) -> List[int]:
    n = len(fps)
    if target_n >= n:
        return list(range(n))
    if target_n <= 0:
        return []

    rng = random.Random(seed)
    start = rng.randrange(n)
    selected = [start]

    remaining_indices = [i for i in range(n) if i != start]
    remaining_fps = [fps[i] for i in remaining_indices]
    nearest_sim = np.array(
        DataStructs.BulkTanimotoSimilarity(fps[start], remaining_fps), dtype=np.float32
    )

    while len(selected) < target_n and remaining_indices:
        pick_pos = int(np.argmin(nearest_sim))
        pick_idx = remaining_indices.pop(pick_pos)
        pick_fp = remaining_fps.pop(pick_pos)
        nearest_sim = np.delete(nearest_sim, pick_pos)
        selected.append(pick_idx)

        if not remaining_indices:
            break

        sims = np.array(DataStructs.BulkTanimotoSimilarity(pick_fp, remaining_fps), dtype=np.float32)
        nearest_sim = np.maximum(nearest_sim, sims)

        if progress_every > 0 and len(selected) % progress_every == 0:
            print(f"[MaxMin] selected={len(selected):,}/{target_n:,}")

    return selected


def write_scaffold_file(path: str, scaffolds: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for scaffold in scaffolds:
            f.write(f"{scaffold}\n")


def summarize_distribution(candidates: Sequence[CandidateScaffold]):
    out = {
        "attach_points": Counter(),
        "size_bin": Counter(),
        "rings": Counter(),
        "hetero_atoms": Counter(),
    }
    for c in candidates:
        out["attach_points"][str(c.attach_points)] += 1
        out["size_bin"][_size_bin(c.heavy_atoms)] += 1
        out["rings"][str(c.rings)] += 1
        out["hetero_atoms"][str(c.hetero_atoms)] += 1
    return {k: dict(v) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build 2-level scaffold pools for scaffold-conditioned GuacaMol evaluation: "
            "(1) master library and (2) benchmark fixed pool."
        )
    )
    parser.add_argument("--input_smiles", default="data/guacamol_v1_train.smiles")
    parser.add_argument("--master_size", type=int, default=30000)
    parser.add_argument("--benchmark_size", type=int, default=3000)
    parser.add_argument("--max_source_smiles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--min_attach_points", type=int, default=1)
    parser.add_argument("--max_attach_points", type=int, default=6)
    parser.add_argument("--min_heavy_atoms", type=int, default=6)
    parser.add_argument("--max_heavy_atoms", type=int, default=50)

    parser.add_argument(
        "--master_out",
        default="data/scaffold_pools/scaffold_master_v1.txt",
    )
    parser.add_argument(
        "--benchmark_out",
        default="data/scaffold_pools/scaffold_benchmark_v1.txt",
    )
    parser.add_argument(
        "--metadata_out",
        default="data/scaffold_pools/scaffold_pools_metadata_v1.json",
    )
    parser.add_argument("--collect_progress_every", type=int, default=100000)
    parser.add_argument("--maxmin_progress_every", type=int, default=250)

    args = parser.parse_args()

    if args.benchmark_size > args.master_size:
        raise ValueError("--benchmark_size cannot be greater than --master_size.")
    if not os.path.isfile(args.input_smiles):
        raise FileNotFoundError(f"input_smiles not found: {args.input_smiles}")

    print("[Build] Collecting unique scaffold candidates...")
    candidates, collect_stats = collect_unique_candidates(
        input_smiles=args.input_smiles,
        max_source_smiles=args.max_source_smiles,
        min_attach_points=args.min_attach_points,
        max_attach_points=args.max_attach_points,
        min_heavy_atoms=args.min_heavy_atoms,
        max_heavy_atoms=args.max_heavy_atoms,
        progress_every=args.collect_progress_every,
    )
    if not candidates:
        raise RuntimeError("No candidates collected. Relax filters or verify input data.")
    print(f"[Build] Candidate scaffolds collected: {len(candidates):,}")

    print("[Build] Stratified sampling for master pool...")
    strata_to_indices = defaultdict(list)
    for idx, cand in enumerate(candidates):
        strata_to_indices[cand.stratum].append(idx)

    master_indices = stratified_sample_indices(
        strata_to_indices=strata_to_indices,
        target_n=min(args.master_size, len(candidates)),
        seed=args.seed,
    )
    master_candidates = [candidates[i] for i in master_indices]
    master_scaffolds = [c.scaffold for c in master_candidates]
    print(f"[Build] Master pool size: {len(master_scaffolds):,}")

    print("[Build] Computing fingerprints for master pool...")
    master_scaffolds_valid, master_fps = make_morgan_fingerprints(master_scaffolds, n_bits=2048, radius=2)
    if len(master_scaffolds_valid) < len(master_scaffolds):
        print(
            f"[Build] Warning: dropped {len(master_scaffolds) - len(master_scaffolds_valid)} "
            "master scaffolds with invalid fingerprint inputs."
        )
    if len(master_scaffolds_valid) < args.benchmark_size:
        raise RuntimeError(
            "Not enough valid master scaffolds to build benchmark pool. "
            "Reduce benchmark_size or relax filters."
        )

    print("[Build] MaxMin selection for benchmark pool...")
    benchmark_indices = maxmin_select_indices(
        fps=master_fps,
        target_n=args.benchmark_size,
        seed=args.seed,
        progress_every=args.maxmin_progress_every,
    )
    benchmark_scaffolds = [master_scaffolds_valid[i] for i in benchmark_indices]
    print(f"[Build] Benchmark pool size: {len(benchmark_scaffolds):,}")

    write_scaffold_file(args.master_out, master_scaffolds_valid)
    write_scaffold_file(args.benchmark_out, benchmark_scaffolds)
    print(f"[Build] Wrote master pool to: {args.master_out}")
    print(f"[Build] Wrote benchmark pool to: {args.benchmark_out}")

    metadata = {
        "input_smiles": args.input_smiles,
        "parameters": {
            "master_size": args.master_size,
            "benchmark_size": args.benchmark_size,
            "max_source_smiles": args.max_source_smiles,
            "seed": args.seed,
            "min_attach_points": args.min_attach_points,
            "max_attach_points": args.max_attach_points,
            "min_heavy_atoms": args.min_heavy_atoms,
            "max_heavy_atoms": args.max_heavy_atoms,
        },
        "collect_stats": collect_stats,
        "counts": {
            "candidates": len(candidates),
            "master_valid": len(master_scaffolds_valid),
            "benchmark": len(benchmark_scaffolds),
            "strata": len(strata_to_indices),
        },
        "distributions": {
            "candidates": summarize_distribution(candidates),
            "master": summarize_distribution(master_candidates),
        },
        "outputs": {
            "master_out": args.master_out,
            "benchmark_out": args.benchmark_out,
        },
    }
    os.makedirs(os.path.dirname(args.metadata_out) or ".", exist_ok=True)
    with open(args.metadata_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Build] Wrote metadata to: {args.metadata_out}")
    print("[Build] Done.")


if __name__ == "__main__":
    main()
