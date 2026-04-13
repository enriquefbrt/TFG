import csv
import json
import math
import re
from collections import defaultdict


SMILES_TOKEN_PATTERN = re.compile(
    r"\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"
)


def detect_delimiter(csv_path: str) -> str:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]).delimiter
    except Exception:
        return ";" if sample.count(";") > sample.count(",") else ","


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def resolve_column(fieldnames, explicit_name=None, candidates=None, purpose="column"):
    if not fieldnames:
        raise ValueError(f"No columns found while resolving {purpose}.")
    if explicit_name is not None:
        if explicit_name in fieldnames:
            return explicit_name
        normalized_lookup = {normalize_name(c): c for c in fieldnames}
        key = normalize_name(explicit_name)
        if key in normalized_lookup:
            return normalized_lookup[key]
        raise ValueError(
            f"Requested {purpose} '{explicit_name}' not found. Available columns: {fieldnames}"
        )
    normalized_lookup = {normalize_name(c): c for c in fieldnames}
    for candidate in candidates or []:
        key = normalize_name(candidate)
        if key in normalized_lookup:
            return normalized_lookup[key]
    raise ValueError(
        f"Could not auto-detect {purpose}. Available columns: {fieldnames}"
    )


def extract_first_chemical_token(smiles: str):
    if not smiles:
        return None
    parts = SMILES_TOKEN_PATTERN.findall(smiles)
    if not parts:
        return None
    return parts[0]


def chemical_token_to_token_id(chemical_token: str, tokenizer):
    encoded = tokenizer.encode(chemical_token, add_special_tokens=False)
    if not encoded:
        return None
    token_id = encoded[0]
    if tokenizer.unk_token_id is not None and token_id == tokenizer.unk_token_id:
        return None
    return token_id


def parse_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def quantile(sorted_values, q):
    if not sorted_values:
        return 0.0
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def infer_affinity_mode(affinity_column_name: str) -> str:
    name = normalize_name(affinity_column_name)
    if "pchembl" in name:
        return "higher_better"
    lower_better_markers = ["ic50", "ec50", "ki", "kd", "standard value", "nm"]
    for marker in lower_better_markers:
        if marker in name:
            return "lower_better"
    return "higher_better"


def normalize_distribution(values_by_token_id):
    total = float(sum(values_by_token_id.values()))
    if total <= 0:
        return {}
    return {int(k): float(v) / total for k, v in values_by_token_id.items() if v > 0}


def build_first_token_distribution(
    csv_path: str,
    tokenizer,
    smiles_col=None,
    affinity_col=None,
    affinity_mode="auto",
    tau=None,
    weighted_by_affinity=False,
    max_rows=None,
):
    delimiter = detect_delimiter(csv_path)
    counts = defaultdict(float)
    weighted_samples = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        smiles_column = resolve_column(
            reader.fieldnames,
            explicit_name=smiles_col,
            candidates=["Smiles", "smiles", "target_text", "canonical_smiles"],
            purpose="SMILES column",
        )
        affinity_column = None
        if weighted_by_affinity:
            affinity_column = resolve_column(
                reader.fieldnames,
                explicit_name=affinity_col,
                candidates=["pChEMBL Value", "pchembl_value", "Standard Value", "affinity"],
                purpose="affinity column",
            )
            if affinity_mode == "auto":
                affinity_mode = infer_affinity_mode(affinity_column)

        rows_seen = 0
        rows_used = 0
        for row in reader:
            rows_seen += 1
            if max_rows is not None and rows_seen > max_rows:
                break

            smiles = str(row.get(smiles_column, "")).strip()
            first_token = extract_first_chemical_token(smiles)
            if first_token is None:
                continue
            token_id = chemical_token_to_token_id(first_token, tokenizer)
            if token_id is None:
                continue

            if weighted_by_affinity:
                affinity_value = parse_float(row.get(affinity_column))
                if affinity_value is None:
                    continue
                weighted_samples.append((token_id, affinity_value))
            else:
                counts[token_id] += 1.0
                rows_used += 1

    if weighted_by_affinity:
        if not weighted_samples:
            raise ValueError(
                f"No valid weighted samples found in {csv_path}. "
                "Check affinity column and values."
            )

        if affinity_mode == "higher_better":
            scores = [value for _, value in weighted_samples]
        elif affinity_mode == "lower_better":
            scores = [-value for _, value in weighted_samples]
        else:
            raise ValueError(
                f"Invalid affinity_mode '{affinity_mode}'. Use higher_better/lower_better/auto."
            )

        sorted_scores = sorted(scores)
        median_score = quantile(sorted_scores, 0.5)
        if tau is None:
            iqr = quantile(sorted_scores, 0.75) - quantile(sorted_scores, 0.25)
            tau_value = max(iqr, 1e-6)
        else:
            tau_value = max(float(tau), 1e-6)

        for (token_id, _), score in zip(weighted_samples, scores):
            z = max(min((score - median_score) / tau_value, 60.0), -60.0)
            weight = 1.0 / (1.0 + math.exp(-z))
            counts[token_id] += weight
            rows_used += 1

    distribution = normalize_distribution(counts)
    metadata = {
        "csv_path": csv_path,
        "delimiter": delimiter,
        "rows_used": rows_used,
        "weighted": weighted_by_affinity,
    }
    return distribution, metadata


def mix_distributions(active_dist, finetune_dist, pretrain_dist, w_active, w_ft, w_pre):
    total_weight = w_active + w_ft + w_pre
    if total_weight <= 0:
        raise ValueError("Mix weights must have a positive sum.")
    w_active /= total_weight
    w_ft /= total_weight
    w_pre /= total_weight

    mixed = defaultdict(float)
    token_ids = set(active_dist) | set(finetune_dist) | set(pretrain_dist)
    for token_id in token_ids:
        mixed[token_id] = (
            w_active * active_dist.get(token_id, 0.0)
            + w_ft * finetune_dist.get(token_id, 0.0)
            + w_pre * pretrain_dist.get(token_id, 0.0)
        )

    mixed = normalize_distribution(mixed)
    if mixed:
        return mixed
    for fallback in (active_dist, finetune_dist, pretrain_dist):
        if fallback:
            return fallback
    raise ValueError("All first-token distributions are empty.")


def sample_token_id(distribution, rng):
    token_ids = list(distribution.keys())
    probs = [distribution[token_id] for token_id in token_ids]
    return rng.choices(token_ids, weights=probs, k=1)[0]


def load_scaffolds(scaffold, scaffold_file, scaffold_col=None):
    scaffolds = []

    if scaffold:
        scaffolds.append(scaffold.strip())

    if scaffold_file:
        if scaffold_file.lower().endswith(".csv"):
            delimiter = detect_delimiter(scaffold_file)
            with open(scaffold_file, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                resolved_scaffold_col = resolve_column(
                    reader.fieldnames,
                    explicit_name=scaffold_col,
                    candidates=["input_text", "scaffold", "smiles"],
                    purpose="scaffold column",
                )
                for row in reader:
                    value = str(row.get(resolved_scaffold_col, "")).strip()
                    if value:
                        scaffolds.append(value)
        else:
            with open(scaffold_file, "r", encoding="utf-8") as f:
                for line in f:
                    value = line.strip()
                    if value:
                        scaffolds.append(value)

    scaffolds = [scf for scf in scaffolds if scf]
    if not scaffolds:
        raise ValueError("No scaffolds provided. Use --scaffold or --scaffold_file.")
    return scaffolds


def save_distribution_cache(cache_path, active_dist, finetune_dist, pretrain_dist):
    payload = {
        "active_dist": {str(k): v for k, v in active_dist.items()},
        "finetune_dist": {str(k): v for k, v in finetune_dist.items()},
        "pretrain_dist": {str(k): v for k, v in pretrain_dist.items()},
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_distribution_cache(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return (
        {int(k): float(v) for k, v in payload["active_dist"].items()},
        {int(k): float(v) for k, v in payload["finetune_dist"].items()},
        {int(k): float(v) for k, v in payload["pretrain_dist"].items()},
    )


def resolve_decoder_start_id(model, tokenizer):
    if model.config.decoder_start_token_id is not None:
        return model.config.decoder_start_token_id
    if tokenizer.bos_token_id is not None:
        return tokenizer.bos_token_id
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    raise ValueError("Could not resolve decoder_start_token_id.")
