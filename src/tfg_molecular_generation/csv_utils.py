from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

DEFAULT_SEPARATORS = (",", ";", "\t", "|")


def _sniff_delimiter(path: str, candidates: Iterable[str]) -> str | None:
    """Best-effort delimiter detection from the first chunk of a file."""
    try:
        sample = Path(path).read_text(encoding="utf-8", errors="ignore")[:65536]
        if not sample.strip():
            return None
        dialect = csv.Sniffer().sniff(sample, delimiters=list(candidates))
        return dialect.delimiter
    except Exception:
        return None


def _looks_like_wrong_delimiter(columns, used_sep: str, all_candidates: Tuple[str, ...]) -> bool:
    """
    Heuristic: if we got a single giant header containing another known separator,
    parsing likely used the wrong delimiter.
    """
    if len(columns) != 1:
        return False
    header = str(columns[0])
    for candidate in all_candidates:
        if candidate != used_sep and candidate in header:
            return True
    return False


def read_csv_auto_sep(path: str, **read_csv_kwargs) -> tuple[pd.DataFrame, str]:
    """
    Reads CSV/TSV-like files with automatic delimiter fallback.
    Returns (dataframe, detected_separator).
    """
    separators = tuple(read_csv_kwargs.pop("separators", DEFAULT_SEPARATORS))
    sniffed = _sniff_delimiter(path, separators)
    ordered_seps = []
    if sniffed in separators:
        ordered_seps.append(sniffed)
    ordered_seps.extend([sep for sep in separators if sep not in ordered_seps])

    last_error = None
    for sep in ordered_seps:
        try:
            df = pd.read_csv(path, sep=sep, **read_csv_kwargs)
            if _looks_like_wrong_delimiter(df.columns, sep, separators):
                continue
            return df, sep
        except Exception as exc:  # pragma: no cover
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ValueError(f"Could not parse CSV file: {path}")
