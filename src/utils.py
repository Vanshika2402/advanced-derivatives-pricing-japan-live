import os
import pandas as pd


def ensure_dirs(*paths: str) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def normalize_option_type(value: str) -> str:
    raw = str(value).upper().strip()

    if raw in ["CALL", "C", "CE"]:
        return "CALL"

    if raw in ["PUT", "P", "PE"]:
        return "PUT"

    raise ValueError(f"Invalid option type: {value}")


def to_float(value, default=float("nan")) -> float:
    try:
        if pd.isna(value):
            return default
        return float(str(value).replace(",", "").strip())
    except Exception:
        return default
