"""Consolidates commodity and environmental data files into a single
long-format CSV. These assets serve as global hub nodes in the graph
architecture, broadcasting regime information to equity nodes.

References:
    Silvennoinen, A. & Thorp, S. (2013). Financialization, Crisis and
    Commodity Correlation Dynamics. Journal of International Financial
    Markets, Institutions and Money, 24, 42-65.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Mapping of expected environmental files and their clean names
ENVIRONMENTAL_FILES_CONFIG = {
    "GOLD_FULL": {
        "clean_name": "GOLD",
        "description": "Gold London PM Fixing (Spot)",
        "category": "precious_metal",
    },
    "SILVER_FULL": {
        "clean_name": "SILVER",
        "description": "Silver London Fixing (Spot)",
        "category": "precious_metal",
    },
    "CRUDE_OIL_FULL": {
        "clean_name": "CRUDE_OIL",
        "description": "WTI Crude Oil Spot Price",
        "category": "energy",
    },
    "VIX_FULL": {
        "clean_name": "VIX",
        "description": "CBOE Volatility Index",
        "category": "volatility_index",
    },
    "PLATINUM_FULL": {
        "clean_name": "PLATINUM",
        "description": "Platinum London PM Fixing (Spot)",
        "category": "precious_metal",
    },
}


def get_environmental_files(raw_path: Path) -> Dict[str, Path]:
    """Discover environmental CSV files and map them to clean names."""
    if not raw_path.exists():
        logger.error(f"Raw path does not exist: {raw_path}")
        return {}

    found_files: Dict[str, Path] = {}
    for csv_file in raw_path.glob("*.csv"):
        file_stem = csv_file.stem
        if file_stem in ENVIRONMENTAL_FILES_CONFIG:
            found_files[file_stem] = csv_file

    for expected in ENVIRONMENTAL_FILES_CONFIG:
        if expected not in found_files:
            logger.warning(f"Expected environmental file not found: {expected}")

    return found_files


def process_environmental_file(
    file_path: Path, file_stem: str, config: Dict[str, Any]
) -> Optional[pd.DataFrame]:
    """Read and standardise a single environmental asset CSV, preserving all columns."""
    clean_name = config.get("clean_name", file_stem)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.warning(f"Failed to read {file_stem}: {e}")
        return None

    if len(df) == 0:
        return None

    df.columns = [str(c).strip() for c in df.columns]
    col_lower = {c.lower(): c for c in df.columns}

    # Identify Date column
    date_col = None
    for candidate in ("date", "dates", "datetime"):
        if candidate in col_lower:
            date_col = col_lower[candidate]
            break
    if date_col is None:
        logger.warning(f"{file_stem}: No date column found")
        return None

    def standardise_numeric(df: pd.DataFrame, target: str, candidates: List[str]):
        """Find a candidate column and rename it to the target name."""
        found_orig = None
        for c in candidates:
            if c.lower() in col_lower:
                found_orig = col_lower[c.lower()]
                break

        if found_orig:
            df = df.rename(columns={found_orig: target})
            df[target] = pd.to_numeric(df[target], errors="coerce")
        return df

    # Standardise OHLCV columns
    df = df.rename(columns={date_col: "Date"})
    df = standardise_numeric(df, "Open", ["Open"])
    df = standardise_numeric(df, "High", ["High"])
    df = standardise_numeric(df, "Low", ["Low"])
    df = standardise_numeric(df, "Close", ["Close", "Adjusted_close", "Adj Close"])
    df = standardise_numeric(df, "Volume", ["Volume", "Vol"])

    df["Asset"] = clean_name

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    if "Close" not in df.columns or df["Close"].isna().all():
        logger.warning(f"{file_stem}: No valid Close prices found")
        return None

    # Place standard columns first, then everything else
    std_cols = ["Date", "Asset", "Open", "High", "Low", "Close", "Volume"]
    existing_std = [c for c in std_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in std_cols]
    df = df[existing_std + other_cols]

    return df.sort_values("Date")


def consolidate_batch(env_files: Dict[str, Path], output_path: Path) -> None:
    """Consolidate environmental files by concatenating in memory.

    Since commodity data is small (~33k rows), in-memory concatenation
    is safe and ensures consistent schema handling across files.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_dfs = []

    logger.info("Processing commodity files...")

    for file_stem, file_path in tqdm(env_files.items(), desc="Loading commodities"):
        config = ENVIRONMENTAL_FILES_CONFIG.get(file_stem)
        df = process_environmental_file(file_path, file_stem, config)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        logger.warning("No environmental data processed.")
        return

    logger.info("Concatenating datasets...")
    # Concat handles schema mismatches by filling missing columns with NaN
    consolidated = pd.concat(all_dfs, ignore_index=True)

    logger.info(f"Saving to {output_path}...")
    consolidated.to_csv(output_path, index=False)

    logger.info(f"Commodities consolidation complete: {len(consolidated):,} rows")


def main(raw_env_path: Path, output_path: Path) -> None:
    """Orchestrate the commodities consolidation pipeline."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Consolidate Commodities (Environmental) Data")
    logger.info("=" * 60)

    env_files = get_environmental_files(raw_env_path)
    logger.info(f"Found {len(env_files)} environmental files")

    if not env_files:
        logger.warning("No environmental files found to process.")
        return

    consolidate_batch(env_files, output_path)

    logger.info("=" * 60)
    logger.info("STAGE 1 (COMMODITIES) COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    RAW_ENV = PROJECT_ROOT / "data" / "raw" / "macro"
    OUTPUT_FILE = PROJECT_ROOT / "data" / "interim" / "commodities_consolidated.csv"

    main(raw_env_path=RAW_ENV, output_path=OUTPUT_FILE)
