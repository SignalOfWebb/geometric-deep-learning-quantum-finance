"""Consolidates yield curve data from multiple countries into a single
long-format CSV. These series serve as global hub nodes in the graph
architecture and provide regime signals for the quantum layer.

References:
    Nelson, C. R. & Siegel, A. F. (1987). Parsimonious Modeling of Yield
    Curves. Journal of Business, 60(4), 473-489.
"""

import logging
import gc
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Mapping of expected yield series to their metadata
YIELD_FILES_CONFIG = {
    "US_10Y": {
        "frequency": "daily",
        "country": "US",
        "description": "10-Year Treasury Constant Maturity Rate",
    },
    "US_02Y": {
        "frequency": "daily",
        "country": "US",
        "description": "2-Year Treasury Constant Maturity Rate",
    },
    "UK_10Y": {
        "frequency": "monthly",
        "country": "UK",
        "description": "UK 10-Year Government Bond Yield",
    },
    "GER_10Y": {
        "frequency": "monthly",
        "country": "DE",
        "description": "Germany 10-Year Government Bond Yield",
    },
    "JPN_10Y": {
        "frequency": "monthly",
        "country": "JP",
        "description": "Japan 10-Year Government Bond Yield",
    },
}


def get_yield_files(raw_path: Path) -> Dict[str, Path]:
    """Discover yield CSV files and map them to series names."""
    if not raw_path.exists():
        logger.error(f"Raw path does not exist: {raw_path}")
        return {}

    found_files: Dict[str, Path] = {}
    csv_files = list(raw_path.glob("*YIELD*.csv"))
    if not csv_files:
        csv_files = list(raw_path.glob("*.csv"))

    for csv_file in csv_files:
        stem = csv_file.stem
        # Map filenames like US_10Y_YIELD to series US_10Y
        series_name = stem.replace("_YIELD", "") if stem.endswith("_YIELD") else stem
        if series_name in YIELD_FILES_CONFIG:
            found_files[series_name] = csv_file

    for expected in YIELD_FILES_CONFIG:
        if expected not in found_files:
            logger.warning(f"Expected yield file not found: {expected}")

    return found_files


def process_yield_file(
    file_path: Path, series_name: str, config: Dict[str, Any]
) -> Optional[pd.DataFrame]:
    """Read and standardise a single yield CSV, preserving all columns."""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.warning(f"Failed to read {series_name}: {e}")
        return None

    if len(df) == 0:
        return None

    df.columns = [str(c).strip() for c in df.columns]

    # Identify Date column
    date_col = None
    for col in df.columns:
        if col.lower() in ("date", "dates"):
            date_col = col
            break
    if not date_col:
        logger.warning(f"{series_name}: No date column found")
        return None

    # Identify Value column
    value_col = None
    for col in ("Close", "Value", "Yield", series_name, "value"):
        if col in df.columns:
            value_col = col
            break
    if not value_col:
        # Fallback to first numeric column that is not the date
        for col in df.columns:
            if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
                value_col = col
                break
    if not value_col:
        logger.warning(f"{series_name}: No value column found")
        return None

    # Standardise column names but keep all others
    df = df.rename(columns={date_col: "Date", value_col: "Value"})

    df["Series"] = series_name
    df["Frequency"] = config["frequency"]
    df["Country"] = config["country"]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Date", "Value"])

    # Place standard columns first for readability
    cols = ["Date", "Series", "Value", "Frequency", "Country"]
    other_cols = [c for c in df.columns if c not in cols]
    df = df[cols + other_cols]

    return df.sort_values("Date")


def consolidate_stream(yield_files: Dict[str, Path], output_path: Path) -> None:
    """Consolidate yield files by streaming each to disk sequentially."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    is_first = True

    logger.info(f"Streaming consolidation to {output_path}")

    for series_name, file_path in tqdm(yield_files.items(), desc="Processing macro"):
        config = YIELD_FILES_CONFIG.get(series_name)
        df = process_yield_file(file_path, series_name, config)

        if df is not None:
            mode = "w" if is_first else "a"
            header = is_first

            df.to_csv(output_path, mode=mode, header=header, index=False)

            total_rows += len(df)
            is_first = False

            del df
            gc.collect()

    if is_first:
        logger.warning("No macro data processed.")
    else:
        logger.info(f"Macro consolidation complete: {total_rows:,} rows")


def main(raw_macro_path: Path, output_path: Path) -> None:
    """Orchestrate the macro yield consolidation pipeline."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Consolidate Macro (Yields) Data (Streaming)")
    logger.info("=" * 60)

    yield_files = get_yield_files(raw_macro_path)
    logger.info(f"Found {len(yield_files)} yield files")

    if not yield_files:
        logger.warning("No yield files found to process.")
        return

    consolidate_stream(yield_files, output_path)

    logger.info("=" * 60)
    logger.info("STAGE 1 (YIELDS) COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    RAW_MACRO = PROJECT_ROOT / "data" / "raw" / "macro"
    OUTPUT_FILE = PROJECT_ROOT / "data" / "interim" / "macro_consolidated.csv"

    main(raw_macro_path=RAW_MACRO, output_path=OUTPUT_FILE)
