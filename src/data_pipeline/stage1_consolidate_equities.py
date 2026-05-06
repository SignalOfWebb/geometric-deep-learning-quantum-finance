"""Consolidates individual raw equity CSV files into a single long-format
dataset. Processes files in batches to keep memory usage bounded and
applies suffix-based filtering to exclude non-common-stock instruments.
"""

import logging
import re
import gc
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Suffix patterns to exclude (warrants, units, and other non-common-stock
# instruments that EODHD mislabels as common stock)
EXCLUDED_SUFFIXES = [
    r"-WT$",   # Warrants
    r"-WS$",   # Warrants (alternate)
    r"-U$",    # Units
    r"-WI$",   # When-issued
    r"-R$",    # Rights
    r"-P[A-Z]?$",  # Preferred stock variants
]

SUFFIX_PATTERN = re.compile("|".join(EXCLUDED_SUFFIXES), re.IGNORECASE)

# Columns that must exist in every raw CSV
REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

DEFAULT_BATCH_SIZE = 500


def get_equity_files(raw_path: Path) -> List[Path]:
    """Discover CSV files in the raw equities directory, excluding bad suffixes.

    Args:
        raw_path: Path to data/raw/equities/.

    Returns:
        Sorted list of CSV file paths that passed suffix filtering.
    """
    if not raw_path.exists():
        logger.error(f"Raw path does not exist: {raw_path}")
        return []

    files = list(raw_path.glob("*.csv"))

    filtered_files = []
    excluded_count = 0

    for file in files:
        ticker = file.stem
        if not SUFFIX_PATTERN.search(ticker):
            filtered_files.append(file)
        else:
            excluded_count += 1

    logger.info(f"Filtered {excluded_count} files based on suffix pattern.")
    return sorted(filtered_files)


def extract_ticker_from_path(file_path: Path) -> str:
    """Extract the ticker symbol from a file path."""
    return file_path.stem


def process_single_file(file_path: Path) -> Optional[pd.DataFrame]:
    """Read and validate a single equity CSV file, retaining all columns."""
    ticker = extract_ticker_from_path(file_path)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.warning(f"Failed to read {ticker}: {e}")
        return None

    if len(df) == 0:
        return None

    df.columns = [str(c).strip() for c in df.columns]

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        logger.warning(f"{ticker}: Missing required columns {missing_cols}")
        return None

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    if len(df) == 0:
        return None

    df.insert(1, "Ticker", ticker)

    # Cast standard numeric columns where present
    numeric_cols = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Date")
    return df


def consolidate_stream(
    files: List[Path], output_path: Path, batch_size: int = DEFAULT_BATCH_SIZE
) -> None:
    """Process files in batches and stream results directly to a CSV on disk.

    Args:
        files: List of file paths to process.
        output_path: Path to write the consolidated CSV.
        batch_size: Number of files per batch.
    """
    n_batches = (len(files) + batch_size - 1) // batch_size
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"Overwriting existing file: {output_path}")

    total_rows = 0
    is_first_batch = True

    logger.info(f"Streaming consolidation of {len(files)} files to {output_path}")

    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(files))
        batch_files = files[start_idx:end_idx]

        batch_dfs = []
        for file_path in batch_files:
            df = process_single_file(file_path)
            if df is not None:
                batch_dfs.append(df)

        if batch_dfs:
            batch_df = pd.concat(batch_dfs, ignore_index=True)

            mode = "w" if is_first_batch else "a"
            header = is_first_batch

            batch_df.to_csv(output_path, mode=mode, header=header, index=False)

            total_rows += len(batch_df)
            is_first_batch = False

            del batch_df
            del batch_dfs
            gc.collect()

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"Batch {batch_idx + 1}/{n_batches}: Processed up to {end_idx} files. Total rows so far: {total_rows:,}"
            )

    if is_first_batch:
        logger.warning("No valid data found in any files. Output file not created.")
    else:
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Consolidation complete. Written to: {output_path}")
        logger.info(f"  Total Rows: {total_rows:,}")
        logger.info(f"  File Size: {file_size_mb:.2f} MB")


def main(
    raw_equities_path: Path, output_path: Path, batch_size: int = DEFAULT_BATCH_SIZE
) -> None:
    """Orchestrate the streaming equity consolidation pipeline."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Consolidate Raw Equity Files (Streaming)")
    logger.info("=" * 60)

    logger.info(f"Scanning: {raw_equities_path}")
    files = get_equity_files(raw_equities_path)
    logger.info(f"Found {len(files):,} equity files (after suffix filter)")

    if not files:
        logger.warning("No files found to process. Exiting.")
        return

    consolidate_stream(files, output_path, batch_size=batch_size)

    logger.info("=" * 60)
    logger.info("STAGE 1 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    RAW_EQUITIES = PROJECT_ROOT / "data" / "raw" / "equities"
    OUTPUT_FILE = PROJECT_ROOT / "data" / "interim" / "equities_consolidated.csv"

    main(
        raw_equities_path=RAW_EQUITIES,
        output_path=OUTPUT_FILE,
    )
