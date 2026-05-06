"""Merges all Stage 1 consolidated outputs into a single master dataset
and applies configurable filters (exchange whitelist, security type,
date range). No columns are removed and no rows are synthesised.

"""

from pathlib import Path
from typing import Optional
import logging

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_FILTERS = {
    "exchange_whitelist": ["NYSE", "NASDAQ"],
    "type_whitelist": ["Common Stock"],
    "start_date": "1999-01-04",
    "end_date": None,
    "min_history_days": None,  # Explicitly None to prevent survivorship bias
}


def load_equities(file_path: Path) -> pd.DataFrame:
    """Load consolidated equities data, downcasting floats to save memory."""
    if not file_path.exists():
        logger.error(f"Equities file not found: {file_path}")
        return pd.DataFrame()

    logger.info(f"Loading equities from: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["Date"])

    numeric_cols = df.select_dtypes(include=["float64"]).columns
    for col in numeric_cols:
        df[col] = df[col].astype("float32")

    logger.info(f"  Loaded {len(df):,} rows, {df['Ticker'].nunique():,} tickers")
    return df


def load_yields(file_path: Path) -> pd.DataFrame:
    """Load consolidated yields data."""
    if not file_path.exists():
        logger.error(f"Yields file not found: {file_path}")
        return pd.DataFrame()

    logger.info(f"Loading yields from: {file_path}")
    df = pd.read_csv(file_path, parse_dates=["Date"])
    logger.info(f"  Loaded {len(df):,} rows, {df['Series'].nunique():,} series")
    return df


def load_commodities(file_path: Path) -> pd.DataFrame:
    """Load consolidated commodities data."""
    if not file_path.exists():
        logger.error(f"Commodities file not found: {file_path}")
        return pd.DataFrame()

    logger.info(f"Loading commodities from: {file_path}")
    df = pd.read_csv(file_path, parse_dates=["Date"])
    logger.info(f"  Loaded {len(df):,} rows, {df['Asset'].nunique():,} assets")
    return df


def load_metadata(file_path: Path) -> pd.DataFrame:
    """Load ticker metadata for exchange and type filtering."""
    if not file_path.exists():
        logger.error(f"Metadata file not found: {file_path}")
        return pd.DataFrame()

    logger.info(f"Loading metadata from: {file_path}")
    df = pd.read_csv(file_path, dtype=str)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()

    # Align column name with equities convention
    if "Code" in df.columns and "Ticker" not in df.columns:
        df = df.rename(columns={"Code": "Ticker"})

    logger.info(f"  Loaded metadata for {len(df):,} tickers")
    return df


def apply_equity_filters(
    equities_df: pd.DataFrame, metadata_df: pd.DataFrame, filters: dict
) -> pd.DataFrame:
    """Apply exchange, type, and date range filters to equities data."""
    if equities_df.empty:
        return equities_df

    initial_count = len(equities_df)

    # Left join preserves equities that might lack metadata
    df = equities_df.merge(metadata_df, on="Ticker", how="left")

    if filters.get("exchange_whitelist"):
        valid = filters["exchange_whitelist"]
        logger.info(f"  Filtering Exchange: {valid}")
        df = df[df["Exchange"].isin(valid)]

    if filters.get("type_whitelist"):
        valid = filters["type_whitelist"]
        logger.info(f"  Filtering Type: {valid}")
        df = df[df["Type"].isin(valid)]

    if filters.get("start_date"):
        df = df[df["Date"] >= pd.to_datetime(filters["start_date"])]
    if filters.get("end_date"):
        df = df[df["Date"] <= pd.to_datetime(filters["end_date"])]

    logger.info(f"  Filtered: {initial_count:,} -> {len(df):,} rows")
    return df


def apply_date_filter(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply date range filter to non-equity asset classes."""
    if df.empty:
        return df

    if filters.get("start_date"):
        df = df[df["Date"] >= pd.to_datetime(filters["start_date"])]
    if filters.get("end_date"):
        df = df[df["Date"] <= pd.to_datetime(filters["end_date"])]
    return df


def standardise_yields_schema(yields_df: pd.DataFrame) -> pd.DataFrame:
    """Rename yields columns to match the common schema (no broadcasting)."""
    if yields_df.empty:
        return pd.DataFrame()

    logger.info("  Standardising yields schema (No broadcasting)...")

    df = yields_df.rename(columns={"Series": "Ticker", "Value": "Close"})
    df["AssetClass"] = "yield"
    return df


def standardise_commodities_schema(commodities_df: pd.DataFrame) -> pd.DataFrame:
    """Rename commodities columns to match the common schema.

    Appends _commodity to ticker names to avoid collision with equity
    tickers (e.g. GOLD equity vs GOLD spot).
    """
    if commodities_df.empty:
        return pd.DataFrame()

    df = commodities_df.rename(columns={"Asset": "Ticker"})
    df["Ticker"] = df["Ticker"].astype(str) + "_commodity"
    df["AssetClass"] = "commodity"
    return df


def standardise_equities_schema(equities_df: pd.DataFrame) -> pd.DataFrame:
    """Add the AssetClass tag to equities."""
    if equities_df.empty:
        return pd.DataFrame()

    df = equities_df.copy()
    df["AssetClass"] = "equity"
    return df


def merge_all_assets(
    equities_df: pd.DataFrame, yields_df: pd.DataFrame, commodities_df: pd.DataFrame
) -> pd.DataFrame:
    """Combine all asset classes, preserving all columns via concatenation."""
    logger.info("Merging asset classes...")

    # Non-overlapping columns become NaN automatically
    combined = pd.concat([equities_df, yields_df, commodities_df], ignore_index=True)

    logger.info("Sorting merged data...")
    combined = combined.sort_values(["Date", "AssetClass", "Ticker"])

    # Place core columns first for readability
    core_cols = [
        "Date",
        "Ticker",
        "AssetClass",
        "Open",
        "High",
        "Low",
        "Close",
        "Adjusted_close",
        "Volume",
    ]
    existing_cols = [c for c in core_cols if c in combined.columns]
    other_cols = [c for c in combined.columns if c not in core_cols]

    return combined[existing_cols + other_cols]


def export_master(df: pd.DataFrame, output_path: Path) -> None:
    """Export the merged master dataset to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting to {output_path}...")
    df.to_csv(output_path, index=False)

    logger.info(f"Export complete. Size: {output_path.stat().st_size / 1e6:.2f} MB")


def main(
    equities_path: Path,
    yields_path: Path,
    commodities_path: Path,
    metadata_path: Path,
    output_path: Path,
    filters: Optional[dict] = None,
) -> None:
    """Orchestrate the Stage 2 merge-and-filter pipeline."""
    if filters is None:
        filters = DEFAULT_FILTERS

    logger.info("=" * 60)
    logger.info("STAGE 2: Merge and Filter Master Dataset (Pure Merge)")
    logger.info("=" * 60)

    equities = load_equities(equities_path)
    yields = load_yields(yields_path)
    commodities = load_commodities(commodities_path)
    metadata = load_metadata(metadata_path)

    equities_filtered = apply_equity_filters(equities, metadata, filters)

    yields_filtered = apply_date_filter(yields, filters)
    commodities_filtered = apply_date_filter(commodities, filters)

    logger.info("Standardising schemas...")
    equities_std = standardise_equities_schema(equities_filtered)
    yields_std = standardise_yields_schema(yields_filtered)
    commodities_std = standardise_commodities_schema(commodities_filtered)

    master = merge_all_assets(equities_std, yields_std, commodities_std)

    export_master(master, output_path)

    logger.info("STAGE 2 COMPLETE")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    main(
        equities_path=PROJECT_ROOT / "data" / "interim" / "equities_consolidated.csv",
        yields_path=PROJECT_ROOT / "data" / "interim" / "macro_consolidated.csv",
        commodities_path=PROJECT_ROOT
        / "data"
        / "interim"
        / "commodities_consolidated.csv",
        metadata_path=PROJECT_ROOT
        / "data"
        / "raw"
        / "metadata"
        / "ticker_metadata_master.csv",
        output_path=PROJECT_ROOT / "data" / "interim" / "master_consolidated.csv",
    )
