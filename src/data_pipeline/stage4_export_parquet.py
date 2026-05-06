"""Converts the feature-engineered CSV to Parquet format for efficient
model training. Also validates data integrity, optimises dtypes, and
generates schema metadata before export.

All columns from Stage 3 are retained in the exported Parquet file
(Alternative B — keep everything, add a feature mask). The model reads
only the columns listed in MODEL_FEATURE_COLS. This preserves full
diagnostic and evaluation flexibility without irreversible column
dropping during active development.

References:
    McKinney, W. (2010). Data Structures for Statistical Computing
    in Python. Proceedings of the 9th Python in Science Conference.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# The 12 features the model reads from the fact table.
# Mirrors MODEL_FEATURE_COLS in stage3_engineer_features.py exactly.
# This is the single source of truth for the feature mask in this stage.
# All other columns are retained in the Parquet file but ignored by the model.
MODEL_FEATURE_COLS: list[str] = [
    # Returns — where has this stock been recently
    "log_return_5d",
    "log_return_20d",
    # Volatility — how risky is it at multiple time scales
    "vol_10d",
    "vol_22d",
    "vol_60d",
    # Momentum — is the trend persistent (Jegadeesh & Titman, 1993)
    "mom_3m",
    "mom_6m",
    "mom_12m",
    # Liquidity — how easily can it be traded
    "volume_ratio",
    "amihud_illiquidity",
    # Price level — where is it relative to its own history
    "high_low_spread_22d",
    "price_to_52w_high",
]

# Columns whose absence is a hard error (structurally required)
REQUIRED_COLUMNS = ["Date", "Ticker", "AssetClass"]

# Columns where NaN is treated as a warning, not a blocking error
WARN_NULL_COLUMNS = ["Close"]

# Low-cardinality string columns to convert to categorical
CATEGORICAL_COLUMNS = ["Ticker", "AssetClass", "Country", "Exchange", "Currency", "Type"]


def validate_data_integrity(df: pd.DataFrame) -> dict:
    """Check required columns, types, duplicates, and data quality.

    Uses a tiered approach: missing required columns and wrong types
    are hard errors; NaN in price columns and duplicates are warnings.
    Also verifies that all MODEL_FEATURE_COLS are present in the
    dataframe — a missing model feature column is a hard error.

    Args:
        df: Feature-engineered DataFrame.

    Returns:
        Dictionary with 'passed', 'errors', and 'warnings' keys.
    """
    results: dict = {
        "passed": True,
        "errors": [],
        "warnings": [],
    }

    # Check required index columns exist and have no NaN
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            results["errors"].append(f"Missing required column: {col}")
            results["passed"] = False
        else:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                results["errors"].append(f"{col}: {null_count:,} NaN values")
                results["passed"] = False

    # Check all model feature columns are present — missing one means
    # the feature engineering stage did not complete correctly
    equity_df = df[df["AssetClass"] == "equity"] if "AssetClass" in df.columns else df
    missing_features = [
        col for col in MODEL_FEATURE_COLS if col not in df.columns
    ]
    if missing_features:
        results["errors"].append(
            f"Missing model feature columns: {missing_features}. "
            f"Re-run Stage 3 to regenerate."
        )
        results["passed"] = False
    else:
        logger.info(
            f"All {len(MODEL_FEATURE_COLS)} model feature columns present: {MODEL_FEATURE_COLS}"
        )

    # Check warn-level columns (NaN is a warning, not a failure)
    for col in WARN_NULL_COLUMNS:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                null_pct = null_count / len(df) * 100
                results["warnings"].append(
                    f"{col}: {null_count:,} NaN values ({null_pct:.2f}%)"
                )

    # Check Date dtype
    if "Date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            results["errors"].append("Date column is not datetime type")
            results["passed"] = False

    # Check for duplicate (Date, Ticker) pairs
    if "Date" in df.columns and "Ticker" in df.columns:
        duplicates = df.duplicated(subset=["Date", "Ticker"]).sum()
        if duplicates > 0:
            results["warnings"].append(
                f"Found {duplicates:,} duplicate (Date, Ticker) pairs"
            )

    # Check for extreme daily returns (|log_return_1d| > 1.0 = > 100%)
    # These are expected for penny stocks but worth flagging for awareness
    if "log_return_1d" in df.columns:
        extreme_returns = (df["log_return_1d"].abs() > 1.0).sum()
        if extreme_returns > 0:
            results["warnings"].append(
                f"Found {extreme_returns:,} extreme returns (|log_return_1d| > 1.0). "
                f"Expected for penny stocks and historical data errors."
            )

    # Warn if Amihud illiquidity contains infinite values
    # (can occur when volume is zero and return is non-zero after NaN replacement)
    if "amihud_illiquidity" in df.columns:
        inf_count = np.isinf(df["amihud_illiquidity"].fillna(0)).sum()
        if inf_count > 0:
            results["warnings"].append(
                f"amihud_illiquidity: {inf_count:,} infinite values. "
                f"These will be clipped or masked in the data loader."
            )

    # Warn if price_to_52w_high contains values outside [0, 1]
    # (can occur for prices above the rolling window maximum due to gaps)
    if "price_to_52w_high" in df.columns:
        out_of_range = (
            (df["price_to_52w_high"] > 1.05) | (df["price_to_52w_high"] < 0)
        ).sum()
        if out_of_range > 0:
            results["warnings"].append(
                f"price_to_52w_high: {out_of_range:,} values outside [0, 1.05]. "
                f"Expected near boundaries due to rolling window edge effects."
            )

    return results


def report_validation_results(results: dict) -> None:
    """Log validation results clearly.

    Args:
        results: Dictionary from validate_data_integrity().
    """
    if results["passed"]:
        logger.info("Data validation PASSED")
    else:
        logger.error("Data validation FAILED")

    if results["errors"]:
        logger.error("Errors:")
        for error in results["errors"]:
            logger.error(f"  - {error}")

    if results["warnings"]:
        logger.warning("Warnings:")
        for warning in results["warnings"]:
            logger.warning(f"  - {warning}")


def optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 to float32 and convert repeated strings to categoricals.

    For financial data, float32 precision (7 decimal digits) is sufficient
    for model training. Categorical encoding stores each unique string once
    and uses integer codes per row, drastically reducing memory for columns
    like Ticker (~20,000 unique values repeated across ~7,000 dates).

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with optimised dtypes.
    """
    df = df.copy()

    mem_before = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"  Memory before optimisation: {mem_before:.1f} MB")

    # Convert low-cardinality string columns to category
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype("category")
            logger.info(f"  Converted {col} to category ({df[col].nunique()} unique)")

    # Downcast only genuine float64 columns to float32 — bool and object columns
    # are excluded because select_dtypes(include=["float64"]) already filters them,
    # but we make the guard explicit to avoid any future dtype confusion.
    float_cols = [
        c for c in df.select_dtypes(include=["float64"]).columns
        if df[c].dtype == np.float64
    ]
    downcast_count = 0
    for col in float_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if pd.notna(min_val) and pd.notna(max_val):
            if -3.4e38 < min_val and max_val < 3.4e38:
                df[col] = df[col].astype("float32")
                downcast_count += 1

    logger.info(f"  Downcast {downcast_count} float64 columns to float32")

    mem_after = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"  Memory after optimisation: {mem_after:.1f} MB")
    logger.info(f"  Reduction: {(1 - mem_after / mem_before) * 100:.1f}%")

    return df


def generate_schema_metadata(df: pd.DataFrame) -> dict:
    """Generate a metadata dictionary describing the dataset schema.

    Computes per-column statistics (dtype, null count, min/max/mean for
    numeric, unique count for categorical). Used for documentation and
    downstream data loader validation.

    Args:
        df: Final DataFrame.

    Returns:
        Dictionary with column types, value ranges, and row counts.
    """
    logger.info("  Computing column-level statistics...")

    metadata: dict = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "date_range": {
            "start": str(df["Date"].min()),
            "end": str(df["Date"].max()),
        },
        "model_feature_cols": MODEL_FEATURE_COLS,
        "columns": {},
    }

    for col in df.columns:
        col_meta: dict = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_pct": round(df[col].isnull().mean() * 100, 2),
            "is_model_feature": col in MODEL_FEATURE_COLS,
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_meta["min"] = (
                float(df[col].min()) if pd.notna(df[col].min()) else None
            )
            col_meta["max"] = (
                float(df[col].max()) if pd.notna(df[col].max()) else None
            )
            col_meta["mean"] = (
                float(df[col].mean()) if pd.notna(df[col].mean()) else None
            )

        elif df[col].dtype.name == "category" or df[col].dtype == object:
            col_meta["unique_count"] = int(df[col].nunique())
            if df[col].nunique() <= 10:
                unique_vals = df[col].dropna().unique().tolist()
                col_meta["unique_values"] = sorted(
                    [str(v) for v in unique_vals]
                )

        metadata["columns"][col] = col_meta

    return metadata


def export_metadata(metadata: dict, output_path: Path) -> None:
    """Export schema metadata to a JSON file.

    Args:
        metadata: Schema metadata dictionary.
        output_path: Path for JSON output.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Metadata exported to: {output_path}")


def export_to_parquet(
    df: pd.DataFrame, output_path: Path, compression: str = "snappy"
) -> None:
    """Export DataFrame to Parquet with the specified compression.

    Snappy provides a good balance of compression ratio and encode/decode
    speed. The Parquet format natively supports dictionary encoding for
    categorical columns, preserving the memory savings from optimise_dtypes.

    All columns are retained. The model reads only MODEL_FEATURE_COLS.

    Args:
        df: DataFrame to export.
        output_path: Path for Parquet file.
        compression: Algorithm ('snappy', 'gzip', 'brotli', or None).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        output_path,
        compression=compression,
        index=False,
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Exported Parquet to: {output_path}")
    logger.info(f"  File size: {file_size_mb:.1f} MB")
    logger.info(f"  Compression: {compression}")
    logger.info(f"  Total columns retained: {len(df.columns)}")
    logger.info(f"  Model feature columns: {len(MODEL_FEATURE_COLS)}")


def export_sample_csv(
    df: pd.DataFrame, output_path: Path, n_rows: int = 1000
) -> None:
    """Export a small CSV sample for quick inspection without Python.

    Samples proportionally across asset classes so each class is
    represented in the output.

    Args:
        df: Full DataFrame.
        output_path: Path for sample CSV.
        n_rows: Number of rows to include.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_classes = df["AssetClass"].nunique()
    rows_per_class = max(1, n_rows // n_classes)

    sample = df.groupby("AssetClass", group_keys=False, observed=True).apply(
        lambda x: x.head(rows_per_class), include_groups=False
    )
    sample = sample.head(n_rows)

    sample.to_csv(output_path, index=False)

    logger.info(f"Sample CSV exported to: {output_path}")
    logger.info(f"  Sample rows: {len(sample):,}")


def generate_final_report(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the final fact table.

    Provides a per-asset-class breakdown of ticker counts, row counts,
    and date ranges for the pipeline completion log.

    Args:
        df: Final fact table DataFrame.

    Returns:
        Dictionary with row counts, date ranges, and per-asset-class breakdowns.
    """
    report: dict = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "model_feature_columns": len(MODEL_FEATURE_COLS),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 1),
        "date_range": (str(df["Date"].min()), str(df["Date"].max())),
        "by_asset_class": {},
    }

    for ac in sorted(df["AssetClass"].unique()):
        subset = df[df["AssetClass"] == ac]
        report["by_asset_class"][str(ac)] = {
            "tickers": int(subset["Ticker"].nunique()),
            "rows": len(subset),
            "date_range": (
                str(subset["Date"].min()),
                str(subset["Date"].max()),
            ),
        }

    return report


def main(
    features_path: Path,
    output_dir: Path,
    validate: bool = True,
    optimise: bool = True,
) -> None:
    """Orchestrate the complete Parquet export pipeline.

    Loads the Stage 3 CSV, validates integrity (including presence of all
    MODEL_FEATURE_COLS), optimises dtypes, then exports Parquet, sample
    CSV, and schema metadata JSON. All columns are retained per Alternative B.

    Args:
        features_path: Path to master_features.csv (from Stage 3).
        output_dir: Directory for output files (data/processed/).
        validate: Whether to run data validation.
        optimise: Whether to optimise data types.
    """
    logger.info("STAGE 4: Export to Parquet")
    logger.info(
        f"Model feature mask ({len(MODEL_FEATURE_COLS)} features): {MODEL_FEATURE_COLS}"
    )

    # Load the feature-engineered CSV.
    # Columns such as Country, Currency, Exchange, Isin, Name, Type are string/metadata
    # but have NaNs or empty values for yields/commodities; with chunked inference pandas
    # reports "mixed types". We force these to string so dtype is consistent and the
    # DtypeWarning is avoided; Stage 4 later converts them to category where appropriate.
    logger.info(f"Loading features from: {features_path}")
    if not features_path.exists():
        logger.error(f"File not found: {features_path}")
        return

    str_columns = [
        "Ticker", "AssetClass", "Country", "Currency", "Exchange", "Frequency",
        "Isin", "Name", "Type",
    ]
    dtype_spec = {c: str for c in str_columns}
    df = pd.read_csv(
        features_path,
        parse_dates=["Date"],
        dtype=dtype_spec,
        low_memory=False,
    )
    logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    logger.info(
        f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
    )

    # Clean structurally required columns
    for col in REQUIRED_COLUMNS:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            logger.warning(f"  Dropping {null_count:,} rows with NaN in {col}")
            df = df.dropna(subset=[col])

    # Drop columns that are entirely null — they add file size with no value
    null_counts = df.isnull().sum()
    fully_null_cols = null_counts[null_counts == len(df)].index.tolist()
    if fully_null_cols:
        df = df.drop(columns=fully_null_cols)
        logger.info(f"  Dropped fully null columns: {fully_null_cols}")

    # Drop unused pure-metadata columns never referenced by the data loader
    unused_metadata = ["Isin", "Name", "Exchange", "Currency", "Type"]
    to_drop = [c for c in unused_metadata if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        logger.info(f"  Dropped unused metadata columns: {to_drop}")

    # Validate data integrity
    if validate:
        logger.info("Validating data integrity...")
        validation_results = validate_data_integrity(df)
        report_validation_results(validation_results)
        if not validation_results["passed"]:
            raise ValueError(
                "Data validation failed. Fix errors before exporting."
            )

    # Optimise dtypes for compression and memory
    if optimise:
        logger.info("Optimising data types")
        df = optimise_dtypes(df)

    # Generate schema metadata
    logger.info("Generating schema metadata")
    metadata = generate_schema_metadata(df)

    # Export Parquet (the primary output — all columns retained)
    parquet_path = output_dir / "fact_table.parquet"
    export_to_parquet(df, parquet_path)

    # Export a small sample CSV for quick human inspection
    sample_path = output_dir / "fact_table_sample.csv"
    export_sample_csv(df, sample_path)

    # Export metadata JSON
    metadata_path = output_dir / "fact_table_metadata.json"
    export_metadata(metadata, metadata_path)

    # Post-export schema validation — non-blocking, log-level only
    if "is_tradable" not in df.columns or "target_fwd" not in df.columns:
        logger.error(
            "Missing 'is_tradable' or 'target_fwd' in exported Parquet. "
            "Re-run Stage 3 with the updated FEATURE_CONFIG."
        )
    else:
        if df["is_tradable"].sum() == 0:
            logger.error(
                "'is_tradable' is entirely False. "
                "Check Stage 3 tradability thresholds."
            )
        if df["target_fwd"].notna().sum() == 0:
            logger.error(
                "'target_fwd' is entirely null. "
                "Check Stage 3 groupby sort order."
            )

    raw_cols = ["log_return_1d_raw", "log_return_5d_raw", "log_return_20d_raw"]
    if not all(c in df.columns for c in raw_cols):
        logger.warning(
            "Winsorisation diagnostic '_raw' columns are missing — "
            "winsorise_returns may not have been applied."
        )

    if "Frequency" in df.columns:
        logger.warning(
            "Column 'Frequency' is still present in the Parquet output. "
            "It is 100%% null and should be dropped to reduce file size."
        )

    # Generate and log the final summary report
    report = generate_final_report(df)
    logger.info("Final Report:")
    logger.info(f"  Total rows: {report['total_rows']:,}")
    logger.info(f"  Total columns (all retained): {report['total_columns']}")
    logger.info(f"  Model feature columns: {report['model_feature_columns']}")
    logger.info(f"  Memory: {report['memory_mb']:.1f} MB")
    logger.info(
        f"  Date range: {report['date_range'][0]} to {report['date_range'][1]}"
    )
    for ac, details in report["by_asset_class"].items():
        logger.info(
            f"  {ac}: {details['tickers']} tickers, {details['rows']:,} rows"
        )

    logger.info("STAGE 4 COMPLETE - Data pipeline finished!")
    logger.info(f"Model-ready data: {parquet_path}")
    logger.info(
        f"Load model features with: df[MODEL_FEATURE_COLS] from stage4_export_parquet"
    )


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent

    FEATURES_FILE = PROJECT_ROOT / "data" / "interim" / "master_features.csv"
    OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

    main(features_path=FEATURES_FILE, output_dir=OUTPUT_DIR)