"""Diagnostic visualisations for the full model.

Generates figures for the evaluation report: regime entropy over
time, mean geodesic distance between embeddings, and attention
heatmap placeholders (future work). Figures are saved to the
configured experiments/figures/ directory.

Theoretical Context:
    To interpret the "Black Box," we extract internal geometric and quantum states:
    1. Regime Entropy (Haven & Khrennikov, 2013): Measures the model's
       uncertainty about the market state (High = Mixed/Confused, Low = Pure/Confident).
    2. Geodesic Clustering (Bronstein et al., 2021): Measures the "curvature"
       of the market. When stocks cluster tightly on the manifold, it indicates
       high systemic correlation (Fragility).

References:
    Haven and Khrennikov (2013) "Quantum Social Science."
    Bronstein et al. (2021) "Geometric Deep Learning."
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# matplotlib is optional; plots are skipped if not installed.
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for server/CI use
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    logger.warning("matplotlib not available; diagnostic plots will be skipped.")

from src.layers.quantum import von_neumann_entropy


@torch.no_grad()
def collect_regime_entropy_series(
    model: torch.nn.Module,
    dataset: Any,
    device: torch.device,
) -> dict[str, list]:
    """Collects regime entropy for each date in the dataset.

    Runs the model forward on each cross-section, extracts the
    density matrix, and computes the von Neumann entropy. High
    entropy means the model is uncertain about which regime the
    market is in; low entropy means strong conviction.

    Args:
        model: Trained model instance.
        dataset: FinancialGraphDataset with valid dates.
        device: Compute device.

    Returns:
        Dictionary with 'dates' and 'entropy' lists.
    """
    model.eval()
    dates: list[str] = []
    entropies: list[float] = []

    for idx in range(len(dataset)):
        data = dataset[idx]

        if data.target_mask.sum() == 0:
            continue

        data = data.to(device)
        batch_vec = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        output = model(
            x=data.x, edge_index=data.edge_index,
            edge_weight=data.edge_attr, context=data.context,
            batch=batch_vec,
        )

        rho = output.get("density_matrix")
        if rho is None:
            continue

        # Intent: Uncertainty Quantification (Haven & Khrennikov).
        # We don't just want a prediction; we want to know if the model is confident.
        # The Von Neumann Entropy of the density matrix gives us a rigorous,
        # physically-grounded measure of "Systemic Uncertainty."
        entropy = von_neumann_entropy(rho).mean().item()
        entropies.append(entropy)

        # Extract date string from the dataset
        date_str = str(dataset.valid_dates[idx].date())
        dates.append(date_str)

    logger.info(
        "Collected regime entropy for %d dates.",
        len(entropies),
    )
    return {"dates": dates, "entropy": entropies}


@torch.no_grad()
def collect_embedding_geodesics(
    model: torch.nn.Module,
    dataset: Any,
    device: torch.device,
    n_dates: int | None = None,
) -> dict[str, list]:
    """Measures how spread out embeddings are on the sphere per date.

    Computes the average geodesic distance between all stock
    embeddings for each cross-section. Tight clustering during
    crises suggests the model captures correlation convergence.

    Args:
        model: Trained model instance.
        dataset: FinancialGraphDataset with valid dates.
        device: Compute device.
        n_dates: Maximum dates to process (None = all).

    Returns:
        Dictionary with 'dates' and 'mean_geodesic_distance'.
    """
    model.eval()
    dates: list[str] = []
    distances: list[float] = []
    eps = 1e-7

    total = len(dataset)
    if n_dates is not None:
        total = min(total, n_dates)

    for idx in range(total):
        data = dataset[idx]

        if data.target_mask.sum() == 0:
            continue

        data = data.to(device)
        batch_vec = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        output = model(
            x=data.x, edge_index=data.edge_index,
            edge_weight=data.edge_attr, context=data.context,
            batch=batch_vec,
        )
        embeddings = output.get("embeddings")

        if embeddings is None:
            continue

        # Embeddings are (N, E) in flat format
        emb = embeddings.cpu()
        n = emb.shape[0]

        if n < 2:
            continue

        # Pairwise geodesic distances on the sphere
        dots = emb @ emb.T
        dots = dots.clamp(-1.0 + eps, 1.0 - eps)
        geo_dists = torch.acos(dots)

        # Intent: Measuring Systemic Fragility.
        # When the mean geodesic distance drops, stocks are "clumping" together
        # on the manifold. This geometric convergence often precedes
        # market crashes (Contagion).
        # Mean off-diagonal distance
        mask = ~torch.eye(n, dtype=torch.bool)
        mean_dist = float(geo_dists[mask].mean().item())
        distances.append(mean_dist)

        date_str = str(dataset.valid_dates[idx].date())
        dates.append(date_str)

    logger.info(
        "Collected mean geodesic distances for %d dates.",
        len(distances),
    )
    return {"dates": dates, "mean_geodesic_distance": distances}


def plot_regime_entropy(
    entropy_data: dict[str, list],
    figures_dir: Path,
    title: str = "Von Neumann Entropy of Market Regime Density Matrix",
) -> Path | None:
    """Plots regime entropy S(rho_t) over time.

    Args:
        entropy_data: Output of collect_regime_entropy_series().
        figures_dir: Directory to save figures.
        title: Plot title.

    Returns:
        Path to the saved figure, or None if matplotlib is unavailable.
    """
    if not _MPL_AVAILABLE:
        logger.warning("Skipping entropy plot: matplotlib unavailable.")
        return None

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "regime_entropy_over_time.png"

    dates = entropy_data["dates"]
    entropies = entropy_data["entropy"]

    if not dates:
        logger.warning("No entropy data to plot.")
        return None

    fig, ax = plt.subplots(figsize=(14, 5))

    # Parse dates for proper axis formatting
    try:
        import pandas as pd

        date_objects = pd.to_datetime(dates)
        ax.plot(date_objects, entropies, linewidth=0.8, color="#2C3E50")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate()
    except Exception:
        ax.plot(range(len(entropies)), entropies, linewidth=0.8)
        ax.set_xlabel("Time step")

    ax.set_ylabel(r"$S(\rho_t)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add theoretical bounds
    k = 4  # default num_regimes from config
    ax.axhline(
        y=0.0,
        color="green",
        linestyle="--",
        alpha=0.5,
        label="Pure state (S=0)",
    )
    ax.axhline(
        y=np.log(k),
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Maximally mixed (S=log({k})={np.log(k):.2f})",
    )
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Regime entropy plot saved to %s.", fig_path)
    return fig_path


def plot_embedding_geodesic_distance(
    geodesic_data: dict[str, list],
    figures_dir: Path,
    title: str = "Mean Geodesic Distance Between Embeddings Over Time",
) -> Path | None:
    """Plots mean geodesic distance of embeddings over time.

    Tighter clustering (lower distances) during crises indicates
    correlation convergence captured by the geometric layer.

    Args:
        geodesic_data: Output of collect_embedding_geodesics().
        figures_dir: Directory to save figures.
        title: Plot title.

    Returns:
        Path to the saved figure, or None if matplotlib is unavailable.
    """
    if not _MPL_AVAILABLE:
        logger.warning("Skipping geodesic plot: matplotlib unavailable.")
        return None

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "embedding_geodesic_distance.png"

    dates = geodesic_data["dates"]
    distances = geodesic_data["mean_geodesic_distance"]

    if not dates:
        logger.warning("No geodesic data to plot.")
        return None

    fig, ax = plt.subplots(figsize=(14, 5))

    try:
        import pandas as pd

        date_objects = pd.to_datetime(dates)
        ax.plot(date_objects, distances, linewidth=0.8, color="#8E44AD")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate()
    except Exception:
        ax.plot(range(len(distances)), distances, linewidth=0.8)
        ax.set_xlabel("Time step")

    ax.set_ylabel("Mean geodesic distance (radians)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Geodesic distance plot saved to %s.", fig_path)
    return fig_path


def plot_ablation_comparison(
    summary_csv_path: Path,
    figures_dir: Path,
    metrics: list[str] | None = None,
) -> Path | None:
    """Generates a bar chart comparing ablation configurations.

    Reads the ablation summary CSV and produces a grouped bar chart
    of mean metric values with standard deviation error bars.

    Args:
        summary_csv_path: Path to ablation_summary.csv.
        figures_dir: Directory to save figures.
        metrics: Metric names to plot (default: key subset).

    Returns:
        Path to the saved figure, or None if unavailable.
    """
    if not _MPL_AVAILABLE:
        logger.warning("Skipping ablation chart: matplotlib unavailable.")
        return None

    if not summary_csv_path.exists():
        logger.warning(
            "Summary CSV not found at %s; skipping chart.",
            summary_csv_path,
        )
        return None

    import pandas as pd

    df = pd.read_csv(summary_csv_path)

    if metrics is None:
        metrics = ["ic", "directional_accuracy", "sharpe_daily", "sharpe_annual"]

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "ablation_comparison.png"

    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=(4 * n_metrics, 5),
        squeeze=False,
    )

    config_ids = df["config_id"].tolist()
    cmap = matplotlib.colormaps.get_cmap("Set3")
    colours = cmap(np.linspace(0, 1, len(config_ids)))

    for col_idx, metric in enumerate(metrics):
        ax = axes[0, col_idx]
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        if mean_col not in df.columns:
            continue

        means = df[mean_col].values
        stds = df[std_col].values if std_col in df.columns else np.zeros_like(means)

        ax.bar(
            range(len(config_ids)),
            means,
            yerr=stds,
            capsize=3,
            color=colours,
            edgecolor="grey",
            linewidth=0.5,
        )
        ax.set_xticks(range(len(config_ids)))
        ax.set_xticklabels(config_ids, rotation=45, fontsize=8)
        ax.set_title(metric.replace("_", " ").title(), fontsize=10)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        "Ablation Study: Mean Metric Comparison",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Ablation comparison chart saved to %s.", fig_path)
    return fig_path


def collect_cumulative_returns(
    model: torch.nn.Module,
    dataset: Any,
    device: torch.device,
) -> dict[str, list]:
    """Collects cumulative returns (equity curves) for the model vs market.

    Simulates a simple trading strategy:
    1. Long positions where predicted return > 0.
    2. Short positions where predicted return < 0.
    3. Equal weight across active positions.

    Also tracks "Market" (Buy and Hold) performance for comparison.

    Args:
        model: Trained model instance.
        dataset: FinancialGraphDataset.
        device: Compute device.

    Returns:
        Dictionary with 'dates', 'strategy_cum_return', 'market_cum_return'.
    """
    model.eval()
    dates: list[str] = []
    strategy_returns: list[float] = []
    market_returns: list[float] = []

    for idx in range(len(dataset)):
        data = dataset[idx]
        if data.target_mask.sum() == 0:
            continue

        data = data.to(device)
        batch_vec = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        x_seq = getattr(data, "x_seq", None)

        output = model(
            x=data.x,
            edge_index=data.edge_index,
            edge_weight=data.edge_attr,
            context=data.context,
            batch=batch_vec,
            x_seq=x_seq,
        )
        preds = output["predictions"].squeeze(-1)  # (N,)

        # Filter valid targets
        valid_preds = preds[data.target_mask]
        valid_targets = data.y[data.target_mask]

        if len(valid_preds) == 0:
            continue

        # Strategy: Sign-based (Long/Short)
        # Position = sign(prediction)
        # Return = position * target
        positions = torch.sign(valid_preds)
        strat_ret = (positions * valid_targets).mean().item()

        # Market: Buy and Hold (Mean of all valid targets)
        mkt_ret = valid_targets.mean().item()

        strategy_returns.append(strat_ret)
        market_returns.append(mkt_ret)
        dates.append(str(dataset.valid_dates[idx].date()))

    # Compute cumulative returns
    # cum_ret_t = (1 + r_1) * (1 + r_2) ... * (1 + r_t) - 1
    # Use log returns for summation: sum(log(1+r))?
    # Dataset provides log returns. Sum of log returns = Cumulative Log Return.
    # Cumulative Simple Return = exp(Sum of Log Returns) - 1.

    strat_cum = np.exp(np.cumsum(strategy_returns)) - 1.0
    mkt_cum = np.exp(np.cumsum(market_returns)) - 1.0

    return {
        "dates": dates,
        "strategy_cum_return": strat_cum.tolist(),
        "market_cum_return": mkt_cum.tolist(),
    }


def plot_cumulative_returns(
    returns_data: dict[str, list],
    figures_dir: Path,
    title: str = "Cumulative Returns: Model Strategy vs Market",
) -> Path | None:
    """Plots equity curves for the model strategy vs market benchmark.

    Args:
        returns_data: Output of collect_cumulative_returns().
        figures_dir: Directory to save figures.
        title: Plot title.

    Returns:
        Path to saved figure or None.
    """
    if not _MPL_AVAILABLE:
        logger.warning("Skipping equity curve plot: matplotlib unavailable.")
        return None

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "cumulative_returns.png"

    dates = returns_data["dates"]
    strat = returns_data["strategy_cum_return"]
    mkt = returns_data["market_cum_return"]

    if not dates:
        logger.warning("No return data to plot.")
        return None

    fig, ax = plt.subplots(figsize=(14, 6))

    try:
        import pandas as pd

        date_objects = pd.to_datetime(dates)
        ax.plot(
            date_objects,
            strat,
            label="Model Strategy (L/S)",
            color="#27AE60",
            linewidth=1.5,
        )
        ax.plot(
            date_objects,
            mkt,
            label="Market (Buy & Hold)",
            color="#7F8C8D",
            linestyle="--",
            linewidth=1.2,
        )
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate()
    except Exception:
        ax.plot(range(len(strat)), strat, label="Model Strategy")
        ax.plot(range(len(mkt)), mkt, label="Market", linestyle="--")
        ax.set_xlabel("Time step")

    ax.set_ylabel("Cumulative Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add y=0 line
    ax.axhline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Cumulative returns plot saved to %s.", fig_path)
    return fig_path


def generate_all_diagnostics(
    model: torch.nn.Module | None,
    dataset: Any | None,
    device: torch.device | None,
    experiments_dir: Path,
    figures_dir: Path,
) -> list[Path]:
    """Generates all available diagnostic visualisations.

    If a trained A8 model and dataset are provided, collects entropy
    and embedding data and generates time-series plots.  Always
    attempts to generate the ablation comparison bar chart from saved
    CSV results.

    Args:
        model: Trained A8 model (or None to skip model-based plots).
        dataset: Validation dataset (or None).
        device: Compute device (or None).
        experiments_dir: Directory containing ablation CSV results.
        figures_dir: Directory to save figures.

    Returns:
        List of paths to saved figure files.
    """
    saved_figures: list[Path] = []

    # Model-based diagnostics (entropy, geodesic distance)
    if model is not None and dataset is not None and device is not None:
        entropy_data = collect_regime_entropy_series(model, dataset, device)
        fig = plot_regime_entropy(entropy_data, figures_dir)
        if fig is not None:
            saved_figures.append(fig)

        geodesic_data = collect_embedding_geodesics(model, dataset, device)
        fig = plot_embedding_geodesic_distance(geodesic_data, figures_dir)
        if fig is not None:
            saved_figures.append(fig)

    # Ablation comparison chart (from CSV)
    summary_csv = experiments_dir / "ablation_summary.csv"
    fig = plot_ablation_comparison(summary_csv, figures_dir)
    if fig is not None:
        saved_figures.append(fig)

    # Cumulative returns equity curve (A8 model required)
    if model is not None and dataset is not None and device is not None:
        returns_data = collect_cumulative_returns(model, dataset, device)
        fig = plot_cumulative_returns(returns_data, figures_dir)
        if fig is not None:
            saved_figures.append(fig)

    if not saved_figures:
        logger.warning("No diagnostic figures were generated.")

    # Document attention heatmaps as future work
    logger.info(
        "GAT attention heatmaps and sector-level embedding geometry "
        "are documented as future work. "
        "The current GAT implementation does not expose per-edge "
        "attention weights in the forward output; extracting them "
        "requires hook-based instrumentation."
    )

    return saved_figures
