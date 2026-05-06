"""Graph construction utilities for dynamic correlation networks.

Builds correlation-based graphs from return data: computes pairwise
correlations over a trailing window, applies a threshold, and converts
to PyTorch Geometric COO format with self-loops.

References:
    Sandhu et al. (2021) "Network Geometry and Market Instability."
        Nature Physics.
    Garcia-Mondragon et al. (2021) "The Hyperbolic Geometry of
    Financial Networks." Nature Scientific Reports.
"""

import logging

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def build_correlation_graph(
    returns: pd.DataFrame,
    threshold: float = 0.3,
    min_periods: int = 60,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds a graph where edges connect correlated stocks.

    Computes pairwise Pearson correlations from a return matrix,
    keeps only pairs above the threshold, and adds self-loops so
    each stock retains its own information during message passing.

    Ref: Sandhu et al. (2021), correlation-based financial networks.

    Args:
        returns: DataFrame of shape (T, N) with log returns per ticker.
        threshold: Minimum |correlation| to create an edge.
        min_periods: Minimum overlapping observations for a valid
            correlation. Pairs with fewer produce NaN, replaced by 0.

    Returns:
        Tuple of:
            - edge_index: Long tensor, shape (2, num_edges), COO format.
            - edge_weight: Float tensor, shape (num_edges,).
    """
    num_tickers = returns.shape[1]

    # Compute pairwise correlations over the trailing window
    corr_matrix = returns.corr(min_periods=min_periods)
    corr_matrix = corr_matrix.fillna(0.0)

    # Threshold on absolute correlation, excluding diagonal
    corr_np: np.ndarray = corr_matrix.values.astype(np.float32)
    abs_corr: np.ndarray = np.abs(corr_np)

    mask = (abs_corr > threshold) & ~np.eye(num_tickers, dtype=bool)
    rows, cols = np.where(mask)
    weights = abs_corr[rows, cols]

    # Add self-loops (weight 1.0) so each node keeps its own signal
    self_indices = np.arange(num_tickers, dtype=np.intp)
    rows = np.concatenate([rows, self_indices])
    cols = np.concatenate([cols, self_indices])
    weights = np.concatenate([weights, np.ones(num_tickers, dtype=np.float32)])

    edge_index = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    return edge_index, edge_weight


def build_dynamic_graph_gpu(
    returns: torch.Tensor,
    threshold: float = 0.3,
    max_neighbours: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU-accelerated dynamic graph construction.

    Args:
        returns: Tensor of shape (N, T) containing past returns for N nodes.
        threshold: Correlation threshold.
        max_neighbours: Maximum retained neighbours per node after
            thresholding. Nodes with fewer surviving neighbours keep all
            available edges.

    Returns:
        (edge_index, edge_weight) on the same device as inputs.
    """
    N, T = returns.shape
    
    # 1. Compute Correlation Matrix (N, N)
    # torch.corrcoef computes correlation of rows (variables)
    # Stability: add small noise or handle constant rows? 
    # nan_to_num handles cases where std is 0.
    corr = torch.corrcoef(returns)
    corr = torch.nan_to_num(corr, 0.0)
    
    # 2. Threshold weak correlations, then cap each node's degree so crisis
    # periods cannot explode graph density.
    abs_corr = corr.abs()
    thresholded = torch.where(
        abs_corr > threshold,
        abs_corr,
        torch.zeros_like(abs_corr),
    )
    thresholded.fill_diagonal_(0.0)

    if max_neighbours is None or max_neighbours <= 0:
        mask = thresholded > 0
        edge_index = mask.nonzero().t()
        edge_weight = thresholded[edge_index[0], edge_index[1]]
    else:
        k = min(int(max_neighbours), max(N - 1, 1))
        topk_values, topk_indices = torch.topk(thresholded, k=k, dim=1)
        valid_mask = topk_values > 0
        row_indices = torch.arange(N, device=returns.device).unsqueeze(1).expand_as(
            topk_indices
        )
        edge_index = torch.stack(
            [row_indices[valid_mask], topk_indices[valid_mask]],
            dim=0,
        )
        edge_weight = topk_values[valid_mask]
    
    # 4. Add Self-Loops
    loop_index = torch.arange(N, device=returns.device)
    loop_index = torch.stack([loop_index, loop_index], dim=0)
    loop_weight = torch.ones(N, device=returns.device)
    
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
    
    return edge_index, edge_weight


def build_batched_graphs_gpu(
    returns: torch.Tensor,
    batch: torch.Tensor,
    threshold: float = 0.3,
    max_neighbours: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds per-graph correlation edges for an entire PyG batch.

    Partitions the concatenated return matrix by the batch vector,
    calls build_dynamic_graph_gpu for each sub-graph, then merges
    the edge lists back into block-diagonal format compatible with
    PyG message passing.

    Args:
        returns: Trailing returns (total_nodes, T) concatenated
            across all graphs in the batch.
        batch: Graph membership vector (total_nodes,) mapping each
            node to its originating graph index.
        threshold: Minimum |correlation| to retain an edge.
        max_neighbours: Maximum retained neighbours per node after
            thresholding.

    Returns:
        (edge_index, edge_weight) in block-diagonal COO format.
    """
    edge_indices: list[torch.Tensor] = []
    edge_weights: list[torch.Tensor] = []
    offset = 0

    for g in batch.unique():
        mask = batch == g
        n_g = int(mask.sum().item())
        pr = returns[mask]
        ei, ew = build_dynamic_graph_gpu(
            pr,
            threshold=threshold,
            max_neighbours=max_neighbours,
        )
        edge_indices.append(ei + offset)
        edge_weights.append(ew)
        offset += n_g

    return torch.cat(edge_indices, dim=1), torch.cat(edge_weights)
