"""Graph Layer: dynamic correlation networks for cross-asset dependencies.

Captures relationships between stocks using a two-stage GNN: first a
GCN aggregates information weighted by pairwise correlations, then a
GAT learns which neighbours matter most via attention. Residual
connections and DropEdge prevent over-smoothing.

Theoretical Context:
    Financial markets exhibit contagion where shocks spread through correlation
    networks (Sandhu et al., 2021). We model this using a dynamic graph topology.
    A GCN layer (Kipf & Welling, 2016) broadcasts volatility signals across the
    network, while a GAT layer (Velickovic et al., 2017) applies attention to
    selectively focus on the most relevant neighbors during stress events.

References:
    Kipf and Welling (2016) "Semi-Supervised Classification with
        Graph Convolutional Networks." arXiv:1609.02907.
    Velickovic et al. (2017) "Graph Attention Networks."
        arXiv:1710.10903.
    Rong et al. (2019) "DropEdge: Towards Deep Graph Convolutional
        Networks on Node Classification." arXiv:1907.10903.
    Sandhu et al. (2021) "Network Geometry and Market Instability."
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv

logger = logging.getLogger(__name__)


class GraphDependencyLayer(nn.Module):
    """Learns cross-asset dependencies from correlation-based graphs.

    Two-stage message passing: GCN spreads information weighted by
    how correlated stocks are, then GAT learns to focus on the most
    relevant neighbours. A residual connection preserves the original
    signal and DropEdge randomly removes edges during training to
    prevent all node representations from collapsing together.

    Ref: Kipf and Welling (2016) Section 2; Velickovic et al. (2017) Section 2.1; Rong et al. (2019) Section 3.1.

    Args:
        in_channels: Dimension of input node features.
        hidden_channels: Internal GCN output dimension.
        out_channels: Final output dimension. Must be divisible by heads.
        heads: Number of GAT attention heads.
        dropout: Dropout rate for features between GCN and GAT.
        edge_drop: Fraction of edges randomly dropped during training.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.1,
        edge_drop: float = 0.1,
    ) -> None:
        super().__init__()

        if out_channels % heads != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by "
                f"heads ({heads}) for multi-head attention concatenation."
            )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_drop = edge_drop

        # Three GCN layers for multi-hop message passing (3-hop contagion)
        self.gcn = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.gcn3 = GCNConv(hidden_channels, hidden_channels)

        # GAT learns which neighbours are most informative via attention.
        # Each head operates on (out_channels / heads) dimensions,
        # then heads are concatenated back to out_channels.
        self.gat = GATv2Conv(
            hidden_channels,
            out_channels // heads,
            heads=heads,
            concat=True,
            edge_dim=1,
        )

        # Residual: if input and output dimensions differ, project to match
        self.residual: nn.Module = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

        logger.info(
            "GraphDependencyLayer initialised: in=%d, hidden=%d, out=%d, "
            "heads=%d, dropout=%.2f, edge_drop=%.2f",
            in_channels,
            hidden_channels,
            out_channels,
            heads,
            dropout,
            edge_drop,
        )

    def _drop_edges(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Randomly removes edges during training for regularisation.

        Acts as structural dropout: each edge is independently kept
        with probability (1 - edge_drop). This prevents the GNN from
        over-relying on specific connections.

        Ref: Rong et al. (2019), Section 3.1 ("DropEdge").

        Args:
            edge_index: Graph connectivity, shape (2, num_edges).
            edge_weight: Edge weights, shape (num_edges,) or None.

        Returns:
            Filtered edge_index and edge_weight.
        """
        if self.edge_drop <= 0.0:
            return edge_index, edge_weight

        num_edges = edge_index.size(1)

        # Intent: Structural Regularization (Rong et al., 2019).
        # By randomly removing edges (DropEdge), we force the model to learn
        # redundant paths for contagion. This prevents "Over-smoothing,"
        # where all stock embeddings collapse to the same value in deep graphs.
        # Keep each edge independently with probability (1 - edge_drop)
        keep_mask = torch.rand(num_edges, device=edge_index.device) >= self.edge_drop

        edge_index = edge_index[:, keep_mask]

        if edge_weight is not None:
            edge_weight = edge_weight[keep_mask]

        return edge_index, edge_weight

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Passes information across the asset graph.

        Applies GCN (correlation-weighted aggregation), then GAT
        (attention-based aggregation), with a residual skip connection.
        
        Ref: Kipf & Welling (2016) Eq (9) and Appendix B Eq (14) for residual.

        Args:
            x: Node features, shape [num_nodes, in_channels].
            edge_index: Graph connectivity, shape [2, num_edges].
            edge_weight: Correlation-based edge weights ``(num_edges,)``.
                If ``None``, each edge is treated as weight 1.0 (same dtype and
                device as ``x``) so GCN and GAT receive a consistent tensor.

        Returns:
            Updated node embeddings, shape [num_nodes, out_channels].
        """
        num_edges = edge_index.size(1)
        if edge_weight is None:
            # Standard unweighted convention; avoids ``unsqueeze`` on ``None``
            # in the GAT path and keeps DropEdge masking aligned with edges.
            edge_weight = torch.ones(num_edges, dtype=x.dtype, device=x.device)

        # Drop random edges during training to regularise
        if self.training:
            edge_index, edge_weight = self._drop_edges(edge_index, edge_weight)

        # Intent: Broadcasting Contagion (Kipf & Welling, 2016).
        # We use a GCN to spread volatility signals across the correlation network.
        # The edge weights (correlations) determine how much impact one stock
        # has on its neighbors.
        # Stage 1: Three-hop GCN spreads information across the correlation network
        h = self.gcn(x, edge_index, edge_weight)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gcn2(h, edge_index, edge_weight)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gcn3(h, edge_index, edge_weight)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Intent: Selective Attention (Velickovic et al., 2017).
        # Not all correlations matter equally at all times. The GAT layer
        # learns dynamic attention weights, allowing the model to focus on
        # specific neighbors (e.g., Tech stocks) during specific market events.
        # Stage 2: GAT learns which neighbours matter most
        h = self.gat(h, edge_index, edge_attr=edge_weight.unsqueeze(-1))
        h = F.elu(h)

        # Intent: Information Preservation (Residual Connection).
        # We add the original input features back to the graph output.
        # This "Gradient Highway" ensures that raw financial signals (returns, etc.)
        # aren't lost even if the graph structure is noisy.
        # Add residual to preserve original signal
        h = h + self.residual(x)

        return h
