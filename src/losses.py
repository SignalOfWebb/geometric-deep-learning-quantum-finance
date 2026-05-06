"""Loss functions for the GDL-Quantum-Finance model.

Combines prediction accuracy (MSE), manifold regularisation (keeping
embeddings spread on the sphere), regime entropy regularisation, and a
directional accuracy reward into a single training objective.

Theoretical Context:
    Financial prediction on manifolds requires a composite loss function.
    We use MSE for magnitude accuracy, but add a Uniformity Loss (Wang & Isola, 2020)
    to prevent "feature collapse" where all stocks map to the same point on the hypersphere.
    A Directional Loss rewards correct sign prediction (critical for trading strategies),
    while regime entropy regularisation discourages regime collapse in the
    quantum layer.

References:
    Wang and Isola (2020) "Understanding Contrastive Representation
        Learning through Alignment and Uniformity on the Hypersphere."
        ICML 2020. Section 3.1 ("Feature Distribution on the Hypersphere").
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Standard numerical stability constant (same as the default eps in
# PyTorch optimisers like Adam). Prevents division-by-zero errors.
_EPS: float = 1e-8


def prediction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Standard MSE between predicted and actual log returns.

    This is the primary training signal.

    Args:
        predictions: Predicted log returns, shape (batch, num_nodes, 1)
            or (batch, num_nodes).
        targets: Actual forward log returns, matching shape.

    Returns:
        Scalar MSE loss.
    """
    predictions = predictions.squeeze(-1)
    targets = targets.squeeze(-1)
    return F.mse_loss(predictions, targets)


def uniformity_loss(
    embeddings: torch.Tensor,
    eps: float = _EPS,
    max_points: int | None = None,
) -> torch.Tensor:
    """Encourages embeddings to spread evenly across the hypersphere.

    Computes a Gaussian kernel over all pairwise geodesic distances.
    Minimising this pushes points apart, preventing the model from
    collapsing all stocks into the same region of the sphere.

    Accepts either batched (B, N, E) or flat (total_nodes, E) format.
    In flat format all nodes across the mini-batch are compared,
    which approximates per-graph uniformity with lower overhead.

    Ref: Wang and Isola (2020), Section 3.1 ("Uniformity").

    Args:
        embeddings: Points on the unit sphere, shape
            (batch, num_nodes, embedding_dim) or (total_nodes, embedding_dim).
        eps: Stability constant for arccos clamping.
        max_points: Optional cap on points used for pairwise computation.
            If provided and the input has more rows, a random subset is
            sampled to keep memory bounded.

    Returns:
        Scalar uniformity loss (lower is more uniform).
    """
    if embeddings.dim() == 3:
        # Legacy batched format
        batch_losses = []
        for b in range(embeddings.shape[0]):
            batch_losses.append(
                _pairwise_uniformity(
                    _sample_points(embeddings[b], max_points),
                    eps,
                )
            )
        return torch.stack(batch_losses).mean()

    # Flat format (total_nodes, E) — compute over all nodes at once
    sampled = _sample_points(embeddings, max_points)
    return _pairwise_uniformity(sampled, eps)


def _sample_points(points: torch.Tensor, max_points: int | None) -> torch.Tensor:
    """Subsamples points to keep uniformity loss memory-bounded.

    Args:
        points: Input points, shape (N, embedding_dim).
        max_points: Maximum number of points to keep.

    Returns:
        Original points if no cap is needed; otherwise a random subset.
    """
    n_points = points.shape[0]
    if max_points is None or max_points <= 0 or n_points <= max_points:
        return points

    indices = torch.randperm(n_points, device=points.device)[:max_points]
    return points[indices]


def _pairwise_uniformity(
    points: torch.Tensor,
    eps: float = _EPS,
) -> torch.Tensor:
    """Computes uniformity loss for a single cross-section.

    Args:
        points: Points on S^{n-1}, shape (N, embedding_dim).
        eps: Stability constant.

    Returns:
        Scalar uniformity loss for this cross-section.
    """
    n = points.shape[0]
    if n < 2:
        return torch.tensor(0.0, dtype=points.dtype, device=points.device)

    # Pairwise cosine similarities (all dot products for unit vectors)
    dot_products = points @ points.T

    # Clamp before arccos to avoid NaN from floating-point overshoot
    dot_products = dot_products.clamp(-1.0 + 1e-6, 1.0 - 1e-6)

    # Convert to geodesic distances on the sphere
    distances = torch.acos(dot_products)

    # Intent: Prevent Feature Collapse (Wang & Isola, 2020).
    # On a hypersphere, neural networks tend to bunch all points together
    # to minimise distance (collapsing the geometry). This term forces
    # them apart, ensuring the embedding space is fully utilised.
    # Gaussian kernel penalises nearby points
    kernel_values = torch.exp(-2.0 * distances.pow(2))

    # Exclude self-distances (always zero, giving exp(0) = 1)
    mask = ~torch.eye(n, dtype=torch.bool, device=points.device)
    off_diagonal = kernel_values[mask]

    return torch.log(off_diagonal.mean())


def regime_entropy_loss(
    regime_probs: torch.Tensor,
    rho: torch.Tensor | None = None,
    eps: float = _EPS,
) -> torch.Tensor:
    """Discourages the quantum layer from collapsing onto one regime.

    Computes the negative Shannon entropy of the regime distribution,
    plus a Von Neumann entropy term on the density matrix (if provided)
    to further penalise regime collapse at the quantum state level.

    Args:
        regime_probs: Regime probabilities, shape (batch, num_regimes).
        rho: Optional density matrix, shape (batch, K, K). If provided,
            the Von Neumann entropy of rho is added as a secondary term.
        eps: Stability constant to avoid log(0).

    Returns:
        Scalar entropy regularisation loss.
    """
    # Shannon entropy of regime probabilities
    entropy = (regime_probs * torch.log(regime_probs + eps)).sum(dim=-1)
    loss = -entropy.mean()

    # Von Neumann entropy of density matrix (if available)
    if rho is not None:
        # Cast to float32 for eigvalsh (not supported for FP16 on CUDA)
        rho_fp32 = rho.float()
        eigenvalues = torch.linalg.eigvalsh(rho_fp32)
        eigenvalues = eigenvalues.clamp(min=eps)
        vn_entropy = -(eigenvalues * torch.log(eigenvalues)).sum(dim=-1)
        loss = loss - vn_entropy.mean()

    return loss


def directional_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Rewards predictions that get the direction (up/down) right.

    In finance, predicting the correct sign of returns is often more
    valuable than predicting exact magnitude. This loss gives credit
    when prediction and actual return agree in sign.

    Args:
        predictions: Predicted log returns.
        targets: Actual forward log returns.

    Returns:
        Scalar directional loss (more negative = better alignment).
    """
    predictions = predictions.squeeze(-1)
    targets = targets.squeeze(-1)

    # Intent: Financial Utility.
    # A low MSE doesn't matter if you predict +0.1% when the market drops -0.1%.
    # This term explicitly penalises "wrong sign" predictions, aligning the
    # model with trading profitability.
    return -torch.mean(torch.sign(targets) * predictions)


class GDLQuantumLoss(nn.Module):
    """Combined multi-objective loss for the full model.

    Sums weighted components: prediction MSE, manifold uniformity,
    regime entropy regularisation, and directional accuracy.

    Args:
        w_pred: Weight for prediction MSE.
        w_uniform: Weight for manifold uniformity.
        w_dm: Weight for regime entropy regularisation.
        w_dir: Weight for directional accuracy.
    """

    def __init__(
        self,
        w_pred: float = 1.0,
        w_uniform: float = 0.01,
        w_dm: float = 0.001,
        w_dir: float = 0.1,
        uniformity_max_points: int | None = None,
    ) -> None:
        super().__init__()
        self.w_pred = w_pred
        self.w_uniform = w_uniform
        self.w_dm = w_dm
        self.w_dir = w_dir
        self.uniformity_max_points = uniformity_max_points

        logger.info(
            "GDLQuantumLoss initialised: w_pred=%.3f, w_uniform=%.3f, "
            "w_dm=%.4f, w_dir=%.2f, uniformity_max_points=%s.",
            w_pred,
            w_uniform,
            w_dm,
            w_dir,
            str(uniformity_max_points),
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        embeddings: torch.Tensor,
        rho: torch.Tensor,
        regime_probs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Computes the weighted combined loss.

        Only evaluates components whose weight is non-zero so that
        dummy tensors from baseline models (e.g. all-zero embeddings)
        cannot inject NaN via ``0.0 * NaN``.

        Args:
            predictions: Predicted log returns.
            targets: Actual forward log returns.
            embeddings: Geometric embeddings on the hypersphere.
            rho: Density matrix from the quantum layer, kept for API compatibility.
            regime_probs: Regime probabilities from the quantum layer.

        Returns:
            Dictionary with 'total', 'prediction', 'uniformity',
            'density_matrix', and 'directional' loss components.
        """
        device = predictions.device
        zero = torch.tensor(0.0, device=device)

        # Prediction MSE is always computed
        loss_pred = prediction_loss(predictions, targets)
        total = self.w_pred * loss_pred

        # Uniformity: skip for baselines where w_uniform == 0
        loss_uniform = zero
        if self.w_uniform > 0:
            loss_uniform = uniformity_loss(
                embeddings,
                max_points=self.uniformity_max_points,
            )
            total = total + self.w_uniform * loss_uniform

        # Regime entropy regularisation: skip when disabled
        loss_dm = zero
        if self.w_dm > 0:
            loss_dm = regime_entropy_loss(regime_probs, rho=rho)
            total = total + self.w_dm * loss_dm

        # Directional accuracy reward
        loss_dir = zero
        if self.w_dir > 0:
            loss_dir = directional_loss(predictions, targets)
            total = total + self.w_dir * loss_dir

        return {
            "total": total,
            "prediction": loss_pred,
            "uniformity": loss_uniform,
            "density_matrix": loss_dm,
            "directional": loss_dir,
        }
