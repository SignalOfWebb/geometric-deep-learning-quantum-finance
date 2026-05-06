"""Unit tests for the loss functions module.

Tests the individual loss components (prediction, uniformity, regime
entropy, directional) and the combined GDLQuantumLoss.
"""

import torch
import torch.nn.functional as F

from src.losses import (
    GDLQuantumLoss,
    directional_loss,
    prediction_loss,
    regime_entropy_loss,
    uniformity_loss,
)


# Shared test dimensions.
_BATCH_SIZE = 4
_NUM_NODES = 20
_EMBEDDING_DIM = 32
_NUM_REGIMES = 4


def _make_predictions_and_targets(
    batch_size: int = _BATCH_SIZE,
    num_nodes: int = _NUM_NODES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Creates random predictions and target tensors."""
    predictions = torch.randn(batch_size, num_nodes, 1)
    targets = torch.randn(batch_size, num_nodes)
    return predictions, targets


def _make_embeddings(
    batch_size: int = _BATCH_SIZE,
    num_nodes: int = _NUM_NODES,
    embedding_dim: int = _EMBEDDING_DIM,
) -> torch.Tensor:
    """Creates random unit-norm embeddings on S^{n-1}."""
    raw = torch.randn(batch_size, num_nodes, embedding_dim)
    return F.normalize(raw, p=2, dim=-1)


def _make_valid_density_matrix(
    batch_size: int = _BATCH_SIZE,
    num_regimes: int = _NUM_REGIMES,
) -> torch.Tensor:
    """Constructs a valid density matrix via Cholesky factorisation."""
    L = torch.randn(batch_size, num_regimes, num_regimes)
    L = torch.tril(L)
    rho = L @ L.transpose(-2, -1)
    trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(
        dim=-1, keepdim=True
    ).unsqueeze(-1)
    rho = rho / trace.clamp(min=1e-8)
    return rho


def _make_regime_probs(
    batch_size: int = _BATCH_SIZE,
    num_regimes: int = _NUM_REGIMES,
) -> torch.Tensor:
    """Creates a valid regime probability table (rows sum to 1)."""
    logits = torch.randn(batch_size, num_regimes)
    return F.softmax(logits, dim=-1)


# Prediction loss tests.


def test_prediction_loss_scalar_output() -> None:
    """Prediction loss returns a scalar tensor."""
    predictions, targets = _make_predictions_and_targets()
    loss = prediction_loss(predictions, targets)

    assert loss.dim() == 0, "Loss should be a scalar."
    assert torch.isfinite(loss), "Loss should be finite."


def test_prediction_loss_zero_for_perfect_predictions() -> None:
    """MSE is zero when predictions exactly match targets."""
    targets = torch.randn(_BATCH_SIZE, _NUM_NODES)
    predictions = targets.unsqueeze(-1)
    loss = prediction_loss(predictions, targets)

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-7), (
        f"Expected zero loss for perfect predictions, got {loss.item():.2e}"
    )


def test_prediction_loss_positive_for_imperfect() -> None:
    """MSE is strictly positive when predictions differ from targets."""
    predictions = torch.ones(_BATCH_SIZE, _NUM_NODES, 1)
    targets = torch.zeros(_BATCH_SIZE, _NUM_NODES)
    loss = prediction_loss(predictions, targets)

    assert loss > 0, "MSE should be positive for non-matching inputs."


def test_prediction_loss_shape_invariance() -> None:
    """Loss handles both (B, N, 1) and (B, N) target shapes."""
    predictions = torch.randn(_BATCH_SIZE, _NUM_NODES, 1)
    targets_2d = torch.randn(_BATCH_SIZE, _NUM_NODES)
    targets_3d = targets_2d.unsqueeze(-1)

    loss_2d = prediction_loss(predictions, targets_2d)
    loss_3d = prediction_loss(predictions, targets_3d)

    assert torch.allclose(loss_2d, loss_3d, atol=1e-7), (
        "Loss should be identical regardless of target shape."
    )


# Uniformity loss tests.


def test_uniformity_loss_finite() -> None:
    """Uniformity loss returns a finite scalar."""
    embeddings = _make_embeddings()
    loss = uniformity_loss(embeddings)

    assert loss.dim() == 0, "Loss should be a scalar."
    assert torch.isfinite(loss), "Loss should be finite."


def test_uniformity_loss_lower_for_spread_points() -> None:
    """More uniformly spread embeddings yield a lower uniformity loss.

    Points on opposite ends of the hypersphere (well spread) should
    produce a lower Gaussian potential than points clustered together.
    """
    base = torch.randn(1, _EMBEDDING_DIM)
    base = F.normalize(base, p=2, dim=-1)
    noise = torch.randn(1, _NUM_NODES, _EMBEDDING_DIM) * 0.01
    clustered = F.normalize(
        base.unsqueeze(0).expand(1, _NUM_NODES, -1) + noise, p=2, dim=-1,
    )

    spread = _make_embeddings(batch_size=1, num_nodes=_NUM_NODES)

    loss_clustered = uniformity_loss(clustered)
    loss_spread = uniformity_loss(spread)

    assert loss_spread < loss_clustered, (
        f"Spread points should have lower uniformity loss: "
        f"spread={loss_spread.item():.4f}, clustered={loss_clustered.item():.4f}"
    )


def test_uniformity_loss_2d_input() -> None:
    """Uniformity loss handles 2D input (N, E) without batch dimension."""
    points = F.normalize(
        torch.randn(_NUM_NODES, _EMBEDDING_DIM), p=2, dim=-1,
    )
    loss = uniformity_loss(points)

    assert loss.dim() == 0, "Loss should be a scalar."
    assert torch.isfinite(loss), "Loss should be finite."


# Regime entropy loss tests (replaces standalone density-matrix regularisation).


def test_regime_entropy_loss_scalar() -> None:
    """Regime entropy loss returns a finite scalar."""
    probs = _make_regime_probs()
    rho = _make_valid_density_matrix()
    loss = regime_entropy_loss(probs, rho=rho)

    assert loss.dim() == 0, "Loss should be a scalar."
    assert torch.isfinite(loss), "Loss should be finite."


def test_regime_entropy_loss_without_rho() -> None:
    """Shannon term alone is finite when rho is omitted."""
    probs = _make_regime_probs()
    loss = regime_entropy_loss(probs, rho=None)

    assert loss.dim() == 0, "Loss should be a scalar."
    assert torch.isfinite(loss), "Loss should be finite."


def test_regime_entropy_differs_with_rho() -> None:
    """Including rho changes the objective compared to probabilities alone."""
    probs = _make_regime_probs()
    rho = _make_valid_density_matrix()
    loss_probs_only = regime_entropy_loss(probs, rho=None)
    loss_with_rho = regime_entropy_loss(probs, rho=rho)

    assert not torch.allclose(loss_probs_only, loss_with_rho, atol=1e-6), (
        "Von Neumann term should alter the loss when rho is provided."
    )


def test_regime_entropy_uniform_vs_peaked() -> None:
    """Peaked versus uniform regime distributions yield different losses."""
    batch = _BATCH_SIZE
    uniform = torch.full((batch, _NUM_REGIMES), 1.0 / _NUM_REGIMES)
    peaked = F.one_hot(torch.zeros(batch, dtype=torch.long), _NUM_REGIMES).float()

    loss_u = regime_entropy_loss(uniform, rho=None)
    loss_p = regime_entropy_loss(peaked, rho=None)

    assert not torch.allclose(loss_u, loss_p, atol=1e-4), (
        "Uniform and peaked regime vectors should not give identical loss."
    )


# Directional loss tests.


def test_directional_loss_negative_for_correct_signs() -> None:
    """Directional loss is negative when all predicted signs are correct.

    When sign(y) * y_hat > 0 for all predictions, the mean product
    is positive and the negation makes the loss negative.
    """
    targets = torch.tensor([[1.0, -1.0, 0.5, -0.3]]).unsqueeze(0)
    predictions = torch.tensor([[0.5, -0.8, 0.2, -0.1]]).unsqueeze(-1)

    loss = directional_loss(predictions, targets)

    assert loss < 0, (
        f"Loss should be negative for correct sign alignment, got {loss.item():.4f}"
    )


def test_directional_loss_positive_for_wrong_signs() -> None:
    """Directional loss is positive when all predicted signs are wrong."""
    targets = torch.tensor([[1.0, -1.0, 0.5, -0.3]]).unsqueeze(0)
    predictions = torch.tensor([[-0.5, 0.8, -0.2, 0.1]]).unsqueeze(-1)

    loss = directional_loss(predictions, targets)

    assert loss > 0, (
        f"Loss should be positive for wrong sign alignment, got {loss.item():.4f}"
    )


def test_directional_loss_scalar_output() -> None:
    """Directional loss returns a scalar tensor."""
    predictions, targets = _make_predictions_and_targets()
    loss = directional_loss(predictions, targets)

    assert loss.dim() == 0, "Loss should be a scalar."
    assert torch.isfinite(loss), "Loss should be finite."


# Combined GDLQuantumLoss tests.


def test_combined_loss_returns_all_keys() -> None:
    """Combined loss returns a dictionary with all expected keys."""
    criterion = GDLQuantumLoss()
    predictions, targets = _make_predictions_and_targets()
    embeddings = _make_embeddings()
    rho = _make_valid_density_matrix()
    regime_probs = _make_regime_probs()

    result = criterion(
        predictions,
        targets,
        embeddings,
        rho,
        regime_probs,
    )

    expected_keys = {
        "total", "prediction", "uniformity",
        "density_matrix", "directional",
    }
    assert set(result.keys()) == expected_keys, (
        f"Missing keys: {expected_keys - set(result.keys())}"
    )


def test_combined_loss_all_finite() -> None:
    """All loss components are finite scalars."""
    criterion = GDLQuantumLoss()
    predictions, targets = _make_predictions_and_targets()
    embeddings = _make_embeddings()
    rho = _make_valid_density_matrix()
    regime_probs = _make_regime_probs()

    result = criterion(
        predictions,
        targets,
        embeddings,
        rho,
        regime_probs,
    )

    for key, value in result.items():
        assert value.dim() == 0, f"'{key}' should be a scalar."
        assert torch.isfinite(value), f"'{key}' should be finite."


def test_combined_loss_gradient_flow() -> None:
    """Gradients propagate through the total loss to all inputs.

    Verifies that backpropagation through the combined loss produces
    finite gradients for predictions and embeddings, confirming the
    loss is fully differentiable.
    """
    predictions = torch.randn(
        _BATCH_SIZE, _NUM_NODES, 1, requires_grad=True,
    )
    targets = torch.randn(_BATCH_SIZE, _NUM_NODES)

    embeddings = F.normalize(
        torch.randn(
            _BATCH_SIZE, _NUM_NODES, _EMBEDDING_DIM, requires_grad=True,
        ),
        p=2, dim=-1,
    )
    embeddings.retain_grad()

    rho = _make_valid_density_matrix()
    rho.requires_grad_(True)
    regime_probs = _make_regime_probs().requires_grad_(True)

    criterion = GDLQuantumLoss()
    result = criterion(
        predictions,
        targets,
        embeddings,
        rho,
        regime_probs,
    )
    result["total"].backward()

    assert predictions.grad is not None, (
        "Predictions should receive gradients."
    )
    assert torch.isfinite(predictions.grad).all(), (
        "Prediction gradients should be finite."
    )
    assert embeddings.grad is not None, (
        "Embeddings should receive gradients."
    )
    assert torch.isfinite(embeddings.grad).all(), (
        "Embedding gradients should be finite."
    )


def test_combined_loss_weight_influence() -> None:
    """Changing loss weights alters the total loss value.

    Verifies that the weighting mechanism is functional by comparing
    the total loss under different weight configurations.
    """
    predictions, targets = _make_predictions_and_targets()
    embeddings = _make_embeddings()
    rho = _make_valid_density_matrix()
    regime_probs = _make_regime_probs()

    criterion_default = GDLQuantumLoss(
        w_pred=1.0, w_uniform=0.01, w_dm=0.001, w_dir=0.1,
    )
    criterion_heavy_uniform = GDLQuantumLoss(
        w_pred=1.0, w_uniform=10.0, w_dm=0.001, w_dir=0.1,
    )

    result_default = criterion_default(
        predictions,
        targets,
        embeddings,
        rho,
        regime_probs,
    )
    result_heavy = criterion_heavy_uniform(
        predictions,
        targets,
        embeddings,
        rho,
        regime_probs,
    )

    assert torch.allclose(
        result_default["prediction"],
        result_heavy["prediction"],
        atol=1e-7,
    ), "Prediction loss component should be independent of weights."

    assert not torch.allclose(
        result_default["total"],
        result_heavy["total"],
        atol=1e-3,
    ), "Total loss should differ when weights change."
