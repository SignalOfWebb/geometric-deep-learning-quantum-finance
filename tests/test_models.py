"""Integration tests for the GDLQuantumFinanceModel.

Tests the full forward pass, backward gradient flow, variable-sized
graphs, and deterministic reproducibility of the integrated model
that combines the geometric, quantum, and graph layers.
"""

import torch

from src.losses import GDLQuantumLoss
from src.models import GDLQuantumFinanceModel

# Shared model dimensions matching config.yaml defaults.
_FEATURE_DIM = 16
_EMBEDDING_DIM = 32
_HIDDEN_DIM = 64
_GRAPH_OUT_DIM = 32
_NUM_REGIMES = 4
_GAT_HEADS = 4
_CONTEXT_DIM = 3
_PREDICTION_HIDDEN_DIM = 16


def _make_model() -> GDLQuantumFinanceModel:
    """Creates a GDLQuantumFinanceModel with standard test dimensions."""
    return GDLQuantumFinanceModel(
        feature_dim=_FEATURE_DIM,
        embedding_dim=_EMBEDDING_DIM,
        hidden_dim=_HIDDEN_DIM,
        graph_out_dim=_GRAPH_OUT_DIM,
        num_regimes=_NUM_REGIMES,
        gat_heads=_GAT_HEADS,
        context_dim=_CONTEXT_DIM,
        prediction_hidden_dim=_PREDICTION_HIDDEN_DIM,
        dropout=0.0,
        edge_drop=0.0,
    )


def _make_inputs(
    batch_size: int = 1,
    num_nodes: int = 30,
    num_edges: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Creates random model inputs in flat PyG batch format.

    Node features are ``(total_nodes, D)``; ``batch`` maps each row to a graph
    index. For ``batch_size > 1``, graphs are disjoint copies with the same
    local topology, with ``edge_index`` offsets per graph.

    Args:
        batch_size: Number of graphs in the batch.
        num_nodes: Nodes per graph.
        num_edges: Edges per graph (same count for each graph when batched).

    Returns:
        Tuple of ``(x, edge_index, edge_weight, context, batch)``.
    """
    if batch_size == 1:
        x = torch.randn(num_nodes, _FEATURE_DIM)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_weight = torch.rand(num_edges).abs()
        context = torch.randn(1, _CONTEXT_DIM)
        batch = torch.zeros(num_nodes, dtype=torch.long)
        return x, edge_index, edge_weight, context, batch

    total_nodes = batch_size * num_nodes
    x = torch.randn(total_nodes, _FEATURE_DIM)
    edge_chunks = []
    weight_chunks = []
    for b in range(batch_size):
        ei = torch.randint(0, num_nodes, (2, num_edges))
        ei = ei + b * num_nodes
        edge_chunks.append(ei)
        weight_chunks.append(torch.rand(num_edges).abs())
    edge_index = torch.cat(edge_chunks, dim=1)
    edge_weight = torch.cat(weight_chunks, dim=0)
    context = torch.randn(batch_size, _CONTEXT_DIM)
    batch = torch.arange(batch_size, dtype=torch.long).repeat_interleave(num_nodes)
    return x, edge_index, edge_weight, context, batch


def test_full_model_forward() -> None:
    """End-to-end forward pass returns the expected output keys and shapes.

    The model should return a dictionary containing predictions, regime
    probabilities, the evolved density matrix, and geometric embeddings.
    """
    model = _make_model()
    model.eval()
    batch_size, num_nodes = 1, 30
    x, edge_index, edge_weight, context, batch = _make_inputs(
        batch_size=batch_size,
        num_nodes=num_nodes,
    )

    output = model(x, edge_index, edge_weight, context, batch=batch)

    expected_keys = {"predictions", "regime_probs", "density_matrix", "embeddings"}
    assert set(output.keys()) == expected_keys, (
        f"Missing output keys: {expected_keys - set(output.keys())}"
    )
    assert output["predictions"].shape == (batch_size * num_nodes, 1), (
        f"Predictions shape mismatch: {output['predictions'].shape}"
    )
    assert output["regime_probs"].shape == (batch_size, _NUM_REGIMES), (
        f"Regime probs shape mismatch: {output['regime_probs'].shape}"
    )
    assert output["density_matrix"].shape == (
        batch_size,
        _NUM_REGIMES,
        _NUM_REGIMES,
    ), f"Density matrix shape mismatch: {output['density_matrix'].shape}"
    assert output["embeddings"].shape == (num_nodes, _EMBEDDING_DIM), (
        f"Embeddings shape mismatch: {output['embeddings'].shape}"
    )


def test_full_model_backward() -> None:
    """Loss backward through the entire model produces gradients for all parameters.

    Every learnable parameter in the model (geometric projection, quantum
    Cholesky map, H_raw, GCN/GAT weights, gate, prediction head) must
    receive a finite gradient after a backward pass.
    """
    model = _make_model()
    model.train()
    criterion = GDLQuantumLoss()

    x, edge_index, edge_weight, context, batch = _make_inputs(batch_size=1, num_nodes=30)
    targets = torch.randn(30)
    output = model(x, edge_index, edge_weight, context, batch=batch)
    losses = criterion(
        output["predictions"],
        targets,
        output["embeddings"],
        output["density_matrix"],
        output["regime_probs"],
    )
    losses["total"].backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, (
            f"Parameter '{name}' has no gradient."
        )
        assert torch.isfinite(param.grad).all(), (
            f"Parameter '{name}' has non-finite gradient."
        )


def test_full_model_variable_nodes() -> None:
    """Two forward passes with different numbers of active tickers succeed.

    The model must handle variable-sized cross-sections because the
    number of active tickers N_t changes across dates. This test runs
    the model with 20 and 50 nodes to confirm there are no shape
    mismatches or hard-coded size assumptions.
    """
    model = _make_model()
    model.eval()

    x_small, ei_small, ew_small, ctx_small, b_small = _make_inputs(
        batch_size=1,
        num_nodes=20,
        num_edges=60,
    )
    output_small = model(x_small, ei_small, ew_small, ctx_small, batch=b_small)

    assert output_small["predictions"].shape == (20, 1)

    x_large, ei_large, ew_large, ctx_large, b_large = _make_inputs(
        batch_size=1,
        num_nodes=50,
        num_edges=200,
    )
    output_large = model(x_large, ei_large, ew_large, ctx_large, batch=b_large)

    assert output_large["predictions"].shape == (50, 1)

    for key in ("predictions", "regime_probs", "density_matrix"):
        assert torch.isfinite(output_small[key]).all(), (
            f"Small graph: '{key}' contains non-finite values."
        )
        assert torch.isfinite(output_large[key]).all(), (
            f"Large graph: '{key}' contains non-finite values."
        )


def test_full_model_deterministic() -> None:
    """Same input with the same seed produces identical output.

    Verifies that the model is deterministic when dropout and
    DropEdge are disabled (set to 0.0 via _make_model).
    """
    model = _make_model()
    model.eval()

    torch.manual_seed(42)
    x, edge_index, edge_weight, context, batch = _make_inputs()

    output_1 = model(x, edge_index, edge_weight, context, batch=batch)

    output_2 = model(x, edge_index, edge_weight, context, batch=batch)

    for key in ("predictions", "regime_probs", "density_matrix", "embeddings"):
        assert torch.allclose(output_1[key], output_2[key], atol=1e-6), (
            f"Output '{key}' is not deterministic across identical runs."
        )


def test_full_model_regime_probs_valid() -> None:
    """Regime probabilities from the full model form a valid distribution.

    The regime probabilities produced by the quantum layer within
    the integrated model should be non-negative and sum to 1.
    """
    model = _make_model()
    model.eval()
    x, edge_index, edge_weight, context, batch = _make_inputs()

    output = model(x, edge_index, edge_weight, context, batch=batch)
    probs = output["regime_probs"]

    assert (probs >= -1e-7).all(), (
        f"Regime probabilities contain negatives: min={probs.min().item():.2e}"
    )
    assert torch.allclose(
        probs.sum(dim=-1),
        torch.ones(probs.size(0), dtype=probs.dtype, device=probs.device),
        atol=1e-5,
    ), (
        f"Regime probabilities do not sum to 1: sum={probs.sum(dim=-1)}"
    )


def test_full_model_embeddings_on_manifold() -> None:
    """Geometric embeddings from the full model lie on the unit hypersphere.

    All embedding vectors should have unit L2 norm, confirming the
    geometric layer's normalisation is preserved through the forward pass.
    """
    model = _make_model()
    model.eval()
    x, edge_index, edge_weight, context, batch = _make_inputs()

    output = model(x, edge_index, edge_weight, context, batch=batch)
    norms = torch.norm(output["embeddings"], p=2, dim=-1)

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), (
        f"Embedding norms deviate from 1: "
        f"min={norms.min().item():.6f}, max={norms.max().item():.6f}"
    )


def test_full_model_with_loss_integration() -> None:
    """End-to-end integration of model forward pass and loss computation.

    Verifies that model outputs can be fed directly into GDLQuantumLoss
    without shape mismatches or runtime errors.
    """
    model = _make_model()
    model.train()
    criterion = GDLQuantumLoss()

    batch_size, num_nodes = 1, 25
    x, edge_index, edge_weight, context, batch_vector = _make_inputs(
        batch_size=batch_size,
        num_nodes=num_nodes,
    )
    targets = torch.randn(num_nodes)

    output = model(x, edge_index, edge_weight, context, batch=batch_vector)
    losses = criterion(
        output["predictions"],
        targets,
        output["embeddings"],
        output["density_matrix"],
        output["regime_probs"],
    )

    assert "total" in losses, "Loss dictionary must contain 'total' key."
    assert torch.isfinite(losses["total"]), "Total loss should be finite."

    losses["total"].backward()

    param_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    assert param_count == total_params, (
        f"Only {param_count}/{total_params} parameters received gradients."
    )
