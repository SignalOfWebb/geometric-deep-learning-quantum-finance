"""Progressive checks for the integrated model before large-scale training.

Validates sphere constraints, quantum probability axioms, graph dynamics, then
light overfitting stages to confirm optimisation and capacity. Intended as a
structured substitute for ad hoc notebook debugging.

References:
    Niessner, M. (2025) "Progressive Model Debugging Protocol."
    Karpathy, A. (2019) "A Recipe for Training Neural Networks."
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path when run as a script
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_loader import get_device
from src.models import GDLQuantumFinanceModel
from src.train import build_model, load_config, _validate_device

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configures the root logger with the project-standard format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_geometric_layer(model: GDLQuantumFinanceModel, device: torch.device) -> bool:
    """Verifies that the geometric embedding respects manifold constraints.

    Ensures output points lie on the unit hypersphere, gradients flow through the
    projection, and no numerical instabilities (NaN/Inf) occur.

    Args:
        model: Full model instance.
        device: Compute device.

    Returns:
        True if all manifold constraints are satisfied.
    """
    logger.info("Validating Geometric Layer constraints...")
    layer = model.geometric
    layer.train()

    # Synthetic input: (batch=2, nodes=5, features=D)
    x = torch.randn(2, 5, layer.input_dim, device=device, requires_grad=True)

    output = layer(x)

    # Intent: Manifold Constraint (Wang & Isola).
    # If points leave the sphere surface (norm != 1), the geodesic distance
    # calculations (arccos) will return NaN, crashing the training.
    # This check ensures the projection is mathematically exact.
    # Constraint 1: All points must lie on the unit sphere (L2 norm = 1)
    norms = torch.norm(output, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
        logger.error("Constraint violated: Points not on unit sphere. Norms: %s", norms)
        return False

    # Constraint 2: Gradients must flow back to input
    loss = output.sum()
    loss.backward()
    if x.grad is None or torch.isnan(x.grad).any():
        logger.error("Gradient failure: No gradients or NaNs in backward pass.")
        return False

    # Constraint 3: Numerical stability
    if torch.isnan(output).any() or torch.isinf(output).any():
        logger.error("Numerical instability: NaN or Inf detected in output.")
        return False

    logger.info("Geometric Layer validation passed.")
    return True


def validate_quantum_layer(model: GDLQuantumFinanceModel, device: torch.device) -> bool:
    """Verifies quantum probability axioms and unitarity.

    Ensures regime probabilities are valid (sum to 1, non-negative), the Cayley
    transform preserves unitarity, and the layer differentiates between inputs.

    Args:
        model: Full model instance.
        device: Compute device.

    Returns:
        True if quantum axioms and gradient checks pass.
    """
    logger.info("Validating Quantum Layer axioms...")
    layer = model.quantum
    layer.train()

    # Synthetic input: embeddings on sphere + context
    # Use batch_size=2 to verify input differentiation
    emb = torch.randn(2, layer.input_dim, device=device)
    emb = F.normalize(emb, p=2, dim=-1)
    emb.requires_grad_(True)
    
    ctx = torch.randn(2, layer.context_dim, device=device, requires_grad=True)

    regime_probs, _ = layer(emb, ctx)

    # Intent: Born Rule Axioms (Haven & Khrennikov).
    # A density matrix must have unit trace (Sum of probabilities = 1) and
    # non-negative eigenvalues. If this fails, the "probabilities" are nonsense.
    # Axiom 1: Probabilities must sum to 1
    prob_sums = regime_probs.sum(dim=-1)
    if not torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5):
        logger.error("Axiom violated: Probabilities do not sum to 1. Sums: %s", prob_sums)
        return False

    # Axiom 2: Probabilities must be non-negative
    if (regime_probs < -1e-5).any():
        logger.error("Axiom violated: Negative probabilities. Min: %s", regime_probs.min())
        return False

    # Gradient check
    loss = regime_probs.sum()
    loss.backward()
    if emb.grad is None or ctx.grad is None:
        logger.error("Gradient failure: Gradients detached from input.")
        return False

    # Responsiveness check: Different inputs should yield different regimes
    diff = (regime_probs[0] - regime_probs[1]).abs().mean()
    if diff < 1e-6:
        logger.error("Model failure: Identical outputs for different inputs (collapsed).")
        return False

    # Internal Unitarity check for Cayley transform
    try:
        U = layer._cayley_unitary()
        I = torch.eye(layer.num_regimes, device=device)
        if not torch.allclose(U @ U.T, I, atol=1e-5):
            logger.error("Unitarity violated: Cayley transform matrix is not unitary.")
            return False
    except AttributeError:
        logger.warning("Skipping internal unitarity check (method not accessible).")

    logger.info("Quantum Layer validation passed.")
    return True


def validate_graph_layer(model: GDLQuantumFinanceModel, device: torch.device) -> bool:
    """Verifies graph message passing and over-smoothing.

    Checks that node features are actually updated by the GNN (input != output)
    and that representations do not collapse to a single point (over-smoothing).

    Args:
        model: Full model instance.
        device: Compute device.

    Returns:
        True if graph operations behave as expected.
    """
    logger.info("Validating Graph Layer dynamics...")
    layer = model.graph
    layer.train()

    num_nodes = 10
    x = torch.randn(num_nodes, layer.in_channels, device=device, requires_grad=True)
    
    # Simple cycle graph
    edge_index = torch.tensor([
        [i, (i + 1) % num_nodes] for i in range(num_nodes)
    ], dtype=torch.long, device=device).t()
    edge_weight = torch.ones(num_nodes, device=device)

    output = layer(x, edge_index, edge_weight)

    # Shape verification
    if output.shape != (num_nodes, layer.out_channels):
        logger.error("Shape mismatch: Expected (%d, %d), got %s", 
                     num_nodes, layer.out_channels, output.shape)
        return False

    # Feature update verification
    if x.shape == output.shape and torch.allclose(x, output, atol=1e-6):
        logger.error("Dynamics failure: GNN did not update node features (identity mapping).")
        return False

    # Gradient flow
    output.sum().backward()
    if x.grad is None:
        logger.error("Gradient failure: No gradient flow to node features.")
        return False

    # Intent: Over-smoothing Check (Rong et al.).
    # A common failure mode in GNNs is "Over-smoothing," where all node
    # embeddings converge to the same average value. We verify that the
    # output variance is non-zero, proving the graph layer differentiates nodes.
    # Over-smoothing check: Ensure variance across nodes
    variance = output.var(dim=0).mean()
    if variance < 1e-6:
        logger.error("Collapsed representation: Over-smoothing detected (var=%.2e).", variance)
        return False

    logger.info("Graph Layer validation passed.")
    return True


def verify_gradient_flow(model: nn.Module) -> bool:
    """ audits all trainable parameters for dead gradients.

    Returns:
        True if all parameters receive non-zero gradients.
    """
    logger.info("Auditing gradient flow...")
    issues = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if param.grad is None:
            issues.append(f"{name}: None")
        elif param.grad.abs().sum() == 0:
            issues.append(f"{name}: Zero")
            
    if issues:
        for issue in issues:
            logger.error("Gradient dead spot: %s", issue)
        return False
        
    logger.info("Gradient flow healthy across all parameters.")
    return True


def stage1_single_sample_overfit(
    model: GDLQuantumFinanceModel,
    device: torch.device,
    config: dict
) -> bool:
    """Stage 1: Validates that the model can memorise a single sample.

    This "Hello World" of deep learning confirms that the architecture has
    sufficient capacity and the optimisation path is clear. Disables
    regularisation to allow pure fitting.

    Args:
        model: Fresh model instance.
        device: Compute device.
        config: Configuration dict.

    Returns:
        True if loss converges to < 1e-5.
    """
    # Intent: Capacity Check (Karpathy).
    # Before training on real data, we must prove the model has enough
    # capacity to memorise a single batch. If it can't overfit a small dataset,
    # it definitely won't learn general patterns from a large one.
    logger.info("Stage 1: Single Sample Overfit Test")
    
    geo_dim = config["model"]["geometric"]["input_dim"]
    ctx_dim = config["model"]["quantum"]["context_dim"]
    
    # Synthetic single sample (5 nodes)
    N = 5
    x = torch.randn(N, geo_dim, device=device)
    
    # Fully connected graph
    edges = torch.combinations(torch.arange(N, device=device), 2).t()
    edges = torch.cat([edges, edges.flip(0)], dim=1)
    weights = torch.ones(edges.size(1), device=device)
    
    context = torch.randn(1, ctx_dim, device=device)
    target = torch.randn(N, device=device)
    
    # Disable dropout to allow perfect fitting
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            m.p = 0.0
    
    # High LR for fast convergence on single sample
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    for i in range(501):
        optimiser.zero_grad()
        
        # No unsqueeze! PyG expects 2D node features (total_nodes, D)
        # Create a batch index vector of zeros for the single graph
        batch_idx = torch.zeros(N, dtype=torch.long, device=device)
        
        output = model(x, edges, weights, context, batch=batch_idx)
        loss = F.mse_loss(output["predictions"].squeeze(), target)
        
        loss.backward()
        
        # Audit gradients on first step
        if i == 0 and not verify_gradient_flow(model):
            return False
            
        optimiser.step()
        
        if loss.item() < 1e-5:
            logger.info("Converged at iteration %d (loss < 1e-5).", i)
            return True
            
        if i % 100 == 0:
            logger.info("Iter %d: Loss = %.6f", i, loss.item())
            
    logger.error("Failed to converge. Final loss: %.6f", loss.item())
    return False


def stage2_small_batch_overfit(
    model: GDLQuantumFinanceModel,
    device: torch.device,
    config: dict
) -> bool:
    """Stage 2: Validates that the model responds to input variations.

    Trains on a small batch of diverse samples to ensure the model isn't
    just learning a constant output (mode collapse) or ignoring the input.

    Args:
        model: Fresh model instance.
        device: Compute device.
        config: Configuration dict.

    Returns:
        True if loss converges and outputs are diverse.
    """
    logger.info("Stage 2: Small Batch Overfit Test")
    
    batch_size = 5
    N = 5
    geo_dim = config["model"]["geometric"]["input_dim"]
    ctx_dim = config["model"]["quantum"]["context_dim"]
    
    # Create diverse batch
    inputs = [torch.randn(N, geo_dim, device=device) for _ in range(batch_size)]
    targets = [torch.randn(N, device=device) for _ in range(batch_size)]
    contexts = [torch.randn(1, ctx_dim, device=device) for _ in range(batch_size)]
    
    # Shared graph structure
    edges = torch.combinations(torch.arange(N, device=device), 2).t()
    edges = torch.cat([edges, edges.flip(0)], dim=1)
    weights = torch.ones(edges.size(1), device=device)
    
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    # Fit the batch
    converged = False
    for i in range(1501):
        optimiser.zero_grad()
        loss = torch.tensor(0.0, device=device)
        
        for b in range(batch_size):
            batch_idx = torch.zeros(N, dtype=torch.long, device=device)
            out = model(inputs[b], edges, weights, contexts[b], batch=batch_idx)
            loss += F.mse_loss(out["predictions"].squeeze(), targets[b])
            
        loss = loss / batch_size
        loss.backward()
        optimiser.step()
        
        if loss.item() < 1e-5:
            logger.info("Batch converged at iteration %d.", i)
            converged = True
            break
            
        if i % 100 == 0:
            logger.info("Iter %d: Loss = %.6f", i, loss.item())
            
    if not converged:
        logger.warning("Batch training did not fully converge (<1e-5). Final Loss: %.6f", loss.item())
        # We allow a slightly looser tolerance for the batch test if it's close enough
        if loss.item() < 1e-3:
             logger.info("Loss is acceptable (<1e-3). Proceeding with differentiation check.")
        else:
             return False

    # Verification: Ensure outputs differ across the batch
    model.eval()
    with torch.no_grad():
        preds = []
        for b in range(batch_size):
            batch_idx = torch.zeros(N, dtype=torch.long, device=device)
            out = model(inputs[b], edges, weights, contexts[b], batch=batch_idx)
            preds.append(out["predictions"].squeeze())
            
        stacked = torch.stack(preds)
        variance = stacked.var(dim=0).mean().item()
        
        if variance < 1e-6:
            logger.error("Mode collapse: Model ignores input (variance=%.2e).", variance)
            return False
            
        # Check correlation to confirm actual learning
        flat_p = stacked.flatten().cpu().numpy()
        flat_t = torch.cat(targets).cpu().numpy()
        corr = np.corrcoef(flat_p, flat_t)[0, 1]
        
        if corr < 0.9:
            logger.error("Weak learning: Correlation %.4f < 0.9.", corr)
            return False
            
    logger.info("Batch validation passed (Var=%.2e, Corr=%.4f).", variance, corr)
    return True


def validate_lstm_baseline(config: dict, device: torch.device) -> bool:
    """Verifies the LSTM baseline behavior.

    Ensures the LSTM baseline can:
    1. Accept sequence input.
    2. Backpropagate gradients.
    3. Overfit a simple sequence pattern.

    Args:
        config: Configuration dictionary.
        device: Compute device.

    Returns:
        True if validation passes.
    """
    from src.run_ablation import LSTMBaseline

    logger.info("Validating LSTM Baseline...")
    
    hidden_dim = config["model"]["graph"]["hidden_channels"]
    model = LSTMBaseline(input_dim=1, hidden_dim=hidden_dim, num_layers=2).to(device)
    model.train()

    # Synthetic sequence data: (Batch=1, Nodes=5, Time=60, Feats=1)
    N = 5
    T = 60
    # Simple pattern: target = mean of last 5 steps
    x_seq = torch.randn(1, N, T, 1, device=device)
    target = x_seq.squeeze(0).squeeze(-1)[:, -5:].mean(dim=1)  # (N,)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for i in range(201):
        optimizer.zero_grad()
        # Mock other inputs required by forward signature (unused)
        dummy_x = torch.zeros(1, N, 1, device=device)
        dummy_edge = torch.zeros(2, 0, dtype=torch.long, device=device)
        dummy_ctx = torch.zeros(1, 1, device=device)

        # Flatten Batch and Nodes dimensions to match (total_nodes, T, 1)
        x_seq_flat = x_seq.view(-1, T, 1)
        
        out = model(dummy_x, dummy_edge, None, dummy_ctx, x_seq=x_seq_flat)
        preds = out["predictions"].squeeze()  # (N,)
        
        loss = F.mse_loss(preds, target)
        loss.backward()
        
        if i == 0:
            if not verify_gradient_flow(model):
                return False

        optimizer.step()

        if loss.item() < 1e-4:
            logger.info("LSTM Baseline passed (Converged at iter %d).", i)
            return True

    logger.error("LSTM Baseline failed to converge on simple mean pattern.")
    return False


def run_verification(args: argparse.Namespace) -> None:
    """Orchestrates the validation pipeline."""
    config = load_config()

    # Same MPS rule as training: Cayley transform backward is unsafe on MPS.
    raw_device = get_device()
    device = _validate_device(raw_device)

    logger.info("Starting verification on %s", device)
    model = build_model(config, device)
    
    # Protocol: Components -> Single Sample -> Small Batch
    
    if args.stage in [None, 0]:
        if not all([
            validate_geometric_layer(model, device),
            validate_quantum_layer(model, device),
            validate_graph_layer(model, device),
            validate_lstm_baseline(config, device),
        ]):
            sys.exit(1)

    if args.stage in [None, 1]:
        # Fresh model for training tests
        if not stage1_single_sample_overfit(build_model(config, device), device, config):
            sys.exit(1)

    if args.stage in [None, 2]:
        if not stage2_small_batch_overfit(build_model(config, device), device, config):
            sys.exit(1)
            
    logger.info("Verification Complete: All systems nominal.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDL-Quantum-Finance Model Verification")
    parser.add_argument("--stage", type=int, choices=[0, 1, 2], help="0=Components, 1=Single, 2=Batch")
    args = parser.parse_args()
    
    _configure_logging()
    run_verification(args)
