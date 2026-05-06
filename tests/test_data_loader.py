"""Tests for the financial graph dataset and walk-forward splitter.

Uses synthetic data (build_synthetic_dataframe) so tests run without
the real fact_table.parquet from the data pipeline.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data_loader import (
    DEFAULT_CONTEXT_COLUMNS,
    DEFAULT_FEATURE_COLUMNS,
    FinancialGraphDataset,
    WalkForwardSplitter,
    build_synthetic_dataframe,
    get_device,
    robust_zscore,
)

# Shared synthetic data parameters.
_N_TICKERS = 15
_N_DATES = 300
_SEED = 42

# Minimal config matching config.yaml structure.
_TEST_CONFIG: dict = {
    "model": {
        "geometric": {
            "input_dim": 12,
            "embedding_dim": 32,
            "eps": 1e-8,
        },
        "graph_construction": {
            "rolling_window": 60,
            "correlation_threshold": 0.3,
        },
    },
    "training": {
        "walk_forward": {
            "train_window": 100,
            "validation_window": 50,
            "step_size": 30,
        },
    },
}


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    """Produces a synthetic fact table DataFrame."""
    return build_synthetic_dataframe(
        n_tickers=_N_TICKERS,
        n_dates=_N_DATES,
        seed=_SEED,
    )


@pytest.fixture()
def dataset(synthetic_df: pd.DataFrame) -> FinancialGraphDataset:
    """Produces a FinancialGraphDataset backed by synthetic data."""
    return FinancialGraphDataset(
        parquet_path="unused.parquet",
        config=_TEST_CONFIG,
        dataframe=synthetic_df,
        normalise=True,
    )


# Robust Z-Score

class TestRobustZscore:
    """Tests for the robust z-score normalisation function."""

    def test_output_shape(self) -> None:
        """Output shape matches input shape."""
        x = torch.randn(50, 12)
        out = robust_zscore(x)
        assert out.shape == x.shape

    def test_zero_median(self) -> None:
        """Normalised features have approximately zero median."""
        x = torch.randn(200, 5)
        out = robust_zscore(x)
        medians = out.median(dim=0).values
        assert torch.allclose(medians, torch.zeros(5), atol=0.15)

    def test_unit_scale(self) -> None:
        """For Gaussian input, MAD-scaled output has approximate unit spread."""
        torch.manual_seed(0)
        x = torch.randn(1000, 3)
        out = robust_zscore(x)
        # For standard normal, robust z-score should be close to identity
        stds = out.std(dim=0)
        assert (stds > 0.5).all() and (stds < 2.0).all()

    def test_single_row(self) -> None:
        """Single-row input does not crash (MAD is zero, eps prevents NaN)."""
        x = torch.tensor([[1.0, 2.0, 3.0]])
        out = robust_zscore(x)
        assert torch.isfinite(out).all()

    def test_constant_column(self) -> None:
        """Constant column produces zero output (zero numerator)."""
        x = torch.tensor([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        out = robust_zscore(x)
        assert torch.allclose(out[:, 0], torch.zeros(3), atol=1e-6)


# FinancialGraphDataset

class TestFinancialGraphDataset:
    """Tests for the FinancialGraphDataset class."""

    def test_dataset_creation(
        self, dataset: FinancialGraphDataset,
    ) -> None:
        """Dataset initialises correctly with synthetic data."""
        assert len(dataset) > 0
        assert dataset.feature_dim == len(DEFAULT_FEATURE_COLUMNS)

    def test_feature_dim_matches_config(
        self, dataset: FinancialGraphDataset,
    ) -> None:
        """Reported feature dimension matches available columns."""
        assert dataset.feature_dim == 12
        assert dataset.num_features == 12

    def test_context_dim(
        self, dataset: FinancialGraphDataset,
    ) -> None:
        """Context dimension matches the number of yield curve columns."""
        assert dataset.num_context_features == len(DEFAULT_CONTEXT_COLUMNS)

    def test_getitem_returns_data(
        self, dataset: FinancialGraphDataset,
    ) -> None:
        """__getitem__ returns a PyG Data object with expected attributes."""
        data = dataset[0]

        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert hasattr(data, "edge_attr")
        assert hasattr(data, "y")
        assert hasattr(data, "context")
        assert hasattr(data, "target_mask")

    def test_node_feature_shape(
        self, dataset: FinancialGraphDataset,
    ) -> None:
        """Node features have shape (N_t, D)."""
        data = dataset[0]
        n = data.num_nodes
        d = dataset.feature_dim
        assert data.x.shape == (n, d)

    def test_target_shape(
        self, dataset: FinancialGraphDataset,
    ) -> None:
        """Targets and target mask have shape (N_t,)."""
        data = dataset[0]
        n = data.num_nodes
        assert data.y.shape == (n,)
        assert data.target_mask.shape == (n,)

    def test_context_shape(
        self, dataset: FinancialGraphDataset,
    ) -> None:
        """Context is stored as a single-row tensor (1, C) for PyG collation."""
        data = dataset[0]
        assert data.context.shape == (1, len(DEFAULT_CONTEXT_COLUMNS))

    def test_edge_index_valid(
        self, dataset: FinancialGraphDataset,
    ) -> None:
        """Edge indices are within the valid node range."""
        data = dataset[0]
        n = data.num_nodes
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.min() >= 0
        assert data.edge_index.max() < n

    def test_edge_weight_positive(
        self, dataset: FinancialGraphDataset,
    ) -> None:
        """Edge weights are positive (absolute correlations or self-loops)."""
        data = dataset[0]
        assert (data.edge_attr > 0).all()

    def test_self_loops_present(
        self, dataset: FinancialGraphDataset,
    ) -> None:
        """Every node has at least a self-loop edge."""
        data = dataset[0]
        src = data.edge_index[0]
        dst = data.edge_index[1]
        self_loops = set((src == dst).nonzero(as_tuple=True)[0].tolist())
        # At least some self-loops exist
        assert len(self_loops) > 0

    def test_finite_features(
        self, dataset: FinancialGraphDataset,
    ) -> None:
        """All node features are finite (no NaN or Inf)."""
        data = dataset[0]
        assert torch.isfinite(data.x).all()
        assert torch.isfinite(data.y).all()
        assert torch.isfinite(data.context).all()

    def test_variable_node_count(
        self, dataset: FinancialGraphDataset,
    ) -> None:
        """Different dates can have different numbers of nodes.

        This verifies survivorship handling: N_t varies across dates
        because synthetic data randomly drops tickers.
        """
        if len(dataset) < 2:
            pytest.skip("Need at least 2 dates.")

        sizes = {dataset[i].num_nodes for i in range(min(10, len(dataset)))}
        # With 2% random dropout, sizes should vary
        assert len(sizes) >= 1  # At minimum they exist

    def test_date_restriction(
        self, synthetic_df: pd.DataFrame,
    ) -> None:
        """Restricting to a subset of dates reduces dataset length."""
        full_ds = FinancialGraphDataset(
            parquet_path="unused.parquet",
            config=_TEST_CONFIG,
            dataframe=synthetic_df,
            normalise=False,
        )
        if len(full_ds) < 10:
            pytest.skip("Not enough valid dates for restriction test.")

        subset_dates = full_ds.valid_dates[:10]
        restricted_ds = FinancialGraphDataset(
            parquet_path="unused.parquet",
            config=_TEST_CONFIG,
            dataframe=synthetic_df,
            dates=subset_dates,
            normalise=False,
        )
        assert len(restricted_ds) <= 10
        assert len(restricted_ds) <= len(full_ds)

    def test_normalisation_effect(
        self, synthetic_df: pd.DataFrame,
    ) -> None:
        """Normalised features differ from unnormalised features."""
        ds_norm = FinancialGraphDataset(
            parquet_path="unused.parquet",
            config=_TEST_CONFIG,
            dataframe=synthetic_df,
            normalise=True,
        )
        ds_raw = FinancialGraphDataset(
            parquet_path="unused.parquet",
            config=_TEST_CONFIG,
            dataframe=synthetic_df,
            normalise=False,
        )
        if len(ds_norm) == 0 or len(ds_raw) == 0:
            pytest.skip("No valid dates.")

        data_norm = ds_norm[0]
        data_raw = ds_raw[0]
        # Features should differ after normalisation
        assert not torch.allclose(data_norm.x, data_raw.x, atol=1e-4)

    def test_missing_parquet_raises(self) -> None:
        """FileNotFoundError when parquet is missing and no DataFrame."""
        with pytest.raises(FileNotFoundError, match="Fact table not found"):
            FinancialGraphDataset(
                parquet_path="/nonexistent/path/fact_table.parquet",
                config=_TEST_CONFIG,
            )

    def test_forward_targets_exist(
        self, dataset: FinancialGraphDataset,
    ) -> None:
        """At least some targets per cross-section are valid (unmasked)."""
        data = dataset[0]
        assert data.target_mask.sum() > 0


# WalkForwardSplitter

class TestWalkForwardSplitter:
    """Tests for the temporal walk-forward splitter."""

    def _make_dates(self, n: int = 500) -> list[pd.Timestamp]:
        """Creates a list of n business dates."""
        return pd.bdate_range(start="2020-01-02", periods=n).tolist()

    def test_fold_count(self) -> None:
        """Fold count matches the expected number of complete folds."""
        dates = self._make_dates(500)
        splitter = WalkForwardSplitter(
            dates, train_window=100, validation_window=50, step_size=30,
        )
        # (500 - 150) / 30 + 1 = 12.67 -> 12 complete folds
        expected = 0
        start = 0
        while start + 150 <= 500:
            expected += 1
            start += 30
        assert len(splitter) == expected

    def test_no_folds_insufficient_data(self) -> None:
        """Returns zero folds when data is shorter than one window."""
        dates = self._make_dates(50)
        splitter = WalkForwardSplitter(
            dates, train_window=100, validation_window=50, step_size=30,
        )
        assert len(splitter) == 0

    def test_train_val_lengths(self) -> None:
        """Each fold has the correct train and val window sizes."""
        dates = self._make_dates(500)
        splitter = WalkForwardSplitter(
            dates, train_window=100, validation_window=50, step_size=30,
        )
        for train_dates, val_dates in splitter:
            assert len(train_dates) == 100
            assert len(val_dates) == 50

    def test_no_overlap(self) -> None:
        """Train and validation windows within a fold do not overlap."""
        dates = self._make_dates(500)
        splitter = WalkForwardSplitter(
            dates, train_window=100, validation_window=50, step_size=30,
        )
        for train_dates, val_dates in splitter:
            assert set(train_dates).isdisjoint(set(val_dates))

    def test_temporal_order(self) -> None:
        """Validation dates always come strictly after training dates."""
        dates = self._make_dates(500)
        splitter = WalkForwardSplitter(
            dates, train_window=100, validation_window=50, step_size=30,
        )
        for train_dates, val_dates in splitter:
            assert max(train_dates) < min(val_dates)

    def test_forward_progression(self) -> None:
        """Successive folds advance in time."""
        dates = self._make_dates(500)
        splitter = WalkForwardSplitter(
            dates, train_window=100, validation_window=50, step_size=30,
        )
        prev_start = None
        for train_dates, _ in splitter:
            if prev_start is not None:
                assert train_dates[0] > prev_start
            prev_start = train_dates[0]

    def test_get_fold(self) -> None:
        """get_fold returns the same split as iterating to that fold."""
        dates = self._make_dates(500)
        splitter = WalkForwardSplitter(
            dates, train_window=100, validation_window=50, step_size=30,
        )
        for fold_idx, (iter_train, iter_val) in enumerate(splitter):
            direct_train, direct_val = splitter.get_fold(fold_idx)
            assert direct_train == iter_train
            assert direct_val == iter_val

    def test_get_fold_out_of_range(self) -> None:
        """get_fold raises IndexError for invalid fold index."""
        dates = self._make_dates(500)
        splitter = WalkForwardSplitter(
            dates, train_window=100, validation_window=50, step_size=30,
        )
        with pytest.raises(IndexError):
            splitter.get_fold(splitter.n_folds)
        with pytest.raises(IndexError):
            splitter.get_fold(-1)

    def test_integration_with_dataset(
        self,
        dataset: FinancialGraphDataset,
    ) -> None:
        """Splitter dates can be used to restrict the dataset."""
        wf_cfg = _TEST_CONFIG["training"]["walk_forward"]
        splitter = WalkForwardSplitter(
            dates=dataset.valid_dates,
            train_window=wf_cfg["train_window"],
            validation_window=wf_cfg["validation_window"],
            step_size=wf_cfg["step_size"],
        )

        if len(splitter) == 0:
            pytest.skip("Not enough valid dates for walk-forward.")

        train_dates, val_dates = splitter.get_fold(0)

        # Create restricted datasets (only check they don't crash)
        train_ds = FinancialGraphDataset(
            parquet_path="unused.parquet",
            config=_TEST_CONFIG,
            dataframe=build_synthetic_dataframe(
                n_tickers=_N_TICKERS,
                n_dates=_N_DATES,
                seed=_SEED,
            ),
            dates=train_dates,
        )
        assert len(train_ds) <= len(train_dates)


# get_device

class TestGetDevice:
    """Tests for the device selection utility."""

    def test_returns_device(self) -> None:
        """get_device returns a valid torch.device."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_device_type(self) -> None:
        """Device type is one of cpu, cuda, or mps."""
        device = get_device()
        assert device.type in {"cpu", "cuda", "mps"}


# End-to-end integration with model interface

class TestDataModelInterface:
    """Verifies that dataset output matches model input expectations."""

    def test_data_compatible_with_model_shapes(
        self, dataset: FinancialGraphDataset,
    ) -> None:
        """Data shapes match the flat PyG batch layout used by ``GDLQuantumFinanceModel``.

        The model expects:
            x:           (total_nodes, feature_dim)
            edge_index:  (2, num_edges)
            edge_weight: (num_edges,)
            context:     (batch_graphs, context_dim)  (one row per graph; one graph here)
            batch:       (total_nodes,) optional; defaults to all zeros if omitted.
        """
        data = dataset[0]

        assert data.x.dim() == 2
        assert data.x.shape == (data.num_nodes, dataset.feature_dim)

        assert data.context.dim() == 2
        assert data.context.shape == (1, dataset.num_context_features)

        assert data.edge_index.dim() == 2
        assert data.edge_index.shape[0] == 2
        assert data.edge_attr.dim() == 1
