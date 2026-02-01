import numpy as np
import plotly.graph_objects as go

from glidesim.analysis.plots import (
    create_distribution_histogram,
    create_final_balance_histogram,
    create_individual_trajectory,
    create_portfolio_trajectories,
    create_regime_frequency_donut,
    create_regime_transition_heatmap,
    create_success_donut,
)
from glidesim.simulation.market_model import Regime


def test_create_success_donut_returns_figure():
    fig = create_success_donut(0.85)

    assert isinstance(fig, go.Figure)


def test_create_success_donut_data_correct():
    fig = create_success_donut(0.75)
    pie_data = fig.data[0]

    assert list(pie_data.values) == [0.75, 0.25]
    assert list(pie_data.labels) == ["Success", "Failure"]


def test_create_final_balance_histogram_returns_figure():
    portfolio_values = np.array(
        [
            [100_000, 110_000, 120_000],
            [100_000, 90_000, 80_000],
            [100_000, 105_000, 115_000],
        ]
    )
    percentiles = {10: 85_000, 50: 115_000, 90: 120_000}

    fig = create_final_balance_histogram(portfolio_values, percentiles)

    assert isinstance(fig, go.Figure)


def test_create_portfolio_trajectories_returns_figure():
    n_years = 5
    portfolio_values = np.random.default_rng(42).uniform(
        50_000, 200_000, size=(100, n_years + 1)
    )

    fig = create_portfolio_trajectories(portfolio_values, n_years)

    assert isinstance(fig, go.Figure)


def test_create_portfolio_trajectories_percentile_bands():
    n_years = 5
    portfolio_values = np.random.default_rng(42).uniform(
        50_000, 200_000, size=(100, n_years + 1)
    )

    fig = create_portfolio_trajectories(portfolio_values, n_years)

    assert len(fig.data) == 5
    assert fig.data[4].name == "Median"
    assert fig.data[1].name == "10th-90th percentile"
    assert fig.data[3].name == "25th-75th percentile"


def test_create_individual_trajectory_returns_figure():
    n_sims, n_years = 10, 5
    rng = np.random.default_rng(42)

    portfolio_values = rng.uniform(50_000, 200_000, size=(n_sims, n_years + 1))
    returns = rng.normal(0.07, 0.15, size=(n_sims, n_years))
    inflation = rng.normal(0.03, 0.01, size=(n_sims, n_years))
    withdrawals = rng.uniform(30_000, 50_000, size=(n_sims, n_years))
    regimes = rng.integers(0, 4, size=(n_sims, n_years), dtype=np.int8)

    fig = create_individual_trajectory(
        portfolio_values,
        returns,
        inflation,
        withdrawals,
        regimes,
        sim_index=0,
        n_years=n_years,
    )

    assert isinstance(fig, go.Figure)


def test_create_individual_trajectory_has_legend_entries():
    n_sims, n_years = 10, 20
    rng = np.random.default_rng(42)

    portfolio_values = rng.uniform(50_000, 200_000, size=(n_sims, n_years + 1))
    returns = rng.normal(0.07, 0.15, size=(n_sims, n_years))
    inflation = rng.normal(0.03, 0.01, size=(n_sims, n_years))
    withdrawals = rng.uniform(30_000, 50_000, size=(n_sims, n_years))
    regimes = np.tile([0, 1, 2, 3], (n_sims, n_years // 4 + 1))[:, :n_years].astype(
        np.int8
    )

    fig = create_individual_trajectory(
        portfolio_values,
        returns,
        inflation,
        withdrawals,
        regimes,
        sim_index=0,
        n_years=n_years,
    )

    legend_names = {trace.name for trace in fig.data if trace.showlegend}
    assert "Normal" in legend_names
    assert "Recession" in legend_names
    assert "Stagflation" in legend_names
    assert "Depression" in legend_names


def test_create_individual_trajectory_has_withdrawal_bars():
    n_sims, n_years = 10, 5
    rng = np.random.default_rng(42)

    portfolio_values = rng.uniform(50_000, 200_000, size=(n_sims, n_years + 1))
    returns = rng.normal(0.07, 0.15, size=(n_sims, n_years))
    inflation = rng.normal(0.03, 0.01, size=(n_sims, n_years))
    withdrawals = rng.uniform(30_000, 50_000, size=(n_sims, n_years))
    regimes = rng.integers(0, 4, size=(n_sims, n_years), dtype=np.int8)

    fig = create_individual_trajectory(
        portfolio_values,
        returns,
        inflation,
        withdrawals,
        regimes,
        sim_index=0,
        n_years=n_years,
    )

    bar_traces = [trace for trace in fig.data if isinstance(trace, go.Bar)]
    assert len(bar_traces) == n_years


def test_create_individual_trajectory_subplot_structure():
    n_sims, n_years = 10, 5
    rng = np.random.default_rng(42)

    portfolio_values = rng.uniform(50_000, 200_000, size=(n_sims, n_years + 1))
    returns = rng.normal(0.07, 0.15, size=(n_sims, n_years))
    inflation = rng.normal(0.03, 0.01, size=(n_sims, n_years))
    withdrawals = rng.uniform(30_000, 50_000, size=(n_sims, n_years))
    regimes = rng.integers(0, 4, size=(n_sims, n_years), dtype=np.int8)

    fig = create_individual_trajectory(
        portfolio_values,
        returns,
        inflation,
        withdrawals,
        regimes,
        sim_index=0,
        n_years=n_years,
    )

    assert "yaxis" in fig.layout
    assert "yaxis2" in fig.layout
    assert fig.layout.annotations[0].text == "Portfolio Value"
    assert fig.layout.annotations[1].text == "Annual Withdrawals"


def test_create_distribution_histogram_returns_figure():
    rng = np.random.default_rng(42)
    data = rng.normal(0.07, 0.15, size=(100, 30))
    regimes = rng.integers(0, 4, size=(100, 30), dtype=np.int8)

    fig = create_distribution_histogram(
        data, regimes, regime_filter=None, color="#3498db", x_label="Annual Return"
    )

    assert isinstance(fig, go.Figure)


def test_create_distribution_histogram_with_regime_filter():
    rng = np.random.default_rng(42)
    data = rng.normal(0.07, 0.15, size=(100, 30))
    regimes = np.zeros((100, 30), dtype=np.int8)
    regimes[:, 10:20] = Regime.RECESSION

    fig = create_distribution_histogram(
        data,
        regimes,
        regime_filter=Regime.RECESSION,
        color="#3498db",
        x_label="Annual Return",
    )

    assert isinstance(fig, go.Figure)


def test_create_distribution_histogram_empty_filter():
    rng = np.random.default_rng(42)
    data = rng.normal(0.07, 0.15, size=(100, 30))
    regimes = np.zeros((100, 30), dtype=np.int8)

    fig = create_distribution_histogram(
        data,
        regimes,
        regime_filter=Regime.DEPRESSION,
        color="#3498db",
        x_label="Annual Return",
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) == 1
    assert "No data" in fig.layout.annotations[0].text


def test_create_distribution_histogram_without_percent_format():
    rng = np.random.default_rng(42)
    data = rng.normal(50, 10, size=(100, 30))
    regimes = np.zeros((100, 30), dtype=np.int8)

    fig = create_distribution_histogram(
        data,
        regimes,
        regime_filter=None,
        color="#3498db",
        x_label="Value",
        format_as_percent=False,
    )

    assert isinstance(fig, go.Figure)
    annotation = fig.layout.annotations[0]
    assert "Mean:" in annotation.text
    assert "%" not in annotation.text


def test_create_regime_frequency_donut_returns_figure():
    rng = np.random.default_rng(42)
    regimes = rng.integers(0, 4, size=(100, 30), dtype=np.int8)

    fig = create_regime_frequency_donut(regimes)

    assert isinstance(fig, go.Figure)


def test_create_regime_frequency_donut_has_all_regimes():
    regimes = np.array([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=np.int8)

    fig = create_regime_frequency_donut(regimes)
    pie_data = fig.data[0]

    assert len(pie_data.labels) == 4
    assert "Normal" in pie_data.labels
    assert "Recession" in pie_data.labels
    assert "Stagflation" in pie_data.labels
    assert "Depression" in pie_data.labels


def test_create_regime_frequency_donut_correct_counts():
    regimes = np.array([[0, 0, 1, 2]], dtype=np.int8)

    fig = create_regime_frequency_donut(regimes)
    pie_data = fig.data[0]

    assert list(pie_data.values) == [2, 1, 1, 0]


def test_create_regime_transition_heatmap_returns_figure():
    rng = np.random.default_rng(42)
    regimes = rng.integers(0, 4, size=(100, 30), dtype=np.int8)

    fig = create_regime_transition_heatmap(regimes)

    assert isinstance(fig, go.Figure)


def test_create_regime_transition_heatmap_shape():
    rng = np.random.default_rng(42)
    regimes = rng.integers(0, 4, size=(100, 30), dtype=np.int8)

    fig = create_regime_transition_heatmap(regimes)
    heatmap_data = fig.data[0]

    assert np.array(heatmap_data.z).shape == (4, 4)


def test_create_regime_transition_heatmap_rows_sum_to_one():
    rng = np.random.default_rng(42)
    regimes = rng.integers(0, 4, size=(1000, 100), dtype=np.int8)

    fig = create_regime_transition_heatmap(regimes)
    z_matrix = np.array(fig.data[0].z)

    row_sums = z_matrix.sum(axis=1)
    for row_sum in row_sums:
        assert np.isclose(row_sum, 1.0) or np.isclose(row_sum, 0.0)


def test_create_regime_transition_heatmap_deterministic():
    regimes = np.array([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=np.int8)

    fig = create_regime_transition_heatmap(regimes)
    z_matrix = np.array(fig.data[0].z)

    assert np.isclose(z_matrix[0, 1], 1.0)
    assert np.isclose(z_matrix[1, 2], 1.0)
    assert np.isclose(z_matrix[2, 3], 1.0)
    assert np.isclose(z_matrix[3, 0], 1.0)


def test_create_regime_transition_heatmap_single_year():
    regimes = np.array([[0]], dtype=np.int8)

    fig = create_regime_transition_heatmap(regimes)

    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) == 1
    assert "Insufficient" in fig.layout.annotations[0].text
