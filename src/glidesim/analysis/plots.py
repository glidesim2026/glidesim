import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from glidesim.simulation.market_model import Regime

REGIME_COLORS = {
    Regime.NORMAL: "#2ecc71",
    Regime.RECESSION: "#e67e22",
    Regime.STAGFLATION: "#9b59b6",
    Regime.DEPRESSION: "#e74c3c",
}

REGIME_NAMES = {
    Regime.NORMAL: "Normal",
    Regime.RECESSION: "Recession",
    Regime.STAGFLATION: "Stagflation",
    Regime.DEPRESSION: "Depression",
}


def create_regime_frequency_donut(regimes: np.ndarray) -> go.Figure:
    """Create a donut chart showing empirical regime frequency distribution.

    Args:
        regimes: 2D array of regime indices with shape (n_simulations, n_years).

    Returns:
        Plotly Figure with donut chart colored by regime.
    """
    flat_regimes = regimes.flatten()
    counts = np.bincount(flat_regimes, minlength=len(Regime))

    labels = [REGIME_NAMES[r] for r in Regime]
    colors = [REGIME_COLORS[r] for r in Regime]

    fig = go.Figure(
        data=[
            go.Pie(
                values=counts,
                labels=labels,
                hole=0.5,
                marker_colors=colors,
                textinfo="percent",
                textposition="outside",
                domain=dict(x=[0.1, 0.9], y=[0.25, 0.95]),
            )
        ]
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.1,
            xanchor="center",
            x=0.6,
            entrywidthmode="fraction",
            entrywidth=0.45,
        ),
        margin=dict(t=10, b=10, l=10, r=10),
        height=350,
    )
    return fig


def create_regime_transition_heatmap(regimes: np.ndarray) -> go.Figure:
    """Create a heatmap showing empirical regime transition probabilities.

    Calculates the probability of transitioning from regime[year] to regime[year+1]
    across all simulations and years.

    Args:
        regimes: 2D array of regime indices with shape (n_simulations, n_years).

    Returns:
        Plotly Figure with annotated heatmap showing transition percentages.
    """
    if regimes.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for transitions",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(height=350)
        return fig

    n_regimes = len(Regime)
    counts = np.zeros((n_regimes, n_regimes), dtype=np.int64)

    from_regimes = regimes[:, :-1].flatten()
    to_regimes = regimes[:, 1:].flatten()
    np.add.at(counts, (from_regimes, to_regimes), 1)

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probabilities = counts / row_sums

    labels = [REGIME_NAMES[r] for r in Regime]
    annotations = [[f"{val:.0%}" for val in row] for row in probabilities]

    fig = go.Figure(
        data=go.Heatmap(
            z=probabilities,
            x=labels,
            y=labels,
            text=annotations,
            texttemplate="%{text}",
            textfont={"size": 12},
            colorscale="Blues",
            showscale=False,
            zmin=0,
            zmax=1,
        )
    )

    fig.update_layout(
        xaxis_title="To Regime",
        yaxis_title="From Regime",
        yaxis_autorange="reversed",
        margin=dict(t=30, b=40, l=80, r=20),
        height=350,
    )

    return fig


def create_success_donut(success_rate: float) -> go.Figure:
    """Create a donut chart showing simulation success/failure rates.

    Args:
        success_rate: Fraction of successful simulations (0.0 to 1.0).

    Returns:
        Plotly Figure with donut chart.
    """
    failure_rate = 1 - success_rate
    fig = go.Figure(
        data=[
            go.Pie(
                values=[success_rate, failure_rate],
                labels=["Success", "Failure"],
                hole=0.6,
                marker_colors=["#2ecc71", "#e74c3c"],
                textinfo="percent",
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=20, b=20, l=20, r=20),
        height=300,
    )
    return fig


def create_final_balance_histogram(
    portfolio_values: np.ndarray, percentiles: dict[int, float]
) -> go.Figure:
    """Create a histogram of final portfolio balances with percentile markers.

    Args:
        portfolio_values: Portfolio values array of shape (n_simulations, n_years + 1).
        percentiles: Dict mapping percentile numbers to dollar values.

    Returns:
        Plotly Figure with histogram and vertical percentile lines.
    """
    final_values = portfolio_values[:, -1]

    fig = px.histogram(
        x=final_values,
        nbins=50,
        labels={"x": "Final Portfolio Value ($)", "y": "Number of Simulations"},
    )
    fig.update_traces(marker_color="#3498db", showlegend=False)

    colors = {10: "#e74c3c", 25: "#e67e22", 50: "#2ecc71", 75: "#e67e22", 90: "#e74c3c"}
    for pct, value in percentiles.items():
        fig.add_trace(
            go.Scatter(
                x=[value, value],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color=colors.get(pct, "#95a5a6"), width=2),
                name=f"{pct}th: ${value:,.0f}",
                yaxis="y2",
            )
        )

    fig.update_layout(
        xaxis_tickformat="$,.0f",
        yaxis_title="Number of Simulations",
        yaxis2=dict(overlaying="y", range=[0, 1], visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=80, b=40),
        height=400,
    )
    return fig


def create_portfolio_trajectories(
    portfolio_values: np.ndarray, n_years: int
) -> go.Figure:
    """Create a line chart showing portfolio value percentile bands over time.

    Shows median trajectory with shaded bands for 10th-90th and 25th-75th percentiles.

    Args:
        portfolio_values: Portfolio values array of shape (n_simulations, n_years + 1).
        n_years: Number of simulation years.

    Returns:
        Plotly Figure with percentile bands and median line.
    """
    years = np.arange(n_years + 1)

    p10 = np.percentile(portfolio_values, 10, axis=0)
    p25 = np.percentile(portfolio_values, 25, axis=0)
    p50 = np.percentile(portfolio_values, 50, axis=0)
    p75 = np.percentile(portfolio_values, 75, axis=0)
    p90 = np.percentile(portfolio_values, 90, axis=0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=years,
            y=p90,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=p10,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(52, 152, 219, 0.2)",
            name="10th-90th percentile",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=p75,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=p25,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(52, 152, 219, 0.3)",
            name="25th-75th percentile",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=p50,
            mode="lines",
            line=dict(color="#2980b9", width=2),
            name="Median",
        )
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Portfolio Value ($)",
        yaxis_tickformat="$,.0f",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=60, b=40),
        height=450,
    )
    return fig


def create_distribution_histogram(
    data: np.ndarray,
    regimes: np.ndarray,
    regime_filter: Regime | None,
    color: str,
    x_label: str,
    title: str | None = None,
    format_as_percent: bool = True,
) -> go.Figure:
    """Create a histogram with optional regime filtering.

    Args:
        data: 2D array of shape (n_simulations, n_years).
        regimes: 2D array of regime indices, same shape as data.
        regime_filter: If provided, only include data from this regime.
        color: Histogram bar color.
        x_label: Label for x-axis.
        title: Optional chart title.
        format_as_percent: If True, format x-axis as percentages.

    Returns:
        Plotly Figure with histogram, mean line, and stats annotation.
    """
    if regime_filter is not None:
        mask = regimes == regime_filter
        flat_data = data[mask]
    else:
        flat_data = data.flatten()

    if len(flat_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data for selected regime",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(height=300)
        return fig

    mean_val = np.mean(flat_data)
    std_val = np.std(flat_data)
    n_samples = len(flat_data)

    fig = px.histogram(
        x=flat_data, nbins=50, labels={"x": x_label, "y": "Number of Simulations"}
    )
    fig.update_traces(marker_color=color)

    if format_as_percent:
        annotation_text = (
            f"Mean: {mean_val:.1%}<br>Std: {std_val:.1%}<br>n={n_samples:,}"
        )
        fig.update_layout(xaxis_tickformat=".0%")
    else:
        annotation_text = (
            f"Mean: {mean_val:.2f}<br>Std: {std_val:.2f}<br>n={n_samples:,}"
        )

    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="#2c3e50",
        annotation_text=annotation_text,
        annotation_position="top right",
    )

    fig.update_layout(
        yaxis_title="Number of Simulations", margin=dict(t=60, b=40), height=300
    )

    if title:
        fig.add_annotation(
            text=f"<b>{title}</b>",
            xref="x domain",
            yref="paper",
            x=0.5,
            y=1.15,
            showarrow=False,
            font=dict(size=14),
            xanchor="center",
        )

    return fig


def create_individual_trajectory(
    portfolio_values: np.ndarray,
    returns: np.ndarray,
    inflation: np.ndarray,
    withdrawals: np.ndarray,
    regimes: np.ndarray,
    sim_index: int,
    n_years: int,
) -> go.Figure:
    """Create a chart showing a single simulation trajectory with withdrawal bars.

    The top subplot shows the portfolio trajectory as connected line segments,
    each colored by the regime active during that year. The bottom subplot shows
    annual withdrawals as bars, also colored by regime.

    Args:
        portfolio_values: Portfolio values array of shape (n_simulations, n_years + 1).
        returns: Returns array of shape (n_simulations, n_years).
        inflation: Inflation array of shape (n_simulations, n_years).
        withdrawals: Withdrawals array of shape (n_simulations, n_years).
        regimes: Regime indices array of shape (n_simulations, n_years).
        sim_index: Index of the simulation to display.
        n_years: Number of simulation years.

    Returns:
        Plotly Figure with regime-colored trajectory and withdrawal bars.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.08,
        subplot_titles=("Portfolio Value", "Annual Withdrawals"),
    )

    values = portfolio_values[sim_index]
    sim_returns = returns[sim_index]
    sim_inflation = inflation[sim_index]
    sim_withdrawals = withdrawals[sim_index]
    sim_regimes = regimes[sim_index]

    legend_shown = set()

    for year in range(n_years):
        regime = Regime(sim_regimes[year])
        color = REGIME_COLORS[regime]
        regime_name = REGIME_NAMES[regime]
        show_legend = regime not in legend_shown

        x_vals = [year, year + 1]
        y_vals = [values[year], values[year + 1]]

        hover_text = (
            f"<b>Year {year + 1}</b><br>"
            f"Portfolio: ${values[year + 1]:,.0f}<br>"
            f"Return: {sim_returns[year]:.1%}<br>"
            f"Inflation: {sim_inflation[year]:.1%}<br>"
            f"Withdrawal: ${sim_withdrawals[year]:,.0f}<br>"
            f"Regime: {regime_name}"
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(color=color, size=6),
                name=regime_name,
                legendgroup=regime_name,
                showlegend=show_legend,
                hoverinfo="text",
                hovertext=[hover_text, hover_text],
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=[year + 1],
                y=[sim_withdrawals[year]],
                marker_color=color,
                opacity=0.7,
                name=regime_name,
                legendgroup=regime_name,
                showlegend=False,
                hovertemplate=(
                    f"<b>Year {year + 1}</b><br>"
                    f"Withdrawal: ${sim_withdrawals[year]:,.0f}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

        if show_legend:
            legend_shown.add(regime)

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[values[0]],
            mode="markers",
            marker=dict(color="#2c3e50", size=8),
            name="Start",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        height=550,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=100, b=40),
    )
    fig.update_yaxes(title_text="Portfolio Value ($)", tickformat="$,.0f", row=1, col=1)
    fig.update_yaxes(title_text="Withdrawal ($)", tickformat="$,.0f", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)

    return fig
