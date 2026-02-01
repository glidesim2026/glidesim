# GlideSim

A vibe-cded Monte Carlo simulation tool for testing drawdown strategies. This application is for information purposes only and does not provide financial advice. Personal finance is personal.

## Overview

This application simulates portfolio performance over a retirement horizon using Monte Carlo methods. The primary use case is evaluating and comparing different drawdown strategies to understand their impact on portfolio longevity and final wealth.

## Quick Start

```bash
# Install dependencies
uv sync --all-extras

# Run the app
uv run streamlit run src/glidesim/app.py

# Run tests
uv run pytest
```

## Features

### Simulation Engine
- Monte Carlo simulation with regime-switching market model
- Four market regimes: Normal, Recession, Stagflation, Depression
- Correlated stock/bond/inflation returns via Gaussian copula
- AR(1) dynamics with configurable persistence
- Configurable stock/bond allocation with optional rebalancing

### Drawdown Strategies
- **Inflation-Adjusted (4% Rule)**: Fixed real withdrawal adjusted for inflation
- **Fixed Withdrawal**: Constant nominal amount each year
- **Percent of Portfolio**: Fixed percentage of current portfolio value
- **Guardrails**: Inflation-adjusted with floor/ceiling adjustments
- **Percent of Portfolio with Minimum**: Variable percentage with inflation-adjusted floor

### Analysis & Visualization
- Portfolio failure rate (% of simulations exhausting funds)
- Final portfolio value distribution (percentiles)
- Interactive charts:
  - Success rate donut chart
  - Final balance histogram with percentile markers
  - Portfolio trajectory percentile bands
  - Individual run explorer (best/worst/median/random)
  - Return and inflation distributions by regime

## Configuration

Users can configure:
- Initial portfolio value
- Stock/bond allocation ratio
- Drawdown strategy and parameters
- Time horizon (years)
- Annual rebalancing (on/off)
- Number of simulation runs

## Project Structure

```
src/glidesim/
├── app.py                              # Streamlit entry point
├── conftest.py                         # Shared test fixtures
├── data/
│   └── market_params.yaml              # Regime-switching model parameters
├── simulation/
│   ├── engine.py                       # Core Monte Carlo loop
│   ├── market_model.py                 # Regime-switching market model
│   └── tests/
├── strategies/
│   ├── base.py                         # Strategy interface
│   ├── inflation_adjusted.py           # Classic 4% rule
│   ├── fixed.py                        # Fixed nominal withdrawal
│   ├── percent_of_portfolio.py         # Variable percentage
│   ├── percent_of_portfolio_with_minimum.py
│   ├── guardrails.py                   # Guardrails strategy
│   └── tests/
├── analysis/
│   ├── metrics.py                      # Success rate, percentiles
│   ├── plots.py                        # Visualization functions
│   └── tests/
└── ui/
    ├── sidebar.py                      # Input widgets and configuration
    ├── results.py                      # Results display and charts
    └── registry.py                     # Strategy registry for UI
```

## Testing

### Unit Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov=glidesim --cov-report=term-missing
```

### UI Testing

Manual verification:
```bash
uv run streamlit run src/glidesim/app.py
```

Check that all charts render correctly with expected data.

## Tech Stack

- **Python 3.12+**
- **numpy**: Numerical operations
- **Streamlit**: Web UI
- **plotly**: Interactive charts
- **pytest**: Testing
