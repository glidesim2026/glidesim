# GlideSim

A vibe-coded Monte Carlo simulation tool for testing drawdown strategies. This application is for informational purposes only and does not provide financial advice. Personal finance is personal.

## Overview

This application simulates portfolio performance over a retirement horizon using Monte Carlo methods. The primary use case is evaluating and comparing different drawdown strategies to understand their impact on portfolio longevity and final wealth.

## Quick Start

```bash
# Install dependencies (including dev tools)
uv sync --all-extras --group dev

# Install pre-commit hooks
uv run pre-commit install
uv run pre-commit install --hook-type pre-push

# Run the app
uv run streamlit run src/glidesim/app.py

# Run tests
uv run pytest
```

## Development Setup

### Prerequisites

- Python 3.12+ (3.13 specified in `.python-version`)
- [uv](https://docs.astral.sh/uv/) package manager

### Pre-commit Hooks

This project uses pre-commit hooks for code quality:

- **Pre-commit**: Ruff linting (with auto-fix) and formatting
- **Pre-push**: Runs pytest to ensure tests pass before pushing

Hooks run automatically after installation. See `.pre-commit-config.yaml` for configuration.

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
    ├── registry.py                     # Strategy registry for UI
    ├── advanced_settings.py            # Market model parameter overrides
    └── tests/

scripts/
└── analyze_regime_distributions.py     # Market model analysis tool
```

## Testing

### Unit Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=glidesim --cov-report=term-missing
```

## Tech Stack

- **Python 3.12+**
- **numpy/scipy**: Numerical operations and statistics
- **Streamlit**: Web UI
- **plotly**: Interactive charts
- **PyYAML**: Configuration files
- **pytest**: Testing

## Contributing

Issues and PRs are welcome.

## License

MIT License - see [LICENSE.md](LICENSE.md)
