# GlideSim

A Monte Carlo simulator focused on testing drawdown strategies.

## Project Scope

- **Primary Goal**: Test and compare drawdown strategies in a simulated environment
- **Simulation**: Monte Carlo with correlated year-over-year returns
- **Portfolio**: Configurable stocks/bonds allocation (X/Y split)
- **Spending**: Annual drawdown with selectable strategies (start with inflation-adjusted, extensible for future strategies)
- **Rebalancing**: Simple on/off toggle for annual rebalancing to target allocation
- **No tax modeling**: All values are gross amounts

## Key Metrics

- Portfolio failure rate (probability of running out of money)
- Final portfolio values at various percentiles

## Interface

- Locally hosted web app
- Basic visualizations

## Out of Scope

- Income modeling (Social Security, pensions, etc.)
- Tax calculations
- Session persistence
- External data imports

## Testing

### Unit Tests

Run all tests:
```bash
uv run pytest
```

Run with verbose output:
```bash
uv run pytest -v
```

Run with coverage:
```bash
uv run pytest --cov=glidesim --cov-report=term-missing
```

Test files follow the pattern `test_*.py` and are located alongside the modules they test in `tests/` subdirectories.

**Coverage**: 99% on non-UI code. The `ui/` directory and `app.py` are excluded from coverage requirements (tested manually).

### UI Testing

UI testing is manual. Run the app and verify charts render correctly:
```bash
uv run streamlit run src/glidesim/app.py
```

#### Using Playwright MCP

For automated visual verification during development, use the Playwright MCP tools. Always close the browser at the end of each session.

**Pre-flight cleanup:**

Before starting, check for and clean up any existing sessions:

1. Check if Streamlit is already running:
   ```bash
   lsof -i :8501
   ```

2. Check if Chrome windows are open from previous Playwright sessions:
   ```bash
   pgrep -f "Chrome.*--remote-debugging"
   ```

3. If either is running, clean up before proceeding:
   - Close the Playwright browser: `mcp__playwright__browser_close`
   - Kill the Streamlit server: `pkill -f "streamlit run"`

**Workflow:**

1. Start the app in background (if not already running):
   ```bash
   uv run streamlit run src/glidesim/app.py --server.headless=true &
   ```

2. Navigate to the app (always do this before waiting):
   - `mcp__playwright__browser_navigate` with url `http://localhost:8501`

3. Wait for Streamlit to fully load:
   - `mcp__playwright__browser_wait_for` with time `5` (seconds)

4. Get page snapshot (preferred over screenshot):
   - `mcp__playwright__browser_snapshot` returns accessible element tree with refs

5. Interact with elements using refs from the snapshot:
   - `mcp__playwright__browser_click` with ref from snapshot
   - `mcp__playwright__browser_type` for text input

6. **Always close the browser when done** (prevents issues in future sessions):
   - `mcp__playwright__browser_close`

**Re-testing after code changes:**
- The browser persists between operations within a session - just call `browser_navigate` again to reload
- No need to close and reopen between tests in the same session

**Troubleshooting:**
- If you get "Opening in existing browser session" errors, manually close all Chrome windows and retry
- If two windows appear, ensure you're using `--server.headless=true` to prevent Streamlit from auto-opening a browser

### Writing New Tests

- Place test files in `tests/` subdirectory next to the module being tested
- Use `np.random.default_rng(seed)` for deterministic tests
- Use `np.testing.assert_array_equal` for numpy array comparisons
- Use `np.isclose` for floating point comparisons
- Shared fixtures are available in `conftest.py`:
  - `make_config` - factory fixture for SimulationConfig
  - `make_results` - factory fixture for SimulationResults

## Analysis Scripts

### Regime Distribution Analysis

Analyze the return distributions produced by the market model for each regime:

```bash
uv run python scripts/analyze_regime_distributions.py
```

Options:
- `--n-simulations N`: Number of simulations (default: 5000)
- `--n-years N`: Years per simulation (default: 100)
- `--seed N`: Random seed for reproducibility (default: 42)
- `--output FILE`: Save results to JSON file
- `--plots`: Generate histogram plots (requires matplotlib)
- `--plot-dir DIR`: Directory for plot output

Examples:
```bash
# Quick analysis
uv run python scripts/analyze_regime_distributions.py --n-simulations 1000

# Full analysis with JSON output
uv run python scripts/analyze_regime_distributions.py --output results.json

# Generate plots (install matplotlib first: uv add matplotlib --dev)
uv run python scripts/analyze_regime_distributions.py --plots --plot-dir ./plots
```

Output includes per-regime statistics for stocks, bonds, and inflation:
- Count, mean, std, min, max
- Percentiles (1st, 5th, 25th, 50th, 75th, 95th, 99th)
- Skewness and kurtosis

## Code Style

### app.py

Treat `app.py` as a minimal main script. It should only:
- Set page config and title
- Wire together UI components from modules
- Manage top-level session state

All substantive logic should live in modules:
- `ui/sidebar.py` - input widgets and configuration
- `ui/results.py` - results display and visualizations
- `ui/registry.py` - strategy registry and parameter widgets
- `simulation/` - simulation logic
- `analysis/` - metrics and plotting
- `strategies/` - withdrawal strategy implementations

### Adding New Strategies

Withdrawal strategies use a registry pattern. To add a new strategy:

1. Create a new strategy class in `strategies/` that inherits from `Strategy`
2. Add an entry to `STRATEGY_REGISTRY` in `ui/registry.py` with:
   - The strategy class
   - A `render_params` function that renders Streamlit widgets and returns kwargs
