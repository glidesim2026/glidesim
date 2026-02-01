# GlideSim

Monte Carlo simulator for testing retirement drawdown strategies. Streamlit web app with regime-switching market model.

## Commands

```bash
uv run streamlit run src/glidesim/app.py  # Run app
uv run pytest                              # Run tests
uv run pytest --cov=glidesim               # Run with coverage
```

### Using Playwright MCP

For automated visual verification during development, use the Playwright MCP tools. Always close the browser at the end of each session.

**Pre-flight cleanup:**

Before doing any new round of testing, check for and clean up any existing sessions:

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


## Testing

- Tests live in `tests/` subdirectories alongside modules
- Use `np.random.default_rng(seed)` for deterministic tests
- Fixtures in `conftest.py`: `make_config`, `make_results`
- Coverage target: 99% on non-UI code (`ui/` and `app.py` excluded)

## Architecture

`app.py` is a thin entrypoint. All logic lives in modules:
- `simulation/` - Monte Carlo engine and market model
- `strategies/` - Withdrawal strategy implementations
- `analysis/` - Metrics and plotting
- `ui/` - Streamlit widgets and results display
