from pathlib import Path

import numpy as np
import streamlit as st

from glidesim.simulation.market_model import (
    MarketModelParams,
    RegimeParams,
)

_MARKET_PARAMS_PATH = Path(__file__).parent.parent / "data" / "market_params.yaml"

REGIMES = ["normal", "recession", "stagflation", "depression"]

PARAM_HELP = {
    "warm_up_years": "Years to run before simulation starts. Establishes realistic AR dynamics.",
    "max_sigma": "Clamp extreme values to this many standard deviations.",
    "stock_mean": "Expected annual stock return (e.g., 0.10 = 10%)",
    "bond_mean": "Expected annual bond return",
    "inflation_mean": "Expected annual inflation rate",
    "stock_std": "Annualized volatility. Higher = more variable returns.",
    "bond_std": "Bond return volatility",
    "inflation_std": "Inflation rate volatility",
    "stock_bond_corr": "How returns move together. Negative = diversification benefit.",
    "stock_inflation_corr": "Relationship between stocks and inflation",
    "bond_inflation_corr": "Bond-inflation relationship. Often negative.",
    "stock_ar": "Return persistence. Positive = momentum, negative = mean reversion.",
    "bond_ar": "Bond return persistence",
    "inflation_ar": "Inflation stickiness. Higher = more persistent inflation.",
    "stock_skewness": "Distribution asymmetry. Negative = larger crashes than rallies.",
    "bond_skewness": "Bond return asymmetry",
    "inflation_skewness": "Inflation distribution asymmetry",
    "stock_model": "'log' = log-normal (can't go below -100%), 'direct' = normal",
    "inflation_model": "'log' = log-normal, 'direct' = normal (allows deflation)",
}


def load_default_market_params() -> MarketModelParams:
    """Load default market parameters from YAML, cached in session state."""
    if "default_market_params" not in st.session_state:
        st.session_state.default_market_params = MarketModelParams.from_yaml(
            _MARKET_PARAMS_PATH
        )
    return st.session_state.default_market_params


def _render_global_params() -> tuple[int, float]:
    """Render global parameter widgets."""
    with st.expander("Global Parameters", expanded=False):
        warm_up_years = st.slider(
            "Warm-up Years",
            min_value=0,
            max_value=20,
            key="market_param_warm_up_years",
            help=PARAM_HELP["warm_up_years"],
        )

        use_clamping = st.checkbox(
            "Sigma clamping",
            key="market_param_use_sigma_clamping",
            help="When enabled, clamps extreme values to limit tail events.",
        )

        if use_clamping:
            max_sigma = st.number_input(
                "Max Sigma",
                min_value=1.0,
                max_value=10.0,
                step=0.5,
                key="market_param_max_sigma",
                help=PARAM_HELP["max_sigma"],
            )
        else:
            max_sigma = float("inf")

    return warm_up_years, max_sigma


def _render_transition_matrix() -> np.ndarray:
    """Render transition matrix widgets."""
    with st.expander("Regime Transition Probabilities", expanded=False):
        st.caption(
            "Probability of transitioning from row regime to column regime. "
            "Each row must sum to 1.0."
        )

        matrix = np.zeros((4, 4))

        cols = st.columns(5)
        cols[0].write("")
        for j, to_regime in enumerate(REGIMES):
            cols[j + 1].write(f"**{to_regime[:3].title()}**")

        for i, from_regime in enumerate(REGIMES):
            cols = st.columns(5)
            cols[0].write(f"**{from_regime.title()}**")
            for j, to_regime in enumerate(REGIMES):
                matrix[i, j] = cols[j + 1].number_input(
                    f"{from_regime} to {to_regime}",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    format="%.2f",
                    key=f"market_param_transition_{from_regime}_{to_regime}",
                    label_visibility="collapsed",
                )

        row_sums = matrix.sum(axis=1)
        invalid_rows = []
        for i, (regime, row_sum) in enumerate(zip(REGIMES, row_sums)):
            if not np.isclose(row_sum, 1.0, atol=0.01):
                invalid_rows.append(f"{regime.title()} ({row_sum:.2f})")

        if invalid_rows:
            st.error(f"Row sums must equal 1.0. Invalid: {', '.join(invalid_rows)}")

    return matrix


def _render_regime_params(regime: str) -> RegimeParams:
    """Render parameter widgets for a single regime."""
    with st.expander("Returns (Means)", expanded=True):
        col1, col2, col3 = st.columns(3)
        stock_mean = col1.number_input(
            "Stock Mean",
            min_value=-0.50,
            max_value=0.30,
            step=0.01,
            format="%.2f",
            key=f"market_param_{regime}_stock_mean",
            help=PARAM_HELP["stock_mean"],
        )
        bond_mean = col2.number_input(
            "Bond Mean",
            min_value=-0.20,
            max_value=0.20,
            step=0.01,
            format="%.2f",
            key=f"market_param_{regime}_bond_mean",
            help=PARAM_HELP["bond_mean"],
        )
        inflation_mean = col3.number_input(
            "Inflation Mean",
            min_value=-0.10,
            max_value=0.20,
            step=0.01,
            format="%.3f",
            key=f"market_param_{regime}_inflation_mean",
            help=PARAM_HELP["inflation_mean"],
        )

    with st.expander("Volatility (Std Devs)", expanded=False):
        col1, col2, col3 = st.columns(3)
        stock_std = col1.number_input(
            "Stock Std Dev",
            min_value=0.01,
            max_value=0.50,
            step=0.01,
            format="%.2f",
            key=f"market_param_{regime}_stock_std",
            help=PARAM_HELP["stock_std"],
        )
        bond_std = col2.number_input(
            "Bond Std Dev",
            min_value=0.01,
            max_value=0.30,
            step=0.01,
            format="%.2f",
            key=f"market_param_{regime}_bond_std",
            help=PARAM_HELP["bond_std"],
        )
        inflation_std = col3.number_input(
            "Inflation Std Dev",
            min_value=0.005,
            max_value=0.10,
            step=0.005,
            format="%.3f",
            key=f"market_param_{regime}_inflation_std",
            help=PARAM_HELP["inflation_std"],
        )

    with st.expander("Correlations", expanded=False):
        col1, col2, col3 = st.columns(3)
        stock_bond_corr = col1.slider(
            "Stock-Bond",
            min_value=-1.0,
            max_value=1.0,
            step=0.05,
            key=f"market_param_{regime}_stock_bond_corr",
            help=PARAM_HELP["stock_bond_corr"],
        )
        stock_inflation_corr = col2.slider(
            "Stock-Inflation",
            min_value=-1.0,
            max_value=1.0,
            step=0.05,
            key=f"market_param_{regime}_stock_inflation_corr",
            help=PARAM_HELP["stock_inflation_corr"],
        )
        bond_inflation_corr = col3.slider(
            "Bond-Inflation",
            min_value=-1.0,
            max_value=1.0,
            step=0.05,
            key=f"market_param_{regime}_bond_inflation_corr",
            help=PARAM_HELP["bond_inflation_corr"],
        )

    with st.expander("Persistence (AR Coefficients)", expanded=False):
        col1, col2, col3 = st.columns(3)
        stock_ar = col1.slider(
            "Stock AR",
            min_value=-0.5,
            max_value=0.5,
            step=0.05,
            key=f"market_param_{regime}_stock_ar",
            help=PARAM_HELP["stock_ar"],
        )
        bond_ar = col2.slider(
            "Bond AR",
            min_value=-0.5,
            max_value=0.5,
            step=0.05,
            key=f"market_param_{regime}_bond_ar",
            help=PARAM_HELP["bond_ar"],
        )
        inflation_ar = col3.slider(
            "Inflation AR",
            min_value=-0.5,
            max_value=1.0,
            step=0.05,
            key=f"market_param_{regime}_inflation_ar",
            help=PARAM_HELP["inflation_ar"],
        )

    with st.expander("Skewness", expanded=False):
        col1, col2, col3 = st.columns(3)
        stock_skewness = col1.slider(
            "Stock Skew",
            min_value=-10.0,
            max_value=10.0,
            step=0.5,
            key=f"market_param_{regime}_stock_skewness",
            help=PARAM_HELP["stock_skewness"],
        )
        bond_skewness = col2.slider(
            "Bond Skew",
            min_value=-10.0,
            max_value=10.0,
            step=0.5,
            key=f"market_param_{regime}_bond_skewness",
            help=PARAM_HELP["bond_skewness"],
        )
        inflation_skewness = col3.slider(
            "Inflation Skew",
            min_value=-10.0,
            max_value=10.0,
            step=0.5,
            key=f"market_param_{regime}_inflation_skewness",
            help=PARAM_HELP["inflation_skewness"],
        )

    with st.expander("Model Type", expanded=False):
        col1, col2 = st.columns(2)
        stock_model = col1.selectbox(
            "Stock Model",
            options=["log", "direct"],
            key=f"market_param_{regime}_stock_model",
            help=PARAM_HELP["stock_model"],
        )
        inflation_model = col2.selectbox(
            "Inflation Model",
            options=["log", "direct"],
            key=f"market_param_{regime}_inflation_model",
            help=PARAM_HELP["inflation_model"],
        )

    return RegimeParams(
        stock_mean=stock_mean,
        bond_mean=bond_mean,
        inflation_mean=inflation_mean,
        stock_std=stock_std,
        bond_std=bond_std,
        inflation_std=inflation_std,
        stock_ar=stock_ar,
        bond_ar=bond_ar,
        inflation_ar=inflation_ar,
        stock_bond_corr=stock_bond_corr,
        stock_inflation_corr=stock_inflation_corr,
        bond_inflation_corr=bond_inflation_corr,
        stock_skewness=stock_skewness,
        bond_skewness=bond_skewness,
        inflation_skewness=inflation_skewness,
        stock_model=stock_model,
        inflation_model=inflation_model,
    )


def _check_transition_matrix_valid(matrix: np.ndarray) -> bool:
    """Check if transition matrix rows sum to 1.0."""
    row_sums = matrix.sum(axis=1)
    return all(np.isclose(row_sum, 1.0, atol=0.01) for row_sum in row_sums)


def _reset_market_params():
    """Callback to reset all market params to defaults by deleting session state keys."""
    for key in list(st.session_state.keys()):
        if key.startswith("market_param_"):
            del st.session_state[key]


def _ensure_market_params_initialized():
    """Initialize market params in session state if not already set.

    This must be called BEFORE widgets are created to avoid conflicts between
    the widget's value parameter and session state.
    """
    defaults = load_default_market_params()

    # Global params
    if "market_param_warm_up_years" not in st.session_state:
        st.session_state["market_param_warm_up_years"] = defaults.warm_up_years
    if "market_param_use_sigma_clamping" not in st.session_state:
        st.session_state["market_param_use_sigma_clamping"] = False
    if "market_param_max_sigma" not in st.session_state:
        st.session_state["market_param_max_sigma"] = 3.0

    # Transition matrix
    for i, from_regime in enumerate(REGIMES):
        for j, to_regime in enumerate(REGIMES):
            key = f"market_param_transition_{from_regime}_{to_regime}"
            if key not in st.session_state:
                st.session_state[key] = float(defaults.transition_matrix[i, j])

    # Regime params
    regime_params = [
        "stock_mean",
        "bond_mean",
        "inflation_mean",
        "stock_std",
        "bond_std",
        "inflation_std",
        "stock_bond_corr",
        "stock_inflation_corr",
        "bond_inflation_corr",
        "stock_ar",
        "bond_ar",
        "inflation_ar",
        "stock_skewness",
        "bond_skewness",
        "inflation_skewness",
        "stock_model",
        "inflation_model",
    ]
    for regime in REGIMES:
        regime_defaults = getattr(defaults, regime)
        for param in regime_params:
            key = f"market_param_{regime}_{param}"
            if key not in st.session_state:
                st.session_state[key] = getattr(regime_defaults, param)


def render_advanced_settings() -> MarketModelParams | None:
    """Render advanced market settings expander.

    Returns:
        MarketModelParams with user-configured values, or None if validation fails.
    """
    _ensure_market_params_initialized()

    with st.expander("Advanced Market Settings", expanded=False):
        st.caption(
            "These settings control the market simulation model. "
            "Only modify if you understand the implications."
        )

        st.button(
            "Reset to Defaults",
            key="reset_market_params",
            on_click=_reset_market_params,
        )

        warm_up_years, max_sigma = _render_global_params()
        transition_matrix = _render_transition_matrix()

        tabs = st.tabs([r.title() for r in REGIMES])
        regime_params = {}
        for tab, regime in zip(tabs, REGIMES):
            with tab:
                regime_params[regime] = _render_regime_params(regime)

        if not _check_transition_matrix_valid(transition_matrix):
            return None

        return MarketModelParams(
            normal=regime_params["normal"],
            recession=regime_params["recession"],
            stagflation=regime_params["stagflation"],
            depression=regime_params["depression"],
            transition_matrix=transition_matrix,
            warm_up_years=warm_up_years,
            max_sigma=max_sigma,
        )
