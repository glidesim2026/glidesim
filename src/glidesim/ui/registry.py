"""Strategy registry for UI parameter rendering.

This module provides a registry pattern for withdrawal strategies, decoupling
the UI from individual strategy implementations.
"""

from dataclasses import dataclass
from typing import Any, Callable, Type

import streamlit as st

from glidesim.strategies.base import Strategy
from glidesim.strategies.fixed import FixedWithdrawalStrategy
from glidesim.strategies.guardrails import GuardrailsStrategy
from glidesim.strategies.inflation_adjusted import InflationAdjustedStrategy
from glidesim.strategies.percent_of_portfolio_with_minimum import (
    PercentOfPortfolioWithMinimumStrategy,
)
from glidesim.strategies.percent_of_portfolio import (
    PercentOfPortfolioStrategy,
)


@dataclass
class StrategyConfig:
    """Configuration for a withdrawal strategy in the registry.

    Attributes:
        strategy_class: The Strategy subclass to instantiate.
        render_params: Function that renders Streamlit widgets and returns kwargs
            for the strategy constructor.
    """

    strategy_class: Type[Strategy]
    render_params: Callable[[Any], dict]


def _render_inflation_adjusted_params(defaults) -> dict:
    """Render parameters for InflationAdjustedStrategy."""
    initial_withdrawal = st.number_input(
        "Initial Annual Withdrawal ($)",
        min_value=10_000,
        max_value=200_000,
        value=defaults.initial_withdrawal,
        step=5_000,
        format="%d",
    )
    return {"initial_withdrawal": initial_withdrawal}


def _render_fixed_params(defaults) -> dict:
    """Render parameters for FixedWithdrawalStrategy."""
    withdrawal_amount = st.number_input(
        "Annual Withdrawal Amount ($)",
        min_value=10_000,
        max_value=200_000,
        value=defaults.initial_withdrawal,
        step=5_000,
        format="%d",
    )
    return {"withdrawal_amount": withdrawal_amount}


def _render_percent_of_portfolio_params(defaults) -> dict:
    """Render parameters for PercentOfPortfolioStrategy."""
    withdrawal_rate = st.slider(
        "Withdrawal Rate (%)",
        min_value=1,
        max_value=10,
        value=defaults.withdrawal_rate,
    )
    return {"withdrawal_rate": withdrawal_rate / 100}


def _render_guardrails_params(defaults) -> dict:
    """Render parameters for GuardrailsStrategy."""
    initial_withdrawal = st.number_input(
        "Initial Annual Withdrawal ($)",
        min_value=10_000,
        max_value=200_000,
        value=defaults.initial_withdrawal,
        step=5_000,
        format="%d",
    )

    col1, col2 = st.columns(2)
    with col1:
        floor_rate = st.slider(
            "Lower Guardrail (%)",
            min_value=1,
            max_value=8,
            value=defaults.floor_rate,
            help="When your portfolio grows enough that you're withdrawing less "
            "than this % of its value, the strategy increases your spending.",
        )

    with col2:
        ceiling_rate = st.slider(
            "Upper Guardrail (%)",
            min_value=3,
            max_value=15,
            value=defaults.ceiling_rate,
            help="When your portfolio drops enough that you're withdrawing more "
            "than this % of its value, the strategy reduces your spending.",
        )

    if floor_rate >= ceiling_rate:
        st.error("Lower guardrail must be less than upper guardrail.")

    adjustment_rate = st.slider(
        "Adjustment Amount (%)",
        min_value=1,
        max_value=25,
        value=defaults.adjustment_rate,
        help="The percentage by which your withdrawal changes when a guardrail "
        "is triggered.",
    )
    return {
        "initial_withdrawal": initial_withdrawal,
        "floor_rate": floor_rate / 100,
        "ceiling_rate": ceiling_rate / 100,
        "adjustment_rate": adjustment_rate / 100,
    }


def _render_percent_of_portfolio_with_minimum_params(defaults) -> dict:
    """Render parameters for PercentOfPortfolioWithMinimumStrategy."""
    withdrawal_rate = st.slider(
        "Withdrawal Rate (%)",
        min_value=1,
        max_value=10,
        value=defaults.withdrawal_rate,
        help="Percentage of current portfolio to withdraw each year",
    )
    minimum_withdrawal = st.number_input(
        "Minimum Withdrawal ($)",
        min_value=10_000,
        max_value=200_000,
        value=defaults.initial_withdrawal,
        step=5_000,
        format="%d",
        help="Floor amount to withdraw regardless of portfolio value",
    )
    inflation_adjustment = st.checkbox(
        "Inflation-Adjust Minimum",
        value=True,
        help="Increase the minimum withdrawal with inflation over time",
    )
    years_at_minimum = st.slider(
        "Years at Minimum Before Switching",
        min_value=0,
        max_value=10,
        value=defaults.years_at_minimum,
        help="Number of years to use only the minimum before switching to "
        "percentage-based withdrawals (helps mitigate sequence of returns risk)",
    )
    return {
        "withdrawal_rate": withdrawal_rate / 100,
        "minimum_withdrawal": minimum_withdrawal,
        "inflation_adjustment": inflation_adjustment,
        "years_at_minimum": years_at_minimum,
    }


STRATEGY_REGISTRY: dict[str, StrategyConfig] = {
    "Inflation-Adjusted (4% Rule)": StrategyConfig(
        strategy_class=InflationAdjustedStrategy,
        render_params=_render_inflation_adjusted_params,
    ),
    "Fixed Withdrawal": StrategyConfig(
        strategy_class=FixedWithdrawalStrategy,
        render_params=_render_fixed_params,
    ),
    "Percent of Portfolio": StrategyConfig(
        strategy_class=PercentOfPortfolioStrategy,
        render_params=_render_percent_of_portfolio_params,
    ),
    "Guardrails": StrategyConfig(
        strategy_class=GuardrailsStrategy,
        render_params=_render_guardrails_params,
    ),
    "Percent of Portfolio with Minimum": StrategyConfig(
        strategy_class=PercentOfPortfolioWithMinimumStrategy,
        render_params=_render_percent_of_portfolio_with_minimum_params,
    ),
}
