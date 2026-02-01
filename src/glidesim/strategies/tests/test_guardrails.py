import numpy as np
import pytest

from glidesim.strategies.guardrails import GuardrailsStrategy


def test_within_guardrails_returns_inflation_adjusted():
    strategy = GuardrailsStrategy(
        initial_withdrawal=40_000,
        floor_rate=0.03,
        ceiling_rate=0.07,
    )
    # 4% rate is within 3%-7% guardrails
    withdrawal = strategy.get_withdrawal(1_000_000, 1.0, 0)
    assert withdrawal == 40_000


def test_ceiling_guardrail_triggers_cut():
    strategy = GuardrailsStrategy(
        initial_withdrawal=40_000,
        floor_rate=0.03,
        ceiling_rate=0.05,
        adjustment_rate=0.10,
    )
    # Portfolio dropped, rate = 40k/500k = 8% > 5% ceiling
    withdrawal = strategy.get_withdrawal(500_000, 1.0, 0)
    # Should cut by 10%: 40_000 * 0.9 = 36_000
    assert withdrawal == 36_000


def test_floor_guardrail_triggers_raise():
    strategy = GuardrailsStrategy(
        initial_withdrawal=40_000,
        floor_rate=0.04,
        ceiling_rate=0.06,
        adjustment_rate=0.10,
    )
    # Portfolio grew, rate = 40k/2M = 2% < 4% floor
    withdrawal = strategy.get_withdrawal(2_000_000, 1.0, 0)
    # Should raise by 10%: 40_000 * 1.1 = 44_000
    assert withdrawal == 44_000


def test_inflation_adjustment_applied():
    strategy = GuardrailsStrategy(
        initial_withdrawal=40_000,
        floor_rate=0.03,
        ceiling_rate=0.07,
    )
    # With 50% cumulative inflation, base withdrawal = 60k
    # Rate = 60k/1.5M = 4% (within guardrails)
    withdrawal = strategy.get_withdrawal(1_500_000, 1.5, 0)
    assert withdrawal == 60_000


def test_ceiling_with_inflation():
    strategy = GuardrailsStrategy(
        initial_withdrawal=40_000,
        floor_rate=0.03,
        ceiling_rate=0.05,
        adjustment_rate=0.10,
    )
    # Base = 40k * 1.5 = 60k, rate = 60k/800k = 7.5% > 5%
    withdrawal = strategy.get_withdrawal(800_000, 1.5, 0)
    # Cut by 10%: 60k * 0.9 = 54k
    assert withdrawal == 54_000


def test_handles_zero_portfolio():
    strategy = GuardrailsStrategy(initial_withdrawal=40_000)
    assert strategy.get_withdrawal(0, 1.5, 0) == 0


def test_caps_withdrawal_at_portfolio_value():
    strategy = GuardrailsStrategy(
        initial_withdrawal=40_000,
        floor_rate=0.03,
        ceiling_rate=0.07,
    )
    # Even though base withdrawal = 40k, portfolio only has 10k
    assert strategy.get_withdrawal(10_000, 1.0, 0) == 10_000


def test_recovery_when_portfolio_recovers():
    strategy = GuardrailsStrategy(
        initial_withdrawal=40_000,
        floor_rate=0.03,
        ceiling_rate=0.05,
        adjustment_rate=0.10,
    )
    # Year 1: portfolio crashed, ceiling triggered
    withdrawal_bad = strategy.get_withdrawal(500_000, 1.03, 0)
    assert np.isclose(withdrawal_bad, 40_000 * 1.03 * 0.9)

    # Year 2: portfolio recovered, back within guardrails
    withdrawal_good = strategy.get_withdrawal(1_500_000, 1.06, 1)
    # Rate = 42.4k / 1.5M = 2.83% < 3% floor, so raise by 10%
    base = 40_000 * 1.06
    assert np.isclose(withdrawal_good, base * 1.1)


def test_floor_must_be_less_than_ceiling():
    with pytest.raises(ValueError, match="floor_rate.*must be less than ceiling_rate"):
        GuardrailsStrategy(
            initial_withdrawal=40_000, floor_rate=0.06, ceiling_rate=0.04
        )


def test_floor_cannot_equal_ceiling():
    with pytest.raises(ValueError, match="floor_rate.*must be less than ceiling_rate"):
        GuardrailsStrategy(
            initial_withdrawal=40_000, floor_rate=0.05, ceiling_rate=0.05
        )
