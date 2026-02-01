from glidesim.strategies.inflation_adjusted import InflationAdjustedStrategy


def test_initial_withdrawal_with_no_inflation():
    strategy = InflationAdjustedStrategy(initial_withdrawal=40_000)

    withdrawal = strategy.get_withdrawal(
        portfolio_value=1_000_000, cumulative_inflation=1.0, year=0
    )

    assert withdrawal == 40_000


def test_withdrawal_scales_with_inflation():
    strategy = InflationAdjustedStrategy(initial_withdrawal=40_000)

    withdrawal = strategy.get_withdrawal(
        portfolio_value=1_000_000, cumulative_inflation=1.5, year=0
    )

    assert withdrawal == 60_000


def test_withdrawal_independent_of_portfolio_value():
    strategy = InflationAdjustedStrategy(initial_withdrawal=40_000)

    withdrawal_high = strategy.get_withdrawal(
        portfolio_value=2_000_000, cumulative_inflation=1.0, year=0
    )
    withdrawal_low = strategy.get_withdrawal(
        portfolio_value=100_000, cumulative_inflation=1.0, year=0
    )

    assert withdrawal_high == withdrawal_low == 40_000


def test_caps_withdrawal_at_portfolio_value():
    strategy = InflationAdjustedStrategy(initial_withdrawal=40_000)
    assert strategy.get_withdrawal(20_000, 1.0, 0) == 20_000


def test_returns_zero_when_portfolio_empty():
    strategy = InflationAdjustedStrategy(initial_withdrawal=40_000)
    assert strategy.get_withdrawal(0, 1.0, 0) == 0
