from glidesim.strategies.percent_of_portfolio import (
    PercentOfPortfolioStrategy,
)


def test_calculates_percentage_of_portfolio():
    strategy = PercentOfPortfolioStrategy(withdrawal_rate=0.04)
    assert strategy.get_withdrawal(1_000_000, 1.0, 0) == 40_000


def test_scales_with_portfolio_value():
    strategy = PercentOfPortfolioStrategy(withdrawal_rate=0.04)
    assert strategy.get_withdrawal(2_000_000, 1.5, 0) == 80_000
    assert strategy.get_withdrawal(500_000, 1.5, 0) == 20_000


def test_ignores_inflation():
    strategy = PercentOfPortfolioStrategy(withdrawal_rate=0.04)
    assert strategy.get_withdrawal(1_000_000, 1.0, 0) == 40_000
    assert strategy.get_withdrawal(1_000_000, 2.0, 0) == 40_000


def test_handles_zero_portfolio():
    strategy = PercentOfPortfolioStrategy(withdrawal_rate=0.04)
    assert strategy.get_withdrawal(0, 1.5, 0) == 0
