from glidesim.strategies.percent_of_portfolio_with_minimum import (
    PercentOfPortfolioWithMinimumStrategy,
)


def test_uses_percentage_when_higher():
    strategy = PercentOfPortfolioWithMinimumStrategy(
        withdrawal_rate=0.05, minimum_withdrawal=40_000
    )
    # percentage = 2M * 0.05 = 100k, minimum = 40k
    assert strategy.get_withdrawal(2_000_000, 1.0, 0) == 100_000


def test_uses_minimum_when_higher():
    strategy = PercentOfPortfolioWithMinimumStrategy(
        withdrawal_rate=0.03, minimum_withdrawal=40_000
    )
    # percentage = 1M * 0.03 = 30k, minimum = 40k
    assert strategy.get_withdrawal(1_000_000, 1.0, 0) == 40_000


def test_minimum_inflation_adjusted_by_default():
    strategy = PercentOfPortfolioWithMinimumStrategy(
        withdrawal_rate=0.03, minimum_withdrawal=40_000
    )
    # percentage = 1M * 0.03 = 30k, minimum = 40k * 1.5 = 60k
    assert strategy.get_withdrawal(1_000_000, 1.5, 0) == 60_000


def test_minimum_not_inflated_when_disabled():
    strategy = PercentOfPortfolioWithMinimumStrategy(
        withdrawal_rate=0.03, minimum_withdrawal=40_000, inflation_adjustment=False
    )
    # percentage = 1M * 0.03 = 30k, minimum = 40k (no inflation)
    assert strategy.get_withdrawal(1_000_000, 1.5, 0) == 40_000


def test_uses_minimum_during_delay_period():
    strategy = PercentOfPortfolioWithMinimumStrategy(
        withdrawal_rate=0.05, minimum_withdrawal=40_000, years_at_minimum=5
    )
    # Even though percentage = 2M * 0.05 = 100k > minimum = 40k,
    # during years 0-4 we use minimum
    assert strategy.get_withdrawal(2_000_000, 1.0, 0) == 40_000
    assert strategy.get_withdrawal(2_000_000, 1.0, 4) == 40_000


def test_switches_to_percentage_after_delay():
    strategy = PercentOfPortfolioWithMinimumStrategy(
        withdrawal_rate=0.05, minimum_withdrawal=40_000, years_at_minimum=5
    )
    # Year 5 uses percentage = 2M * 0.05 = 100k
    assert strategy.get_withdrawal(2_000_000, 1.0, 5) == 100_000


def test_delay_period_with_inflation():
    strategy = PercentOfPortfolioWithMinimumStrategy(
        withdrawal_rate=0.05, minimum_withdrawal=40_000, years_at_minimum=5
    )
    # Year 3 with 1.5x inflation: minimum = 40k * 1.5 = 60k
    assert strategy.get_withdrawal(2_000_000, 1.5, 3) == 60_000


def test_caps_withdrawal_at_portfolio_value():
    strategy = PercentOfPortfolioWithMinimumStrategy(
        withdrawal_rate=0.05, minimum_withdrawal=40_000
    )
    # minimum = 40k but portfolio only has 20k
    assert strategy.get_withdrawal(20_000, 1.0, 0) == 20_000


def test_returns_zero_when_portfolio_empty():
    strategy = PercentOfPortfolioWithMinimumStrategy(
        withdrawal_rate=0.05, minimum_withdrawal=40_000
    )
    assert strategy.get_withdrawal(0, 1.0, 0) == 0
