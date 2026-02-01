from glidesim.strategies.base import Strategy


class PercentOfPortfolioWithMinimumStrategy(Strategy):
    """Percent-of-portfolio withdrawals with a minimum floor.

    Withdraws a percentage of the current portfolio value, but never less than
    a specified minimum amount. The minimum can optionally be inflation-adjusted.
    A delay period can be set before switching from minimum-only to percentage-based.

    Attributes:
        withdrawal_rate: Fraction of portfolio to withdraw (e.g., 0.04 for 4%).
        minimum_withdrawal: Minimum withdrawal amount in dollars.
        inflation_adjustment: Whether to inflate the minimum with cumulative inflation.
        years_at_minimum: Years to wait before switching to percentage-based withdrawals.
    """

    def __init__(
        self,
        withdrawal_rate: float,
        minimum_withdrawal: float,
        inflation_adjustment: bool = True,
        years_at_minimum: int = 0,
    ):
        self.withdrawal_rate = withdrawal_rate
        self.minimum_withdrawal = minimum_withdrawal
        self.inflation_adjustment = inflation_adjustment
        self.years_at_minimum = years_at_minimum

    def get_withdrawal(
        self,
        portfolio_value: float,
        cumulative_inflation: float,
        year: int,
    ) -> float:
        if self.inflation_adjustment:
            minimum = self.minimum_withdrawal * cumulative_inflation
        else:
            minimum = self.minimum_withdrawal

        if year < self.years_at_minimum:
            target = minimum
        else:
            target = max(portfolio_value * self.withdrawal_rate, minimum)

        return min(target, portfolio_value)
