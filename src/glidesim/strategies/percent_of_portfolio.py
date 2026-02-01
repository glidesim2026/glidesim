from glidesim.strategies.base import Strategy


class PercentOfPortfolioStrategy(Strategy):
    """Withdraw a fixed percentage of current portfolio value each year.

    Adapts to market performance: withdrawals rise when portfolio grows and fall
    when it declines. Never depletes the portfolio but income can be volatile.

    Attributes:
        withdrawal_rate: Fraction of portfolio to withdraw (e.g., 0.04 for 4%).
    """

    def __init__(self, withdrawal_rate: float):
        self.withdrawal_rate = withdrawal_rate

    def get_withdrawal(
        self,
        portfolio_value: float,
        cumulative_inflation: float,
        year: int,
    ) -> float:
        return portfolio_value * self.withdrawal_rate
