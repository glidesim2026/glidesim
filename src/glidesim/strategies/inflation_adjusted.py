from glidesim.strategies.base import Strategy


class InflationAdjustedStrategy(Strategy):
    """Classic 4% rule: withdraw a fixed real amount adjusted for inflation.

    The initial withdrawal amount increases each year with inflation to maintain
    constant purchasing power. Does not adapt to portfolio performance.

    Attributes:
        initial_withdrawal: First-year withdrawal amount in today's dollars.
    """

    def __init__(self, initial_withdrawal: float):
        self.initial_withdrawal = initial_withdrawal

    def get_withdrawal(
        self,
        portfolio_value: float,
        cumulative_inflation: float,
        year: int,
    ) -> float:
        return min(self.initial_withdrawal * cumulative_inflation, portfolio_value)
