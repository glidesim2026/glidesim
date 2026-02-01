from glidesim.strategies.base import Strategy


class FixedWithdrawalStrategy(Strategy):
    """Withdraw a fixed nominal amount each year regardless of inflation or portfolio value.

    Simple but purchasing power declines with inflation over time.

    Attributes:
        withdrawal_amount: Annual withdrawal amount in nominal dollars.
    """

    def __init__(self, withdrawal_amount: float):
        self.withdrawal_amount = withdrawal_amount

    def get_withdrawal(
        self,
        portfolio_value: float,
        cumulative_inflation: float,
        year: int,
    ) -> float:
        return min(self.withdrawal_amount, portfolio_value)
