from glidesim.strategies.base import Strategy


class GuardrailsStrategy(Strategy):
    """Inflation-adjusted withdrawals with guardrails to prevent over/under-spending.

    Starts with inflation-adjusted withdrawals but adjusts when the effective
    withdrawal rate breaches floor or ceiling thresholds. Reduces withdrawal
    if rate exceeds ceiling (portfolio stress), increases if below floor
    (portfolio growth).

    Attributes:
        initial_withdrawal: First-year withdrawal amount in today's dollars.
        floor_rate: Increase spending if withdrawal rate falls below this.
        ceiling_rate: Decrease spending if withdrawal rate exceeds this.
        adjustment_rate: Percentage adjustment when a guardrail is hit.
    """

    def __init__(
        self,
        initial_withdrawal: float,
        floor_rate: float = 0.04,
        ceiling_rate: float = 0.06,
        adjustment_rate: float = 0.10,
    ):
        if floor_rate >= ceiling_rate:
            raise ValueError(
                f"floor_rate ({floor_rate}) must be less than ceiling_rate ({ceiling_rate})"
            )
        self.initial_withdrawal = initial_withdrawal
        self.floor_rate = floor_rate
        self.ceiling_rate = ceiling_rate
        self.adjustment_rate = adjustment_rate

    def get_withdrawal(
        self,
        portfolio_value: float,
        cumulative_inflation: float,
        year: int,
    ) -> float:
        base_withdrawal = self.initial_withdrawal * cumulative_inflation

        if portfolio_value <= 0:
            return 0

        current_rate = base_withdrawal / portfolio_value

        if current_rate > self.ceiling_rate:
            target = base_withdrawal * (1 - self.adjustment_rate)
        elif current_rate < self.floor_rate:
            target = base_withdrawal * (1 + self.adjustment_rate)
        else:
            target = base_withdrawal

        return min(target, portfolio_value)
