from abc import ABC, abstractmethod


class Strategy(ABC):
    """Abstract base class for withdrawal strategies.

    Subclasses must implement get_withdrawal() to define how much to withdraw
    each year based on portfolio value and inflation.
    """

    @abstractmethod
    def get_withdrawal(
        self,
        portfolio_value: float,
        cumulative_inflation: float,
        year: int,
    ) -> float:
        """Calculate the withdrawal amount for the current year.

        Args:
            portfolio_value: Current portfolio balance before withdrawal.
            cumulative_inflation: Cumulative inflation factor since simulation start
                (e.g., 1.1 means 10% total inflation).
            year: Current simulation year (0-indexed, where 0 is first year of retirement).

        Returns:
            Dollar amount to withdraw this year.
        """
        pass
