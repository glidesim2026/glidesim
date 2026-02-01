from dataclasses import dataclass

import numpy as np

from glidesim.simulation.engine import SimulationResults


@dataclass
class SimulationMetrics:
    """Summary statistics from a simulation run.

    Attributes:
        success_rate: Fraction of simulations ending with positive balance (0.0 to 1.0).
        final_value_percentiles: Final portfolio values at specified percentiles.
    """

    success_rate: float
    final_value_percentiles: dict[int, float]


def calculate_metrics(
    results: SimulationResults, percentiles: list[int] | None = None
) -> SimulationMetrics:
    """Calculate summary metrics from simulation results.

    Args:
        results: Output from run_simulation().
        percentiles: Which percentiles to compute for final values.
            Defaults to [10, 25, 50, 75, 90].

    Returns:
        SimulationMetrics with success rate and final value percentiles.
    """
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90]

    final_values = results.portfolio_values[:, -1]

    success_rate = np.mean(final_values > 0)

    final_value_percentiles = {
        p: float(np.percentile(final_values, p)) for p in percentiles
    }

    return SimulationMetrics(
        success_rate=success_rate,
        final_value_percentiles=final_value_percentiles,
    )
