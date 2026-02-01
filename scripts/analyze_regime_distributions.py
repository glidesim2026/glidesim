#!/usr/bin/env python3
"""Analyze return distributions by regime for the market model.

This script runs simulations and characterizes the return distributions
for each regime, outputting statistics and optional plots.

Usage:
    uv run python scripts/analyze_regime_distributions.py
    uv run python scripts/analyze_regime_distributions.py --n-simulations 10000 --n-years 100
    uv run python scripts/analyze_regime_distributions.py --plots --output results.json
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from glidesim.simulation.engine import _MARKET_PARAMS_PATH
from glidesim.simulation.market_model import (
    MarketModel,
    MarketModelParams,
    Regime,
)


@dataclass
class DistributionStats:
    """Statistics for a single distribution."""

    count: int
    mean: float
    std: float
    min: float
    max: float
    p1: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    p99: float
    skewness: float
    kurtosis: float


@dataclass
class RegimeStats:
    """Statistics for all assets in a single regime."""

    regime: str
    stock: DistributionStats
    bond: DistributionStats
    inflation: DistributionStats


def compute_distribution_stats(data: np.ndarray) -> DistributionStats:
    """Compute statistics for a 1D array of values."""
    if len(data) == 0:
        return DistributionStats(
            count=0,
            mean=float("nan"),
            std=float("nan"),
            min=float("nan"),
            max=float("nan"),
            p1=float("nan"),
            p5=float("nan"),
            p25=float("nan"),
            p50=float("nan"),
            p75=float("nan"),
            p95=float("nan"),
            p99=float("nan"),
            skewness=float("nan"),
            kurtosis=float("nan"),
        )

    percentiles = np.percentile(data, [1, 5, 25, 50, 75, 95, 99])
    return DistributionStats(
        count=len(data),
        mean=float(np.mean(data)),
        std=float(np.std(data)),
        min=float(np.min(data)),
        max=float(np.max(data)),
        p1=float(percentiles[0]),
        p5=float(percentiles[1]),
        p25=float(percentiles[2]),
        p50=float(percentiles[3]),
        p75=float(percentiles[4]),
        p95=float(percentiles[5]),
        p99=float(percentiles[6]),
        skewness=float(stats.skew(data)),
        kurtosis=float(stats.kurtosis(data, fisher=True)),
    )


def analyze_regime(
    regime: Regime,
    stock_returns: np.ndarray,
    bond_returns: np.ndarray,
    inflation: np.ndarray,
    regime_mask: np.ndarray,
) -> RegimeStats:
    """Compute statistics for all assets in a given regime."""
    return RegimeStats(
        regime=regime.name,
        stock=compute_distribution_stats(stock_returns[regime_mask]),
        bond=compute_distribution_stats(bond_returns[regime_mask]),
        inflation=compute_distribution_stats(inflation[regime_mask]),
    )


def run_analysis(
    n_simulations: int = 5000,
    n_years: int = 100,
    seed: int = 42,
    params_path: Path | None = None,
) -> dict[str, RegimeStats]:
    """Run simulations and compute per-regime statistics."""
    if params_path is None:
        params_path = _MARKET_PARAMS_PATH

    params = MarketModelParams.from_yaml(params_path)
    model = MarketModel(params)
    rng = np.random.default_rng(seed)

    data = model.generate(n_years, n_simulations, rng)

    results = {}
    for regime in Regime:
        mask = data.regimes == regime
        results[regime.name] = analyze_regime(
            regime,
            data.stock_returns,
            data.bond_returns,
            data.inflation,
            mask,
        )

    return results


def print_report(results: dict[str, RegimeStats]) -> None:
    """Print a formatted report of the analysis results."""
    for regime_name, regime_stats in results.items():
        print(f"\n{'=' * 60}")
        print(f"REGIME: {regime_name}")
        print(f"{'=' * 60}")

        for asset_name in ["stock", "bond", "inflation"]:
            asset_stats: DistributionStats = getattr(regime_stats, asset_name)
            print(f"\n  {asset_name.upper()} (n={asset_stats.count:,})")
            print(f"    Mean:     {asset_stats.mean:+.2%}")
            print(f"    Std:      {asset_stats.std:.2%}")
            print(f"    Range:    [{asset_stats.min:+.2%}, {asset_stats.max:+.2%}]")
            print("    Percentiles:")
            print(f"       1st: {asset_stats.p1:+.2%}")
            print(f"       5th: {asset_stats.p5:+.2%}")
            print(f"      25th: {asset_stats.p25:+.2%}")
            print(f"      50th: {asset_stats.p50:+.2%}")
            print(f"      75th: {asset_stats.p75:+.2%}")
            print(f"      95th: {asset_stats.p95:+.2%}")
            print(f"      99th: {asset_stats.p99:+.2%}")
            print(f"    Skewness: {asset_stats.skewness:+.3f}")
            print(f"    Kurtosis: {asset_stats.kurtosis:+.3f}")


def results_to_dict(results: dict[str, RegimeStats]) -> dict:
    """Convert results to a JSON-serializable dictionary."""
    return {name: asdict(stats) for name, stats in results.items()}


def save_results(results: dict[str, RegimeStats], output_path: Path) -> None:
    """Save results to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(results_to_dict(results), f, indent=2)
    print(f"\nResults saved to {output_path}")


def generate_plots(
    results: dict[str, RegimeStats],
    n_simulations: int,
    n_years: int,
    seed: int,
    output_dir: Path | None = None,
) -> None:
    """Generate histogram plots for each regime and asset."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nError: matplotlib is required for plots. Install with:")
        print("  uv add matplotlib --dev")
        return

    params = MarketModelParams.from_yaml(_MARKET_PARAMS_PATH)
    model = MarketModel(params)
    rng = np.random.default_rng(seed)
    data = model.generate(n_years, n_simulations, rng)

    if output_dir is None:
        output_dir = Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    fig.suptitle("Return Distributions by Regime", fontsize=14)

    asset_names = ["stock", "bond", "inflation"]
    asset_data = [data.stock_returns, data.bond_returns, data.inflation]

    for row, regime in enumerate(Regime):
        mask = data.regimes == regime
        regime_stats = results[regime.name]

        for col, (asset_name, asset_array) in enumerate(
            zip(asset_names, asset_data, strict=True)
        ):
            ax = axes[row, col]
            asset_returns = asset_array[mask]

            if len(asset_returns) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(f"{regime.name} - {asset_name.upper()}")
                continue

            ax.hist(asset_returns, bins=50, density=True, alpha=0.7, edgecolor="black")

            asset_stats: DistributionStats = getattr(regime_stats, asset_name)
            ax.axvline(
                asset_stats.mean,
                color="red",
                linestyle="--",
                label=f"Mean: {asset_stats.mean:.1%}",
            )
            ax.axvline(
                asset_stats.p5,
                color="orange",
                linestyle=":",
                label=f"5th: {asset_stats.p5:.1%}",
            )
            ax.axvline(
                asset_stats.p95,
                color="orange",
                linestyle=":",
                label=f"95th: {asset_stats.p95:.1%}",
            )

            ax.set_title(
                f"{regime.name} - {asset_name.upper()} (n={asset_stats.count:,})"
            )
            ax.set_xlabel("Return")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = output_dir / "regime_distributions.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze return distributions by regime"
    )
    parser.add_argument(
        "--n-simulations", type=int, default=5000, help="Number of simulations to run"
    )
    parser.add_argument("--n-years", type=int, default=100, help="Years per simulation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file path"
    )
    parser.add_argument("--plots", action="store_true", help="Generate histogram plots")
    parser.add_argument(
        "--plot-dir", type=str, default=None, help="Directory for plot output"
    )

    args = parser.parse_args()

    print(
        f"Running analysis with {args.n_simulations:,} simulations × {args.n_years} years..."
    )
    results = run_analysis(
        n_simulations=args.n_simulations,
        n_years=args.n_years,
        seed=args.seed,
    )

    print_report(results)

    if args.output:
        save_results(results, Path(args.output))

    if args.plots:
        plot_dir = Path(args.plot_dir) if args.plot_dir else None
        generate_plots(
            results,
            n_simulations=args.n_simulations,
            n_years=args.n_years,
            seed=args.seed,
            output_dir=plot_dir,
        )


if __name__ == "__main__":
    main()
