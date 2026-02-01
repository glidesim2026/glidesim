from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import yaml
from scipy import stats


class Regime(IntEnum):
    """Market regime states."""

    NORMAL = 0
    RECESSION = 1
    STAGFLATION = 2
    DEPRESSION = 3


@dataclass
class RegimeParams:
    """Parameters for a single market regime."""

    stock_mean: float
    bond_mean: float
    inflation_mean: float
    stock_std: float
    bond_std: float
    inflation_std: float
    stock_ar: float
    bond_ar: float
    inflation_ar: float
    stock_bond_corr: float
    stock_inflation_corr: float
    bond_inflation_corr: float
    inflation_model: str
    stock_skewness: float
    bond_skewness: float
    inflation_skewness: float
    stock_model: str = "direct"  # "direct" or "log"

    def build_covariance_matrix(self) -> np.ndarray:
        """Build 3x3 covariance matrix from correlations and std devs."""
        stds = np.array([self.stock_std, self.bond_std, self.inflation_std])
        corr = np.array(
            [
                [1.0, self.stock_bond_corr, self.stock_inflation_corr],
                [self.stock_bond_corr, 1.0, self.bond_inflation_corr],
                [self.stock_inflation_corr, self.bond_inflation_corr, 1.0],
            ]
        )
        return np.outer(stds, stds) * corr


@dataclass
class MarketModelParams:
    """Full market model parameters with four regimes."""

    normal: RegimeParams
    recession: RegimeParams
    stagflation: RegimeParams
    depression: RegimeParams
    transition_matrix: np.ndarray
    warm_up_years: int
    max_sigma: float

    @classmethod
    def from_yaml(cls, path: Path) -> "MarketModelParams":
        """Load parameters from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        def parse_regime(name: str) -> RegimeParams:
            r = data["regimes"][name]
            return RegimeParams(
                stock_mean=r["means"]["stock"],
                bond_mean=r["means"]["bond"],
                inflation_mean=r["means"]["inflation"],
                stock_std=r["std_devs"]["stock"],
                bond_std=r["std_devs"]["bond"],
                inflation_std=r["std_devs"]["inflation"],
                stock_ar=r["ar_coefficients"]["stock"],
                bond_ar=r["ar_coefficients"]["bond"],
                inflation_ar=r["ar_coefficients"]["inflation"],
                stock_bond_corr=r["correlations"]["stock_bond"],
                stock_inflation_corr=r["correlations"]["stock_inflation"],
                bond_inflation_corr=r["correlations"]["bond_inflation"],
                inflation_model=r["inflation_model"],
                stock_skewness=r["skewness"]["stock"],
                bond_skewness=r["skewness"]["bond"],
                inflation_skewness=r["skewness"]["inflation"],
                stock_model=r.get("stock_model", "direct"),
            )

        transition_matrix = np.array(
            [
                data["transitions"]["normal"],
                data["transitions"]["recession"],
                data["transitions"]["stagflation"],
                data["transitions"]["depression"],
            ]
        )

        return cls(
            normal=parse_regime("normal"),
            recession=parse_regime("recession"),
            stagflation=parse_regime("stagflation"),
            depression=parse_regime("depression"),
            transition_matrix=transition_matrix,
            warm_up_years=data["warm_up_years"],
            max_sigma=data.get("max_sigma", float("inf")),
        )

    def get_regime_params(self, regime: Regime) -> RegimeParams:
        """Get parameters for a specific regime."""
        return [self.normal, self.recession, self.stagflation, self.depression][regime]


@dataclass
class MarketData:
    """Output of market model generation."""

    stock_returns: np.ndarray
    bond_returns: np.ndarray
    inflation: np.ndarray
    regimes: np.ndarray


class MarketModel:
    """Regime-switching market model with VAR(1) dynamics."""

    def __init__(self, params: MarketModelParams):
        self.params = params
        self._build_regime_matrices()

    def _build_regime_matrices(self):
        """Pre-compute covariance matrices for each regime."""
        self.covariances = {
            regime: self.params.get_regime_params(regime).build_covariance_matrix()
            for regime in Regime
        }

    def _compute_stationary_distribution(self) -> np.ndarray:
        """Compute stationary distribution of regime Markov chain."""
        P = self.params.transition_matrix
        n = len(Regime)
        A = np.vstack([P.T - np.eye(n), np.ones(n)])
        b = np.zeros(n + 1)
        b[-1] = 1
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
        return pi

    def _generate_regimes(
        self, n_years: int, n_simulations: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate Markov regime sequences."""
        regimes = np.zeros((n_simulations, n_years), dtype=np.int8)

        stationary = self._compute_stationary_distribution()
        regimes[:, 0] = rng.choice(len(Regime), size=n_simulations, p=stationary)

        P = self.params.transition_matrix
        for t in range(1, n_years):
            u = rng.random(n_simulations)
            for from_regime in Regime:
                mask = regimes[:, t - 1] == from_regime
                if not mask.any():
                    continue
                cumprob = np.cumsum(P[from_regime])
                regimes[mask, t] = np.searchsorted(cumprob, u[mask])

        return regimes

    def _clamp_draws(
        self, draws: np.ndarray, means: list[float], stds: np.ndarray
    ) -> np.ndarray:
        """Clamp draws to ±max_sigma standard deviations from means.

        Prevents extreme tail events from compounding via AR dynamics.
        Operates in state space (log space for log-normal models).
        """
        max_sigma = self.params.max_sigma
        if not np.isfinite(max_sigma):
            return draws
        lower = np.array(means) - max_sigma * stds
        upper = np.array(means) + max_sigma * stds
        return np.clip(draws, lower, upper)

    def _sample_skewnorm_copula(
        self,
        n_samples: int,
        means: np.ndarray,
        cov: np.ndarray,
        skewness: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample correlated skew-normal values using Gaussian copula.

        Uses copula method to combine:
        - Correlation structure from multivariate normal
        - Skewness from skew-normal marginals

        Args:
            n_samples: Number of samples to generate.
            means: Mean values for each variable [stock, bond, inflation].
            cov: 3x3 covariance matrix.
            skewness: Skewness parameters (scipy skewnorm 'a') for each variable.
            rng: Random number generator.

        Returns:
            Array of shape (n_samples, 3) with correlated skew-normal samples.
        """
        corr = (
            np.diag(1.0 / np.sqrt(np.diag(cov)))
            @ cov
            @ np.diag(1.0 / np.sqrt(np.diag(cov)))
        )
        normal_samples = rng.multivariate_normal(np.zeros(3), corr, n_samples)

        stds = np.sqrt(np.diag(cov))
        uniform = stats.norm.cdf(normal_samples)

        skewed = np.zeros_like(normal_samples)
        for i, alpha in enumerate(skewness):
            if alpha == 0:
                skewed[:, i] = stats.norm.ppf(uniform[:, i]) * stds[i]
            else:
                raw = stats.skewnorm.ppf(uniform[:, i], alpha)
                raw_centered = raw - stats.skewnorm.mean(alpha)
                skewed[:, i] = raw_centered * stds[i] / stats.skewnorm.std(alpha)

        return skewed + means

    def _generate_returns(
        self,
        n_years: int,
        n_simulations: int,
        regimes: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate returns with regime-specific VAR(1) dynamics.

        Supports both direct and log-normal models for stocks and inflation.
        For log-normal, AR dynamics operate in log space: log(1+r).
        """
        stocks = np.zeros((n_simulations, n_years))
        stock_state = np.zeros((n_simulations, n_years))  # for log model
        bonds = np.zeros((n_simulations, n_years))
        inflation = np.zeros((n_simulations, n_years))
        inflation_state = np.zeros((n_simulations, n_years))

        for regime in Regime:
            mask = regimes[:, 0] == regime
            n_in_regime = mask.sum()
            if n_in_regime == 0:
                continue

            params = self.params.get_regime_params(regime)
            cov = self.covariances[regime]

            # Compute internal means for log models
            # For log-normal, E[return] = exp(μ + σ²/2) - 1 = target_mean
            # So μ = log(1 + target_mean) - σ²/2
            if params.stock_model == "log":
                stock_mean_internal = (
                    np.log(1 + params.stock_mean) - params.stock_std**2 / 2
                )
            else:
                stock_mean_internal = params.stock_mean

            if params.inflation_model == "log":
                inf_mean_internal = (
                    np.log(1 + params.inflation_mean) - params.inflation_std**2 / 2
                )
            else:
                inf_mean_internal = params.inflation_mean

            means = np.array([stock_mean_internal, params.bond_mean, inf_mean_internal])
            stds = np.array([params.stock_std, params.bond_std, params.inflation_std])
            skewness = np.array(
                [params.stock_skewness, params.bond_skewness, params.inflation_skewness]
            )
            draws = self._sample_skewnorm_copula(n_in_regime, means, cov, skewness, rng)
            draws = self._clamp_draws(draws, means.tolist(), stds)

            stock_state[mask, 0] = draws[:, 0]
            bonds[mask, 0] = draws[:, 1]
            inflation_state[mask, 0] = draws[:, 2]

            # Transform from internal state to actual returns
            if params.stock_model == "log":
                stocks[mask, 0] = np.exp(draws[:, 0]) - 1
            else:
                stocks[mask, 0] = draws[:, 0]

            if params.inflation_model == "log":
                inflation[mask, 0] = np.exp(draws[:, 2]) - 1
            else:
                inflation[mask, 0] = draws[:, 2]

        for t in range(1, n_years):
            for regime in Regime:
                mask = regimes[:, t] == regime
                n_in_regime = mask.sum()
                if n_in_regime == 0:
                    continue

                params = self.params.get_regime_params(regime)
                cov = self.covariances[regime]

                # Compute internal means for log models
                # For log-normal, E[return] = exp(μ + σ²/2) - 1 = target_mean
                # So μ = log(1 + target_mean) - σ²/2
                if params.stock_model == "log":
                    stock_mean_internal = (
                        np.log(1 + params.stock_mean) - params.stock_std**2 / 2
                    )
                else:
                    stock_mean_internal = params.stock_mean

                if params.inflation_model == "log":
                    inf_mean_internal = (
                        np.log(1 + params.inflation_mean) - params.inflation_std**2 / 2
                    )
                else:
                    inf_mean_internal = params.inflation_mean

                stds = np.array(
                    [params.stock_std, params.bond_std, params.inflation_std]
                )
                skewness = np.array(
                    [
                        params.stock_skewness,
                        params.bond_skewness,
                        params.inflation_skewness,
                    ]
                )
                innovations = self._sample_skewnorm_copula(
                    n_in_regime, np.zeros(3), cov, skewness, rng
                )
                innovations = self._clamp_draws(innovations, [0, 0, 0], stds)

                # AR dynamics in internal state space
                stock_state[mask, t] = (
                    stock_mean_internal
                    + params.stock_ar * (stock_state[mask, t - 1] - stock_mean_internal)
                    + innovations[:, 0]
                )
                bonds[mask, t] = (
                    params.bond_mean
                    + params.bond_ar * (bonds[mask, t - 1] - params.bond_mean)
                    + innovations[:, 1]
                )
                inflation_state[mask, t] = (
                    inf_mean_internal
                    + params.inflation_ar
                    * (inflation_state[mask, t - 1] - inf_mean_internal)
                    + innovations[:, 2]
                )

                # Transform from internal state to actual returns
                if params.stock_model == "log":
                    stocks[mask, t] = np.exp(stock_state[mask, t]) - 1
                else:
                    stocks[mask, t] = stock_state[mask, t]

                if params.inflation_model == "log":
                    inflation[mask, t] = np.exp(inflation_state[mask, t]) - 1
                else:
                    inflation[mask, t] = inflation_state[mask, t]

        return stocks, bonds, inflation

    def generate(
        self, n_years: int, n_simulations: int, rng: np.random.Generator
    ) -> MarketData:
        """Generate regime-switching market data with warm-up."""
        total_years = self.params.warm_up_years + n_years

        regimes = self._generate_regimes(total_years, n_simulations, rng)
        stocks, bonds, inflation = self._generate_returns(
            total_years, n_simulations, regimes, rng
        )

        warm_up = self.params.warm_up_years

        return MarketData(
            stock_returns=stocks[:, warm_up:],
            bond_returns=bonds[:, warm_up:],
            inflation=inflation[:, warm_up:],
            regimes=regimes[:, warm_up:],
        )
