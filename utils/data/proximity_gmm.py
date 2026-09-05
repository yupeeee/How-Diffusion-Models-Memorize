"""Deterministic two-component Gaussian mixture for proximity observations."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


COMPONENT_NAMES = ("low_sscd_mode", "high_sscd_mode")
DEFAULT_REG_COVAR = 1e-6
MAX_ITERATIONS = 300
TOLERANCE = 1e-10
LIKELIHOOD_DECREASE_TOLERANCE = 1e-9
INITIALIZATION_NAME = "deterministic_first_pc_extrema_kmeans"


@dataclass(frozen=True)
class GaussianMixtureFit:
    """A converged, semantically ordered two-component mixture."""

    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    responsibilities: np.ndarray
    log_likelihood: float
    iterations: int


def standardize_features(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize finite ``(L2, SSCD)`` observations column by column."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("proximity input must be a finite n-by-2 matrix")
    feature_mean = values.mean(axis=0)
    feature_scale = values.std(axis=0, ddof=0)
    if np.any(~np.isfinite(feature_scale)) or np.any(feature_scale == 0.0):
        raise ValueError("L2 and SSCD must each have nonzero finite variance")
    return (values - feature_mean) / feature_scale, feature_mean, feature_scale


def _kmeans_two(values: np.ndarray) -> np.ndarray:
    """Fit dependency-free k=2 using deterministic first-PC extrema."""

    centered = values - values.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ right_vectors[0]
    endpoints = [int(np.argmin(projection)), int(np.argmax(projection))]
    if endpoints[0] == endpoints[1]:
        raise ValueError("k-means cannot initialize two distinct clusters")
    centers = values[endpoints].copy()
    labels = np.full(len(values), -1, dtype=np.int64)
    for _ in range(MAX_ITERATIONS):
        squared_distance = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        next_labels = squared_distance.argmin(axis=1)
        if any(not np.any(next_labels == label) for label in (0, 1)):
            raise ValueError("k-means formed an empty cluster")
        if np.array_equal(next_labels, labels):
            return labels
        labels = next_labels
        centers = np.vstack([values[labels == label].mean(axis=0) for label in (0, 1)])
    raise ValueError(f"k-means did not converge within {MAX_ITERATIONS} iterations")


def _regularize_covariance(covariance: np.ndarray, floor: float) -> np.ndarray:
    symmetric = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if np.any(~np.isfinite(eigenvalues)):
        raise ValueError("Gaussian covariance has nonfinite eigenvalues")
    eigenvalues = np.maximum(eigenvalues, floor)
    result = (eigenvectors * eigenvalues) @ eigenvectors.T
    result = (result + result.T) / 2.0
    np.linalg.cholesky(result)
    return result


def _maximization(
    values: np.ndarray,
    responsibilities: np.ndarray,
    *,
    reg_covar: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    component_mass = responsibilities.sum(axis=0)
    if np.any(~np.isfinite(component_mass)) or np.any(
        component_mass <= values.shape[1]
    ):
        raise ValueError("Gaussian component collapsed below the usable sample mass")
    weights = component_mass / len(values)
    means = (responsibilities.T @ values) / component_mass[:, None]
    covariances = np.empty((2, values.shape[1], values.shape[1]), dtype=np.float64)
    for component in range(2):
        delta = values - means[component]
        covariance = (
            (delta.T * responsibilities[:, component]) @ delta
        ) / component_mass[component]
        covariances[component] = _regularize_covariance(covariance, reg_covar)
    return weights, means, covariances


def _log_gaussian_density(
    values: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    dimensions = values.shape[1]
    result = np.empty((len(values), 2), dtype=np.float64)
    constant = dimensions * math.log(2.0 * math.pi)
    for component in range(2):
        cholesky = np.linalg.cholesky(covariances[component])
        whitened = np.linalg.solve(cholesky, (values - means[component]).T)
        squared_mahalanobis = np.square(whitened).sum(axis=0)
        log_determinant = 2.0 * np.log(np.diag(cholesky)).sum()
        result[:, component] = -0.5 * (constant + log_determinant + squared_mahalanobis)
    return result


def expectation(
    values: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return component posteriors and summed log likelihood."""

    log_joint = _log_gaussian_density(values, means, covariances) + np.log(weights)
    log_normalizer = np.logaddexp(log_joint[:, 0], log_joint[:, 1])
    responsibilities = np.exp(log_joint - log_normalizer[:, None])
    return responsibilities, float(log_normalizer.sum())


def marginal_component_boundary(
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    feature_index: int,
) -> float:
    """Return the equal-posterior marginal boundary between ordered means.

    The component weights are part of the one-dimensional marginal posterior.
    A usable boundary must lie strictly between the two component means, with
    each component dominating at its own mean.
    """

    weights = np.asarray(weights, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    covariances = np.asarray(covariances, dtype=np.float64)
    if (
        weights.shape != (2,)
        or means.ndim != 2
        or means.shape[0] != 2
        or covariances.shape != (2, means.shape[1], means.shape[1])
        or isinstance(feature_index, bool)
        or not isinstance(feature_index, int)
        or not 0 <= feature_index < means.shape[1]
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(means))
        or not np.all(np.isfinite(covariances))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("marginal boundary parameters are invalid")
    component_means = means[:, feature_index]
    variances = covariances[:, feature_index, feature_index]
    if component_means[0] >= component_means[1] or np.any(variances <= 0.0):
        raise ValueError("marginal components are not ordered and nondegenerate")

    def log_density_difference(value: float) -> float:
        return float(
            math.log(weights[0] / weights[1])
            - 0.5 * math.log(variances[0] / variances[1])
            - (value - component_means[0]) ** 2 / (2.0 * variances[0])
            + (value - component_means[1]) ** 2 / (2.0 * variances[1])
        )

    lower, upper = (float(value) for value in component_means)
    lower_difference = log_density_difference(lower)
    upper_difference = log_density_difference(upper)
    if not lower_difference > 0.0 or not upper_difference < 0.0:
        raise ValueError(
            "marginal components have no unambiguous boundary between their means"
        )
    for _ in range(128):
        midpoint = lower + (upper - lower) / 2.0
        if log_density_difference(midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    boundary = lower + (upper - lower) / 2.0
    if not component_means[0] < boundary < component_means[1]:
        raise ValueError("marginal component boundary is outside its means")
    return boundary


def _validate_fit(fit: GaussianMixtureFit, values: np.ndarray) -> None:
    if fit.responsibilities.shape != (len(values), 2):
        raise ValueError("Gaussian responsibilities have the wrong shape")
    if not np.all(np.isfinite(fit.responsibilities)):
        raise ValueError("Gaussian responsibilities are nonfinite")
    if not np.allclose(fit.responsibilities.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Gaussian responsibilities do not sum to one")
    if np.any(fit.responsibilities < 0.0) or np.any(fit.responsibilities > 1.0):
        raise ValueError("Gaussian responsibilities fall outside [0, 1]")
    if (
        np.any(~np.isfinite(fit.weights))
        or np.any(fit.weights <= 0.0)
        or not np.isclose(fit.weights.sum(), 1.0)
    ):
        raise ValueError("Gaussian weights are invalid")
    if np.any(~np.isfinite(fit.means)) or np.any(~np.isfinite(fit.covariances)):
        raise ValueError("Gaussian parameters are nonfinite")
    for covariance in fit.covariances:
        if not np.allclose(covariance, covariance.T, atol=1e-12):
            raise ValueError("Gaussian covariance is not symmetric")
        np.linalg.cholesky(covariance)
    if not math.isfinite(fit.log_likelihood):
        raise ValueError("Gaussian log likelihood is nonfinite")
    if np.unique(fit.responsibilities.argmax(axis=1)).size != 2:
        raise ValueError("Gaussian hard assignments contain an empty component")


def fit_gaussian_mixture(
    values: np.ndarray,
    *,
    reg_covar: float = DEFAULT_REG_COVAR,
) -> GaussianMixtureFit:
    """Fit a deterministic full-covariance GMM ordered by SSCD mean."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("Gaussian input must be a finite n-by-2 matrix")
    if not math.isfinite(reg_covar) or reg_covar <= 0.0:
        raise ValueError("reg_covar must be finite and greater than zero")
    if not np.allclose(values.mean(axis=0), 0.0, atol=1e-10):
        raise ValueError("Gaussian input is not centered")
    if not np.allclose(values.std(axis=0, ddof=0), 1.0, atol=1e-10):
        raise ValueError("Gaussian input does not have unit variance")

    initial_labels = _kmeans_two(values)
    initial_responsibilities = np.eye(2, dtype=np.float64)[initial_labels]
    weights, means, covariances = _maximization(
        values,
        initial_responsibilities,
        reg_covar=reg_covar,
    )
    previous_mean_log_likelihood = -math.inf

    for iteration in range(1, MAX_ITERATIONS + 1):
        responsibilities, log_likelihood = expectation(
            values,
            weights,
            means,
            covariances,
        )
        mean_log_likelihood = log_likelihood / len(values)
        improvement = mean_log_likelihood - previous_mean_log_likelihood
        if improvement < -LIKELIHOOD_DECREASE_TOLERANCE:
            raise ValueError(
                "Gaussian log likelihood decreased during EM: "
                f"{improvement:.3e} per observation"
            )
        if improvement <= TOLERANCE:
            order = np.argsort(means[:, 1], kind="stable")
            ordered_means = means[order]
            if math.isclose(
                float(ordered_means[0, 1]),
                float(ordered_means[1, 1]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "Gaussian components have indistinguishable SSCD means"
                )
            fit = GaussianMixtureFit(
                weights=weights[order],
                means=ordered_means,
                covariances=covariances[order],
                responsibilities=responsibilities[:, order],
                log_likelihood=log_likelihood,
                iterations=iteration,
            )
            _validate_fit(fit, values)
            return fit
        weights, means, covariances = _maximization(
            values,
            responsibilities,
            reg_covar=reg_covar,
        )
        previous_mean_log_likelihood = mean_log_likelihood
    raise ValueError(f"Gaussian EM did not converge within {MAX_ITERATIONS} iterations")


def raw_parameters(
    fit: GaussianMixtureFit,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform standardized component parameters to raw L2/SSCD units."""

    means = feature_mean + fit.means * feature_scale
    scale_matrix = np.diag(feature_scale)
    covariances = np.asarray(
        [scale_matrix @ covariance @ scale_matrix for covariance in fit.covariances]
    )
    return means, covariances
