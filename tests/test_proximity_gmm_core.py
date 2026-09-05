"""Focused tests for deterministic proximity GMM fitting."""

from __future__ import annotations

import math

import numpy as np
import pytest

from utils.data.proximity_gmm import (
    COMPONENT_NAMES,
    expectation,
    fit_gaussian_mixture,
    marginal_component_boundary,
    raw_parameters,
    standardize_features,
)


def _known_clouds() -> np.ndarray:
    low_sscd = np.asarray(
        [
            [1.0, 0.09],
            [1.2, 0.11],
            [1.4, 0.08],
            [1.6, 0.12],
            [1.8, 0.10],
            [2.0, 0.07],
        ],
        dtype=np.float64,
    )
    high_sscd = np.asarray(
        [
            [3.0, 0.82],
            [3.2, 0.78],
            [3.4, 0.75],
            [3.6, 0.71],
            [3.8, 0.68],
            [4.0, 0.64],
        ],
        dtype=np.float64,
    )
    return np.vstack((low_sscd, high_sscd))


def test_known_cloud_fit_is_deterministic_and_semantically_ordered() -> None:
    features, _mean, _scale = standardize_features(_known_clouds())

    first = fit_gaussian_mixture(features)
    second = fit_gaussian_mixture(features)

    assert COMPONENT_NAMES == ("low_sscd_mode", "high_sscd_mode")
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.means, second.means)
    np.testing.assert_array_equal(first.covariances, second.covariances)
    np.testing.assert_array_equal(first.responsibilities, second.responsibilities)
    assert first.log_likelihood == second.log_likelihood
    assert first.iterations == second.iterations
    assert first.means[0, 1] < first.means[1, 1]
    assert np.all(first.responsibilities[:6, 0] > 0.5)
    assert np.all(first.responsibilities[6:, 1] > 0.5)


def test_fit_parameters_and_posteriors_are_valid_and_reproducible() -> None:
    raw = _known_clouds()
    features, feature_mean, feature_scale = standardize_features(raw)
    fit = fit_gaussian_mixture(features)

    assert fit.weights.shape == (2,)
    assert fit.means.shape == (2, 2)
    assert fit.covariances.shape == (2, 2, 2)
    assert fit.responsibilities.shape == (len(features), 2)
    np.testing.assert_allclose(fit.weights.sum(), 1.0)
    np.testing.assert_allclose(fit.responsibilities.sum(axis=1), 1.0)
    assert np.all((0.0 <= fit.responsibilities) & (fit.responsibilities <= 1.0))
    for covariance in fit.covariances:
        np.linalg.cholesky(covariance)

    responsibilities, log_likelihood = expectation(
        features, fit.weights, fit.means, fit.covariances
    )
    np.testing.assert_allclose(responsibilities, fit.responsibilities)
    assert log_likelihood == pytest.approx(fit.log_likelihood)

    means, covariances = raw_parameters(fit, feature_mean, feature_scale)
    assert means[0, 1] < means[1, 1]
    assert covariances.shape == (2, 2, 2)


def test_standardization_produces_centered_unit_variance_features() -> None:
    features, feature_mean, feature_scale = standardize_features(_known_clouds())

    np.testing.assert_allclose(features.mean(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(features.std(axis=0, ddof=0), 1.0, atol=1e-15)
    np.testing.assert_allclose(
        features * feature_scale + feature_mean,
        _known_clouds(),
    )


def test_marginal_boundary_uses_component_weights() -> None:
    weights = np.asarray([0.8, 0.2], dtype=np.float64)
    means = np.asarray([[0.0, -1.0], [0.0, 1.0]], dtype=np.float64)
    covariances = np.asarray([np.eye(2), np.eye(2)], dtype=np.float64)

    boundary = marginal_component_boundary(weights, means, covariances, feature_index=1)

    assert boundary == pytest.approx(math.log(4.0) / 2.0)


def test_marginal_boundary_uses_marginal_variances_and_ignores_tail_crossing() -> None:
    weights = np.asarray([0.5, 0.5], dtype=np.float64)
    means = np.asarray([[0.0, -1.0], [0.0, 1.0]], dtype=np.float64)
    covariances = np.asarray(
        [
            [[1.0, 0.4], [0.4, 4.0]],
            [[1.0, -0.2], [-0.2, 0.25]],
        ],
        dtype=np.float64,
    )

    boundary = marginal_component_boundary(weights, means, covariances, feature_index=1)
    weighted_log_densities = [
        math.log(weights[component])
        - 0.5 * math.log(covariances[component, 1, 1])
        - (boundary - means[component, 1]) ** 2 / (2.0 * covariances[component, 1, 1])
        for component in range(2)
    ]

    assert means[0, 1] < boundary < means[1, 1]
    assert weighted_log_densities[0] == pytest.approx(weighted_log_densities[1])


def test_marginal_boundary_requires_an_interior_component_transition() -> None:
    weights = np.asarray([0.999, 0.001], dtype=np.float64)
    means = np.asarray([[0.0, -1.0], [0.0, 1.0]], dtype=np.float64)
    covariances = np.asarray([np.eye(2), np.eye(2)], dtype=np.float64)

    with pytest.raises(ValueError, match="no unambiguous boundary"):
        marginal_component_boundary(weights, means, covariances, feature_index=1)


@pytest.mark.parametrize(
    "values",
    (
        np.ones((4, 2), dtype=np.float64),
        np.asarray([[1.0, 0.1], [1.0, 0.2]], dtype=np.float64),
    ),
)
def test_standardization_rejects_zero_variance(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="nonzero finite variance"):
        standardize_features(values)


@pytest.mark.parametrize(
    "values",
    (
        np.ones(4, dtype=np.float64),
        np.ones((4, 3), dtype=np.float64),
        np.asarray([[1.0, 0.1], [np.nan, 0.2]], dtype=np.float64),
    ),
)
def test_standardization_rejects_invalid_input(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite n-by-2 matrix"):
        standardize_features(values)


def test_fit_rejects_unstandardized_and_semantically_degenerate_input() -> None:
    with pytest.raises(ValueError, match="not centered"):
        fit_gaussian_mixture(_known_clouds())

    equal_sscd_means = np.asarray(
        [[-4.0, -1.0], [-4.0, 0.0], [-4.0, 1.0], [4.0, -1.0], [4.0, 0.0], [4.0, 1.0]],
        dtype=np.float64,
    )
    features, _mean, _scale = standardize_features(equal_sscd_means)
    with pytest.raises(ValueError, match="indistinguishable SSCD means"):
        fit_gaussian_mixture(features)
