"""Focused tests for deterministic proximity GMM fitting."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from utils.data.proximity_gmm import (
    COMPONENT_NAMES,
    DEFAULT_REG_COVAR,
    GaussianMixtureFit,
    _fit_from_labels,
    _kmeans_two,
    _maximization,
    expectation,
    fit_gaussian_mixture,
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
        np.testing.assert_allclose(covariance, covariance.T, rtol=0.0, atol=1e-12)
        np.linalg.cholesky(covariance)
        assert np.linalg.eigvalsh(covariance).min() >= (
            DEFAULT_REG_COVAR * (1.0 - 1e-10)
        )

    responsibilities, log_likelihood = expectation(
        features, fit.weights, fit.means, fit.covariances
    )
    np.testing.assert_allclose(responsibilities, fit.responsibilities)
    assert log_likelihood == pytest.approx(fit.log_likelihood)

    means, covariances = raw_parameters(fit, feature_mean, feature_scale)
    assert means[0, 1] < means[1, 1]
    assert covariances.shape == (2, 2, 2)
    assert np.any(np.abs(fit.covariances[:, 0, 1]) > 1e-6)
    np.testing.assert_allclose(
        covariances,
        fit.covariances * feature_scale[None, :, None] * feature_scale[None, None, :],
    )
    np.testing.assert_allclose(
        covariances[:, 0, 1],
        fit.covariances[:, 0, 1] * feature_scale.prod(),
    )


def test_standardization_produces_centered_unit_variance_features() -> None:
    features, feature_mean, feature_scale = standardize_features(_known_clouds())

    np.testing.assert_allclose(features.mean(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(features.std(axis=0, ddof=0), 1.0, atol=1e-15)
    np.testing.assert_allclose(
        features * feature_scale + feature_mean,
        _known_clouds(),
    )


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


@pytest.mark.parametrize(
    "values",
    (
        np.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 1.0], [4.0, 3.0], [5.0, 4.0]]),
        np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]]),
    ),
)
def test_full_maximization_uses_weighted_covariance_and_existing_eigenvalue_floor(
    values: np.ndarray,
) -> None:
    responsibilities = np.asarray(
        [[0.9, 0.1], [0.7, 0.3], [0.5, 0.5], [0.3, 0.7], [0.1, 0.9]]
    )
    weights, means, covariances = _maximization(
        values,
        responsibilities,
        reg_covar=DEFAULT_REG_COVAR,
    )
    np.testing.assert_allclose(weights, [0.5, 0.5])
    for component in range(2):
        mass = responsibilities[:, component].sum()
        expected_mean = (values * responsibilities[:, component, None]).sum(
            axis=0
        ) / mass
        np.testing.assert_allclose(means[component], expected_mean)
        delta = values - expected_mean
        weighted_covariance = (
            (delta.T * responsibilities[:, component]) @ delta
        ) / mass
        eigenvalues, eigenvectors = np.linalg.eigh(weighted_covariance)
        expected_covariance = (
            eigenvectors * np.maximum(eigenvalues, DEFAULT_REG_COVAR)
        ) @ eigenvectors.T
        np.testing.assert_allclose(
            covariances[component], expected_covariance, rtol=1e-12, atol=1e-12
        )
        assert abs(covariances[component, 0, 1]) > 0.1
        np.testing.assert_allclose(
            covariances[component], covariances[component].T, rtol=0.0, atol=1e-12
        )
        np.linalg.cholesky(covariances[component])
        np.testing.assert_allclose(
            np.linalg.eigvalsh(covariances[component]),
            np.maximum(eigenvalues, DEFAULT_REG_COVAR),
            rtol=1e-8,
            atol=1e-12,
        )


def test_full_fit_selects_highest_likelihood_deterministic_start() -> None:
    features, _, _ = standardize_features(_known_clouds())
    fits = [
        _fit_from_labels(
            features,
            _kmeans_two(initialization_features),
            reg_covar=DEFAULT_REG_COVAR,
        )
        for initialization_features in (features, features[:, 1:2])
    ]
    expected = max(fits, key=lambda fit: fit.log_likelihood)
    actual = fit_gaussian_mixture(features)
    assert actual.log_likelihood == expected.log_likelihood
    np.testing.assert_array_equal(actual.weights, expected.weights)
    np.testing.assert_array_equal(actual.means, expected.means)
    np.testing.assert_array_equal(actual.covariances, expected.covariances)
    np.testing.assert_array_equal(actual.responsibilities, expected.responsibilities)


def test_full_covariance_update_and_posteriors_rotate_with_features() -> None:
    values = np.asarray(
        [[1.0, 0.0], [2.0, 1.0], [3.0, 1.0], [4.0, 3.0], [5.0, 4.0]]
    )
    responsibilities = np.asarray(
        [[0.9, 0.1], [0.7, 0.3], [0.5, 0.5], [0.3, 0.7], [0.1, 0.9]]
    )
    angle = np.pi / 5.0
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated_values = values @ rotation.T
    weights, means, covariances = _maximization(
        values, responsibilities, reg_covar=DEFAULT_REG_COVAR
    )
    rotated_weights, rotated_means, rotated_covariances = _maximization(
        rotated_values, responsibilities, reg_covar=DEFAULT_REG_COVAR
    )
    np.testing.assert_allclose(rotated_weights, weights)
    np.testing.assert_allclose(rotated_means, means @ rotation.T)
    np.testing.assert_allclose(
        rotated_covariances,
        np.asarray([rotation @ covariance @ rotation.T for covariance in covariances]),
        rtol=1e-12,
        atol=1e-12,
    )
    posterior, likelihood = expectation(values, weights, means, covariances)
    rotated_posterior, rotated_likelihood = expectation(
        rotated_values, rotated_weights, rotated_means, rotated_covariances
    )
    np.testing.assert_allclose(rotated_posterior, posterior, rtol=1e-12, atol=1e-12)
    assert rotated_likelihood == pytest.approx(likelihood, rel=1e-12, abs=1e-12)


def test_diagnostic_ellipse_rotates_with_off_diagonal_covariance() -> None:
    from scripts import check_proximity_gmm as diagnostic

    ellipse = diagnostic.covariance_ellipse(
        np.asarray([2.0, 0.5]),
        np.asarray([[4.0, 1.5], [1.5, 1.0]]),
        component_index=0,
        standard_deviations=1.0,
    )

    assert not math.isclose(ellipse.angle % 90.0, 0.0, abs_tol=1e-12)
    assert ellipse.width > ellipse.height > 0.0


def _diagnostic_observations() -> pd.DataFrame:
    prompts = {
        "low": ((1.0, 0.08), (1.5, 0.11), (2.0, 0.07)),
        "positive": ((7.0, 0.60), (8.0, 0.64), (9.0, 0.68)),
        "constant": ((7.5, 0.66), (7.5, 0.66), (7.5, 0.66)),
        "intermittent": ((1.2, 0.10), (2.1, 0.09), (8.4, 0.69)),
        "rescued": ((1.2, 0.10), (2.1, 0.09), (8.4, 0.86)),
        "incomplete": ((1.1, 0.06), (7.7, 0.83), (math.nan, math.nan)),
    }
    return pd.DataFrame(
        [
            {
                "original_index": name,
                "seed": seed,
                "l2_norm": point[0],
                "sscd": point[1],
                "observation_status": "complete"
                if math.isfinite(point[0])
                else "missing",
                "observation_error": ""
                if math.isfinite(point[0])
                else "missing sample",
            }
            for name, points in prompts.items()
            for seed, point in zip((3, 4, 5), points, strict=True)
        ]
    )


def test_diagnostic_uses_same_seed_majority_decision_without_a_correlation_gate() -> None:
    from scripts import check_proximity_gmm as diagnostic
    from utils.data.selection import _apply_gmm_decisions

    frame = _diagnostic_observations()
    valid = diagnostic._valid_observations(frame)
    features, _, _ = standardize_features(
        frame.loc[valid, ["l2_norm", "sscd"]].to_numpy()
    )
    fit = fit_gaussian_mixture(features)
    annotated = diagnostic.annotate(frame, fit, reg_covar=DEFAULT_REG_COVAR)
    prompts = annotated.drop_duplicates("original_index").set_index("original_index")

    assert prompts.loc["positive", "computed_prompt_spearman"] > 0.0
    assert prompts.loc["positive", "gmm_selection_decision"] == "include"
    assert math.isnan(prompts.loc["constant", "computed_prompt_spearman"])
    assert prompts.loc["constant", "gmm_selection_decision"] == "include"
    assert prompts.loc["intermittent", "gmm_selection_decision"] == "discard"
    intermittent = annotated.loc[annotated["original_index"].eq("intermittent")]
    assert intermittent["gmm_component"].eq("high_sscd_mode").sum() == 1
    assert intermittent["sscd"].le(0.75).all()
    assert prompts.loc["rescued", "gmm_selection_decision"] == "include"
    rescued = annotated.loc[annotated["original_index"].eq("rescued")]
    assert rescued["gmm_component"].eq("low_sscd_mode").sum() == 2
    assert rescued["sscd"].gt(0.75).sum() == 1
    assert prompts.loc["low", "gmm_selection_decision"] == "discard"
    assert prompts.loc["incomplete", "gmm_selection_decision"] == "unusable"
    assert annotated.loc[valid, "gmm_component"].isin(COMPONENT_NAMES).all()
    assert annotated.loc[valid, "gmm_low_mode_probability"].between(0.0, 1.0).all()
    assert annotated.loc[~valid, "gmm_component"].eq("").all()
    assert annotated.loc[~valid, "gmm_low_mode_probability"].isna().all()

    expected = frame.copy()
    expected["include_prompt"] = False
    _apply_gmm_decisions(expected, "gmm")
    np.testing.assert_allclose(
        annotated.loc[valid, "gmm_low_mode_probability"],
        expected.loc[valid, "gmm_low_mode_probability"],
    )
    complete = valid.groupby(frame["original_index"]).transform("all")
    assert (
        annotated.loc[complete, "gmm_selection_decision"].eq("include").to_numpy()
        == expected.loc[complete, "include_prompt"].to_numpy()
    ).all()


def _diagnostic_posterior_fit(
    low_probabilities: list[float],
) -> GaussianMixtureFit:
    low = np.asarray(low_probabilities, dtype=np.float64)
    return GaussianMixtureFit(
        weights=np.asarray([0.5, 0.5]),
        means=np.asarray([[0.0, 0.0], [1.0, 1.0]]),
        covariances=np.asarray([np.eye(2), np.eye(2)]),
        responsibilities=np.column_stack((low, 1.0 - low)),
        log_likelihood=0.0,
        iterations=1,
    )


@pytest.mark.parametrize(
    ("low_probabilities", "maximum_sscd", "included"),
    [
        ([0.9, 0.8, 0.1], 0.75, False),
        ([0.9, 0.1, 0.1], 0.75, True),
        ([0.5, 0.1], 0.75, True),
        ([0.5, 0.5, 0.1], 0.75, False),
        ([0.5], 0.75, False),
        ([0.49], 0.75, True),
        ([0.9, 0.8, 0.1], np.nextafter(0.75, math.inf), True),
        ([0.9, 0.8, 0.1], 0.9, True),
    ],
)
def test_diagnostic_applies_shared_seed_majority_and_strict_high_sscd_override(
    low_probabilities: list[float], maximum_sscd: float, included: bool
) -> None:
    from scripts import check_proximity_gmm as diagnostic
    from utils.data.selection import _gmm_decision

    count = len(low_probabilities)
    scores = np.linspace(0.1, maximum_sscd, count)
    frame = pd.DataFrame(
        {
            "original_index": ["positive"] * count,
            "seed": np.arange(count),
            "l2_norm": np.arange(count, dtype=np.float64) + 1.0,
            "sscd": scores,
            "observation_status": ["complete"] * count,
        }
    )
    low = np.asarray(low_probabilities, dtype=np.float64)
    fit = _diagnostic_posterior_fit(low_probabilities)
    annotated = diagnostic.annotate(frame, fit, reg_covar=DEFAULT_REG_COVAR)

    decision = _gmm_decision(low_probabilities, scores.tolist())
    assert decision[0] is included
    if maximum_sscd > 0.75:
        assert decision[2] == "has_reference_seed_sscd_above_retention_threshold"
    elif 2 * sum(value >= 0.5 for value in low_probabilities) <= count:
        assert decision[2] == "no_low_sscd_mode_majority"
    else:
        assert decision[2] == "majority_reference_seeds_in_low_sscd_mode"
    assert annotated["gmm_selection_decision"].eq(
        "include" if included else "discard"
    ).all()
    np.testing.assert_array_equal(
        annotated["gmm_component"].eq("low_sscd_mode"), low >= 0.5
    )


@pytest.mark.parametrize("invalid_status", ["missing", "complete"])
def test_diagnostic_high_sscd_override_never_rescues_an_incomplete_or_invalid_prompt(
    invalid_status: str,
) -> None:
    from scripts import check_proximity_gmm as diagnostic

    frame = pd.DataFrame(
        {
            "original_index": ["incomplete"] * 3,
            "seed": [3, 4, 5],
            "l2_norm": [1.0, 2.0, 3.0],
            "sscd": [0.1, 0.9, math.nan],
            "observation_status": ["complete", "complete", invalid_status],
        }
    )
    fit = _diagnostic_posterior_fit([0.9, 0.9])
    annotated = diagnostic.annotate(frame, fit, reg_covar=DEFAULT_REG_COVAR)

    assert annotated["gmm_selection_decision"].eq("unusable").all()
    assert annotated.loc[:1, "gmm_component"].eq("low_sscd_mode").all()
    assert annotated.loc[2, "gmm_component"] == ""
    assert math.isnan(annotated.loc[2, "gmm_low_mode_probability"])


def test_diagnostic_loader_keeps_valid_siblings_of_incomplete_prompts(tmp_path) -> None:
    from scripts import check_proximity_gmm as diagnostic
    from utils.data.selection import SELECTION_COLUMNS

    frame = _diagnostic_observations()
    for column in SELECTION_COLUMNS:
        if column not in frame:
            frame[column] = ""
    frame["source_row_number"] = frame.groupby("original_index", sort=False).ngroup()
    frame["generated_image_tile_index"] = frame["seed"] - 3
    frame["include_prompt"] = False
    frame["prompt"] = frame["original_index"]
    frame["kind"] = "N"
    source = tmp_path / "selection.csv"
    frame.loc[:, SELECTION_COLUMNS].to_csv(source, index=False)

    loaded, seeds_per_prompt, incomplete_prompts = diagnostic.load_reference_points(
        source
    )
    assert len(loaded) == len(frame)
    assert seeds_per_prompt == 3
    assert incomplete_prompts == 1
    valid = diagnostic._valid_observations(loaded)
    assert valid.sum() == 17
    incomplete = loaded["original_index"].eq("incomplete")
    assert valid.loc[incomplete].sum() == 2
    assert not loaded["include_prompt"].any()
