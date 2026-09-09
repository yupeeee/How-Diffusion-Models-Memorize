"""Focused tests for deterministic proximity GMM fitting."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from utils.data.proximity_gmm import (
    COMPONENT_NAMES,
    DEFAULT_REG_COVAR,
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
        np.linalg.cholesky(covariance)

    responsibilities, log_likelihood = expectation(
        features, fit.weights, fit.means, fit.covariances
    )
    np.testing.assert_allclose(responsibilities, fit.responsibilities)
    assert log_likelihood == pytest.approx(fit.log_likelihood)

    means, covariances = raw_parameters(fit, feature_mean, feature_scale)
    assert means[0, 1] < means[1, 1]
    assert covariances.shape == (2, 2, 2)
    np.testing.assert_array_equal(fit.covariances[:, 0, 1], 0.0)
    np.testing.assert_array_equal(fit.covariances[:, 1, 0], 0.0)
    np.testing.assert_array_equal(covariances[:, 0, 1], 0.0)
    np.testing.assert_array_equal(covariances[:, 1, 0], 0.0)


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


def test_diagonal_maximization_uses_weighted_variances_and_existing_floor() -> None:
    values = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
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
        expected_variance = np.average(
            (values - means[component]) ** 2,
            axis=0,
            weights=responsibilities[:, component],
        )
        np.testing.assert_allclose(
            covariances[component],
            np.diag(np.maximum(expected_variance, DEFAULT_REG_COVAR)),
        )
        assert covariances[component, 1, 1] == DEFAULT_REG_COVAR


def test_diagonal_fit_selects_highest_likelihood_deterministic_start() -> None:
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
    np.testing.assert_array_equal(actual.responsibilities, expected.responsibilities)


def test_diagonal_fit_reduces_tilted_high_mode_spillover_into_bottom_cloud() -> None:
    rng = np.random.default_rng(92)
    low = np.column_stack((rng.normal(150, 12, 600), rng.normal(0.06, 0.02, 600)))
    high_scores = rng.normal(0.55, 0.18, 300)
    high = np.column_stack(
        (210 - 200 * high_scores + rng.normal(0, 10, 300), high_scores)
    )
    bottom_tail = np.column_stack((rng.normal(195, 4, 20), rng.normal(0.1, 0.01, 20)))
    features, _, _ = standardize_features(np.vstack((low, high, bottom_tail)))
    diagonal = fit_gaussian_mixture(features)
    diagonal_high = diagonal.responsibilities.argmax(axis=1) == 1

    # Low-score tail observations must not be assigned to the high component.
    assert not diagonal_high[900:].any()
    assert not diagonal_high[:600].any()
    # Strong high observations remain high, even when a prompt has many low seeds.
    assert diagonal_high[600:900][high_scores > 0.5].all()
    intermittent_prompt = np.concatenate(
        (diagonal_high[:19], diagonal_high[600:900][high_scores > 0.5][:1])
    )
    assert intermittent_prompt.sum() == 1
    assert intermittent_prompt.any()


def _diagnostic_observations() -> pd.DataFrame:
    prompts = {
        "low": ((1.0, 0.08), (1.5, 0.11), (2.0, 0.07)),
        "positive": ((7.0, 0.75), (8.0, 0.80), (9.0, 0.85)),
        "constant": ((7.5, 0.82), (7.5, 0.82), (7.5, 0.82)),
        "intermittent": ((1.2, 0.10), (2.1, 0.09), (8.4, 0.86)),
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


def test_diagnostic_uses_same_any_high_decision_without_a_correlation_gate() -> None:
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
    assert prompts.loc["intermittent", "gmm_selection_decision"] == "include"
    intermittent = annotated.loc[annotated["original_index"].eq("intermittent")]
    assert intermittent["gmm_component"].eq("high_sscd_mode").sum() == 1
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
    assert valid.sum() == 14
    incomplete = loaded["original_index"].eq("incomplete")
    assert valid.loc[incomplete].sum() == 2
    assert not loaded["include_prompt"].any()
