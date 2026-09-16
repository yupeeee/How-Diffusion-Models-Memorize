"""Code-only authored checks for the fixed evidence supplement; not executed."""
from dataclasses import replace
import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from utils.experiments.theory.evidence_measurements import (
    derive_gaussian_noise_contract,
    injection_geometry,
    measure_evidence_record,
    paired_target_metrics,
    terminal_predictor_bounds,
    terminal_update_qa,
)
from utils.experiments.theory.scheduler_adapter import UpdateCoefficients


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def coefficient(**changes):
    base = UpdateCoefficients(1, 0, -1, 0.8, 0.6, 16 / 9,
                              1.0, 0.0, 0.0, 1.0, 0.0, True, True, "valid_affine")
    return replace(base, **changes)


def reference(mu, mc, target, g=2.0, mean=None):
    mean = torch.zeros_like(target) if mean is None else mean
    return {
        "guidance_scale": g,
        "direct_conditional_error_l2": (mc - target).norm(dim=1),
        "direct_unconditional_reference_error_l2": (mu - mean).norm(dim=1),
        # The two fixed atoms target/mean with p_target=0 give this exact bound.
        "direct_radius_tail_l2": (mean - target).norm().expand(len(mu)),
    }


def predict(state, mu, mc, target, coeff=None, **kwargs):
    return terminal_predictor_bounds(
        state, mu, mc, target, coefficient() if coeff is None else coeff,
        reference(mu, mc, target), final_update=True, run_scope="run-a",
        terminal_update_count=100, **kwargs,
    )


def gaussian_contract():
    return {
        "supported": True, "family": "conditional_isotropic_gaussian",
        "conditional_independence_from_preupdate_history": True,
        "scalar_standard_deviation": True, "reason": "declared_test_contract",
    }


def test_full_geometry_retains_large_off_axis_error_at_unit_alignment():
    mu, mc = tensor([[0, 0]]), tensor([[1, 4]])
    target, mean = tensor([2, 3]), tensor([1, 3])
    result = injection_geometry(mu, mc, target, mean, 7.5)
    assert result["evidence_injection_q"].item() == 1
    assert result["evidence_injection_r_perp"].item() == 4
    assert result["evidence_injection_relative_error"].item() == 4
    assert result["evidence_injection_absolute_error_l2"].item() == 30
    assert result["evidence_injection_status"] == ["valid"]


def test_injection_orthogonality_and_full_error_identity():
    mu = tensor([[1, 1], [2, -1], [0, 0]])
    mc = tensor([[3, 7], [-4, 1], [2, 1]])
    target, mean = tensor([3, 3]), tensor([1, 2])
    values = injection_geometry(mu, mc, target, mean, 2)
    q, rp, error = (values[name] for name in (
        "evidence_injection_q", "evidence_injection_r_perp", "evidence_injection_relative_error"))
    torch.testing.assert_close(error.square(), (q - 1).square() + rp.square())
    torch.testing.assert_close(values["evidence_injection_perpendicular_orthogonality"], torch.zeros(3, dtype=torch.float64), atol=1e-13, rtol=0)
    scaled = injection_geometry(mu, mc, target, mean, 7.5)
    for name in ("evidence_injection_q", "evidence_injection_r_perp", "evidence_injection_relative_error"):
        torch.testing.assert_close(values[name], scaled[name], atol=0, rtol=0)


@pytest.mark.parametrize("unresolved", [False, True])
def test_zero_or_arithmetic_unresolved_direction_retains_absolute_errors(unresolved):
    target = tensor([1.0, 1.0])
    mean = target.clone()
    if unresolved:
        mean[0] = torch.nextafter(mean[0], tensor(0.0))
    result = injection_geometry(tensor([[0, 0]]), tensor([[0, 4]]), target, mean, 2)
    assert not result["evidence_injection_applicable"].item()
    assert result["evidence_injection_status"] == ["arithmetic_unresolved_target_direction" if unresolved else "exact_zero_target_direction"]
    assert torch.isnan(result["evidence_injection_relative_error"]).all()
    assert result["evidence_injection_absolute_error_l2"].item() > 0


def test_small_representable_direction_has_no_absolute_cutoff():
    result = injection_geometry(tensor([[0, 0]]), tensor([[1e-300, 1e-300]]), tensor([1e-300, 0]), tensor([0, 0]), 2)
    assert result["evidence_injection_applicable"].item()
    assert result["evidence_injection_q"].item() == pytest.approx(1)
    assert result["evidence_injection_r_perp"].item() == pytest.approx(1)


def test_joint_target_error_exposes_identical_wrong_branch_vectors():
    wrong, target = tensor([[100, 100]]), tensor([0, 0])
    values = paired_target_metrics(wrong, wrong, target, reference_rhs_l2=tensor([300]))
    assert (wrong - wrong).norm() == 0
    assert values["evidence_joint_target_error_l2"].item() > 100
    assert values["evidence_joint_bound_slack_l2"].item() > 0


def test_terminal_predictor_api_has_no_endpoint_argument():
    names = inspect.signature(terminal_predictor_bounds).parameters
    assert not any(name in names for name in ("observed", "endpoint", "observed_endpoint", "z_next", "x_out"))
    assert all(parameter.kind != inspect.Parameter.VAR_KEYWORD for parameter in names.values())
    with pytest.raises(TypeError, match="unexpected keyword"):
        terminal_predictor_bounds(None, None, None, None, None, None, observed_endpoint=tensor([[0, 0]]))


def test_clean_case_has_zero_defect_and_original_scope():
    state, mu, mc, target = tensor([[1, 3]]), tensor([[0, 0]]), tensor([[1, 1]]), tensor([0, 0])
    result = predict(state, mu, mc, target)
    assert result.scalars["evidence_terminal_original_clean_structural"]
    assert result.scalars["evidence_terminal_scope"] == "original_clean_terminal_theorem"
    assert not result.scalars["evidence_terminal_manuscript_extension_required"]
    assert result.scalars["evidence_terminal_defect_bound_l2"].item() == 0
    torch.testing.assert_close(result.predicted_endpoint, result.guided_clean)


def test_deterministic_defect_is_preupdate_and_altered_endpoint_only_fails_qa():
    state, mu, mc, target = tensor([[1, 3]]), tensor([[0, 0]]), tensor([[1, 1]]), tensor([0, 0])
    coeff = coefficient(A=0.2, B=0.9, destination_alpha=0.99, destination_sigma=0.1)
    result = predict(state, mu, mc, target, coeff)
    expected = 0.2 * state - 0.1 * result.guided_clean
    torch.testing.assert_close(result.predicted_defect, expected)
    before = result.scalars["evidence_terminal_corrected_ref_bound_l2"].clone()
    good = terminal_update_qa(result.predicted_endpoint, result, source_epsilon=torch.finfo(torch.float64).eps)
    bad = terminal_update_qa(result.predicted_endpoint + 1, result, source_epsilon=torch.finfo(torch.float64).eps)
    assert good["evidence_terminal_reconstruction_status"] == ["consistent_with_source_precision"]
    assert bad["evidence_terminal_reconstruction_status"] == ["failed_reconstruction"]
    assert torch.equal(before, result.scalars["evidence_terminal_corrected_ref_bound_l2"])
    assert not result.scalars["evidence_terminal_original_clean_structural"]
    assert result.scalars["evidence_terminal_manuscript_extension_required"]


def test_known_independent_additive_innovation_is_pathwise_but_recovered_one_is_rejected():
    state, mu, mc, target = tensor([[1, 3]]), tensor([[0, 0]]), tensor([[1, 1]]), tensor([0, 0])
    coeff = coefficient(noise_std=0.25, deterministic=False)
    innovation = tensor([[0.2, -0.3]])
    result = predict(state, mu, mc, target, coeff, independent_innovation=innovation,
                     innovation_provenance="independently_saved_additive_noise")
    torch.testing.assert_close(result.predicted_defect, innovation)
    assert "pathwise" in result.scalars["evidence_terminal_certainty"]
    with pytest.raises(ValueError, match="forbids endpoint-recovered"):
        predict(state, mu, mc, target, coeff, independent_innovation=innovation,
                innovation_provenance="observed_endpoint_minus_guided_drift")


def test_gaussian_union_bound_uses_fixed_run_scope_and_keeps_noise_failures_separate():
    from scipy.stats import chi2

    state, mu, mc, target = tensor([[1, 3]]), tensor([[0, 0]]), tensor([[1, 1]]), tensor([0, 0])
    coeff = coefficient(A=0.2, B=0.9, noise_std=0.25, deterministic=False)
    result = predict(state, mu, mc, target, coeff, noise_contract=gaussian_contract())
    assert result.predicted_endpoint is None
    expected_q = float(chi2.isf(0.05 / 100, 2)) ** 0.5
    assert result.scalars["evidence_terminal_noise_chi2_norm_quantile"] == pytest.approx(expected_q)
    assert result.scalars["evidence_terminal_noise_per_update_alpha"] == 0.05 / 100
    assert result.scalars["evidence_terminal_noise_run_scope"] == "run-a"
    assert "trajectory_independence_not_required" in result.scalars["evidence_terminal_noise_probability_scope"]
    assert result.scalars["evidence_terminal_scope"] == "finite_terminal_update_extension_gaussian_noise_bound"
    qa = terminal_update_qa(result.deterministic_drift + 100, result, source_epsilon=torch.finfo(torch.float64).eps)
    assert qa["evidence_terminal_noise_bound_exceeded"].item()
    assert qa["evidence_terminal_reconstruction_status"] == ["unavailable_independent_innovation_for_update_reconstruction"]
    assert result.scalars["evidence_terminal_applicable"].item()


@pytest.mark.parametrize("changes", [dict(affine=False), dict(B=0), dict(B=-1)])
def test_unsupported_terminal_contract_never_becomes_clean_or_corrected(changes):
    state, mu, mc, target = tensor([[1, 3]]), tensor([[0, 0]]), tensor([[1, 1]]), tensor([0, 0])
    result = predict(state, mu, mc, target, coefficient(**changes))
    assert result.scalars["evidence_terminal_scope"] == "unavailable_unsupported_terminal_contract"
    assert not result.scalars["evidence_terminal_applicable"].item()
    assert torch.isnan(result.scalars["evidence_terminal_defect_bound_l2"]).all()


def test_tiny_nonzero_noise_does_not_relax_original_clean_gate():
    state, mu, mc, target = tensor([[1, 3]]), tensor([[0, 0]]), tensor([[1, 1]]), tensor([0, 0])
    result = predict(state, mu, mc, target, coefficient(noise_std=1e-30, deterministic=False), noise_contract=gaussian_contract())
    assert not result.scalars["evidence_terminal_original_clean_structural"]
    assert result.scalars["evidence_terminal_defect_bound_l2"].item() > 0


def test_gaussian_contract_requires_saved_version_and_fixed_variance():
    adapter = SimpleNamespace(name="ddpm", version_status="verified_same_diffusers_version", variance_type="fixed_small",
                              nonlinear=False, recorded_version="v", installed_version="v")
    assert derive_gaussian_noise_contract(adapter, {"sampler_contract_version": 2})["supported"]
    adapter.variance_type = "learned_range"
    assert not derive_gaussian_noise_contract(adapter, {"sampler_contract_version": 2})["supported"]
    adapter.variance_type = "fixed_small"
    adapter.version_status = "unrecorded_diffusers_version"
    assert not derive_gaussian_noise_contract(adapter, {"sampler_contract_version": 2})["supported"]


def test_deterministic_terminal_sample_bounds_imply_cdf_order_on_all_change_points():
    state = tensor([[0, 0], [1, 3], [-2, 1]])
    mu, mc, target = tensor([[0, 0], [0, 0], [0, 0]]), tensor([[0, 0], [1, 1], [-1, 2]]), tensor([0, 0])
    result = predict(state, mu, mc, target, coefficient(A=0.2, B=0.9))
    actual = (result.predicted_endpoint - target).norm(dim=1)
    obs = result.scalars["evidence_terminal_corrected_obs_bound_l2"]
    ref = result.scalars["evidence_terminal_corrected_ref_bound_l2"]
    assert actual[0] == 0 and obs[0] == 0 and ref[0] == 0
    for tolerance in torch.cat([actual, obs, ref]).unique():
        assert (ref <= tolerance).double().mean() <= (obs <= tolerance).double().mean()
        assert (obs <= tolerance).double().mean() <= (actual <= tolerance).double().mean()


@pytest.mark.parametrize("readonly_numpy_views", [False, True])
def test_supplement_uses_saved_scalars_without_reference_evaluation_and_preserves_missingness(monkeypatch, readonly_numpy_views):
    if readonly_numpy_views:
        original_to_numpy = pd.Series.to_numpy

        def to_numpy(series, *args, **kwargs):
            values = original_to_numpy(series, *args, **kwargs)
            if not kwargs.get("copy", False):
                # Reproduce pandas' read-only view even on older pandas versions.
                values = values.view()
                values.setflags(write=False)
            return values

        monkeypatch.setattr(pd.Series, "to_numpy", to_numpy)

    original_as_tensor = torch.as_tensor

    def as_tensor(values, *args, **kwargs):
        if isinstance(values, np.ndarray):
            # Do not rely on PyTorch's warning, which fires only once per process.
            assert values.flags.writeable, "A read-only pandas view reached torch.as_tensor"
        return original_as_tensor(values, *args, **kwargs)

    monkeypatch.setattr(torch, "as_tensor", as_tensor)
    seeds = [4, 9]
    target = tensor([2, 0])
    z = torch.zeros(2, 3, 2, dtype=torch.float64)
    u = torch.zeros(2, 2, 2, dtype=torch.float64)
    c = torch.zeros_like(u)
    metadata = {"scientific_config_hash": "run-a", "seeds": seeds, "stored_prediction_type": "epsilon", "sampler_contract_version": 2}
    rows = []
    for step, seed in ((1, 9), (0, 4), (0, 9), (1, 4)):
        rows.append({"run_id": "run-a", "original_index": "007", "record_id": "record",
                     "target_id": "target", "seed": seed, "step_index": step,
                     "direct_conditional_error_l2": 2.0,
                     "direct_unconditional_target_error_l2": float("nan") if (step, seed) == (0, 9) else 2.0,
                     "direct_unconditional_reference_error_l2": 1.0,
                     "direct_radius_tail_l2": 1.0,
                     "direct_lemma6_rhs_l2": 4.0,
                     "direct_theorem7_applicable": False})
    core = pd.DataFrame(rows)
    tables = {"initial": core.loc[core.step_index.eq(0)].copy(),
              "trajectory": core.copy(), "terminal": core.loc[core.step_index.eq(1)].copy()}
    untouched = {name: frame.copy(deep=True) for name, frame in tables.items()}

    def forbidden(*args, **kwargs):
        raise AssertionError("supplement must not evaluate a posterior or learned model")

    law = SimpleNamespace(support=SimpleNamespace(flat=torch.zeros(2, 2, dtype=torch.float64)),
                          mean_vector=tensor([1, 0]), posterior_mean=forbidden, posterior=forbidden)
    adapter = SimpleNamespace(name="ddim", version_status="verified_same_diffusers_version",
                              nonlinear=False, variance_type="fixed_small", recorded_version="v", installed_version="v",
                              coefficients=lambda step: replace(coefficient(), step_index=step))
    result = measure_evidence_record((z, u, c, target), metadata, law,
                                     {"num_inference_steps": 2, "guidance_scale": 2.0}, adapter,
                                     terminal_update_count=2, run_scope="run-a", core_tables=tables)
    assert result["initial"].seed.tolist() == seeds
    assert len(result["trajectory"]) == 4
    missing = result["trajectory"].loc[lambda f: f.seed.eq(9) & f.step_index.eq(0)]
    assert missing.evidence_joint_target_error_l2.isna().all()
    assert set(result["trajectory"].step_index) == {0, 1}
    assert not any(name.startswith("direct_") for name in result["terminal"].columns)
    for name, frame in tables.items():
        pd.testing.assert_frame_equal(frame, untouched[name], check_exact=True)


def test_nonzero_innovation_cannot_be_inserted_into_deterministic_contract():
    state, mu, mc, target = tensor([[1, 3]]), tensor([[0, 0]]), tensor([[1, 1]]), tensor([0, 0])
    with pytest.raises(ValueError, match="contradicts the deterministic"):
        predict(state, mu, mc, target, independent_innovation=tensor([[1, 0]]),
                innovation_provenance="independently_saved_additive_noise")


def test_stochastic_noise_without_a_verified_law_is_explicitly_unavailable():
    state, mu, mc, target = tensor([[1, 3]]), tensor([[0, 0]]), tensor([[1, 1]]), tensor([0, 0])
    result = predict(state, mu, mc, target, coefficient(noise_std=0.25, deterministic=False))
    assert result.scalars["evidence_terminal_scope"] == "unavailable_unsupported_terminal_contract"
    assert not result.scalars["evidence_terminal_applicable"].item()
    assert "unverified_terminal_Gaussian_variance_contract" in result.scalars["evidence_terminal_status"][0]
