"""Author-run direct math checks; no model download or dataset is required."""
import math

import pytest
import torch

from utils.experiments.theory import direct_integration
from utils.experiments.theory import direct_measurements as direct
from utils.experiments.theory.reference_law import reference_law_from_atoms
from utils.experiments.theory.scheduler_adapter import SchedulerAdapter, UpdateCoefficients


def law(atoms=((0.0, 0.0), (3.0, 0.0)), weights=None):
    return reference_law_from_atoms(
        [torch.tensor(atom, dtype=torch.float64) for atom in atoms],
        [f"atom-{index}" for index in range(len(atoms))], weights=weights,
    )


def adapter(*, noisy=False, clean=True):
    result = SchedulerAdapter.__new__(SchedulerAdapter)
    sigma = math.sqrt(0.19)
    a = sigma / 0.6
    result.timesteps = torch.tensor([1, 0])
    result._coefficients = [
        UpdateCoefficients(0, 1, 0, 0.8, 0.6, 0.8**2 / 0.6**2,
                           0.9, sigma, a, 0.9 - a * 0.8,
                           0.1 if noisy else 0.0, not noisy, True, "valid_affine"),
        UpdateCoefficients(1, 0, -1, 0.9, sigma, 0.9**2 / 0.19,
                           1.0 if clean else 0.999, 0.0 if clean else 0.01,
                           0.0 if clean else 0.02, 1.0 if clean else 0.97,
                           1e-30 if noisy else 0.0, not noisy, True, "valid_affine"),
    ]
    return result


def test_reference_mean_explicit_mass_deduplication_and_exact_radius():
    reference = law(((2.0, 1.0), (2.0, 1.0), (8.0, 1.0)), weights=[1.0, 2.0, 1.0])
    assert reference.support.size == 2
    torch.testing.assert_close(reference.mean_vector, torch.tensor([3.5, 1.0], dtype=torch.float64))
    assert reference.target_radius("atom-0") == 6.0
    assert reference.support.aliases["atom-0"] == reference.support.aliases["atom-1"]
    assert torch.equal(reference.to("cpu").support.weights, reference.support.weights)
    # No supplied masses means equal DISTINCT atoms, not three recovered records.
    uniform = law(((2.0, 1.0), (2.0, 1.0), (8.0, 1.0)))
    torch.testing.assert_close(uniform.mean_vector, torch.tensor([5.0, 1.0], dtype=torch.float64))


def test_law_hash_is_independent_of_chunking_and_zero_mass_is_explicit():
    atoms = [torch.tensor([1.0, 0.0]), torch.tensor([5.0, 0.0])]
    first = reference_law_from_atoms(atoms, ["one", "two"], candidate_chunk_size=1, query_chunk_size=1)
    second = reference_law_from_atoms(atoms, ["one", "two"], candidate_chunk_size=7, query_chunk_size=4)
    assert first.law_hash == second.law_hash
    zero = reference_law_from_atoms(atoms, ["one", "two"], weights=[0.0, 1.0])
    assert zero.support.size == 1 and zero.metadata["zero_mass_source_ids"] == ["one"]
    with pytest.raises(ValueError, match="no positive-mass atom"):
        zero.target_id_for(atoms[0])


def test_reference_vector_uses_raw_gaussian_distance_and_nonzero_mean():
    reference = law(((2.0, 1.0), (6.0, 1.0)), weights=[1.0, 3.0])
    z = torch.tensor([[1.1, 0.7]], dtype=torch.float64)
    expected_weights = (
        torch.tensor([0.25, 0.75], dtype=torch.float64).log()
        - (z[:, None] - 0.8 * reference.support.flat[None]).square().sum(-1) / (2 * 0.6**2)
    ).softmax(-1)
    expected = expected_weights @ reference.support.flat
    torch.testing.assert_close(reference.posterior_mean(z, 0.8, 0.6), expected)
    previous = reference.support.evaluate(z, "atom-0", 0.8, 0.6)
    extended = reference.support.evaluate(z, "atom-0", 0.8, 0.6, include_mean=True)
    for key, value in previous.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, extended[key], equal_nan=True)
        else:
            assert value == extended[key]


def test_corollary3_full_vector_discrepancies_have_different_guidance_coefficients():
    center = torch.tensor([2.0, 3.0], dtype=torch.float64)
    target = torch.tensor([4.0, 3.0], dtype=torch.float64)
    mu = torch.tensor([[2.0, 4.0]], dtype=torch.float64)
    mc = target[None].clone()
    values = direct.corollary3_metrics(mu, mc, target, center, 2.0)
    assert values["direct_cor3_injection_discrepancy_l2"].item() == 2.0
    assert values["direct_cor3_injection_bound_l2"].item() == 2.0
    assert values["direct_cor3_guided_discrepancy_l2"].item() == 1.0
    assert values["direct_cor3_guided_bound_l2"].item() == 1.0
    # The target projection agrees perfectly while the full vector fails.
    discrepancy = 2 * (mc - mu) - 2 * (target - center)
    assert (discrepancy * (target - center)).sum() == 0
    assert discrepancy.norm() > 0
    assert values["direct_cor3_injection_vector_identity_residual_l2"].item() == 0


def test_lemma4_independent_counterfactual_cannot_fit_an_altered_endpoint():
    scheduler = adapter()
    z = torch.tensor([[0.5, -0.5]], dtype=torch.float64)
    eu, ec = z * 0.1, z * -0.1
    coeff = scheduler.coefficients(0)
    mu, mc, _, mg = direct.clean_estimates(z, eu, ec, coeff.alpha, coeff.sigma, 2.0, latent_ndim=1)
    endpoint = coeff.A * z + coeff.kappa * mg
    good, cf, _ = scheduler.direct_matched_update(z, endpoint, eu, ec, 2.0, 0, latent_ndim=1)
    bad, altered_cf, _ = scheduler.direct_matched_update(z, endpoint + 0.01, eu, ec, 2.0, 0, latent_ndim=1)
    assert good["direct_lemma4_independent"]
    torch.testing.assert_close(cf, coeff.A * z + coeff.kappa * mu)
    assert torch.equal(cf, altered_cf)
    assert bad["direct_lemma4_vector_residual_l2"].item() > 0
    assert not bad["direct_lemma4_numeric_within_sensitivity"].item()


def test_ddpm_constructed_noise_and_nonaffine_are_not_independent_checks():
    z = torch.tensor([[0.5, -0.5]], dtype=torch.float64)
    scheduler = adapter(noisy=True)
    result, _, _ = scheduler.direct_matched_update(z, z + 0.1, z * 0, z * 0.1, 2.0, 0, latent_ndim=1)
    assert not result["direct_lemma4_independent"]
    assert "constructed" in result["direct_lemma4_verification_source"]
    with pytest.raises(ValueError, match="verified saved/replayed noise"):
        scheduler.direct_matched_update(z, z, z, z, 2.0, 0, latent_ndim=1,
                                        realized_noise=z, noise_provenance="endpoint_fitted")
    coeff = scheduler.coefficients(0)
    scheduler._coefficients[0] = UpdateCoefficients(**{**coeff.__dict__, "affine": False})
    result, _, _ = scheduler.direct_matched_update(z, z, z, z, 2.0, 0, latent_ndim=1)
    assert not result["direct_lemma4_applicable"]


def test_lemma6_all_current_states_and_wrong_target_branch_agreement():
    reference = law()
    target = reference.support.atoms[0]
    state = torch.tensor([[2.0, 0.0]], dtype=torch.float64)
    wrong = torch.tensor([[100.0, 100.0]], dtype=torch.float64)
    values, _, _ = direct.current_reference_metrics(state, wrong, wrong, target, reference, "atom-0", 0.9, math.sqrt(0.19))
    assert values["direct_lemma6_gap_l2"].item() == 0
    assert values["direct_conditional_error_l2"].item() > 100
    assert values["direct_lemma6_rhs_l2"].item() > 100
    assert values["direct_target_radius_l2"] == 3.0
    assert values["direct_reference_target_error_l2"].item() <= values["direct_radius_tail_l2"].item() + 1e-12


def test_small_vector_and_nonzero_probability_underflow_do_not_become_exact_zero():
    assert direct._norm(torch.tensor([[1e-300, 0.0]], dtype=torch.float64), 1).item() > 0
    reference = law(((0.0, 0.0), (1000.0, 0.0)))
    zero = torch.zeros(1, 2, dtype=torch.float64)
    values, _, _ = direct.current_reference_metrics(zero, zero, zero, zero[0], reference, "atom-0", 0.9, 0.01)
    assert values["direct_radius_tail_underflow"].item()
    assert not values["direct_lemma6_zero_rhs_is_exact"].item()
    assert torch.isfinite(values["direct_target_log_complement"]).all()


def test_singleton_proposition5_has_exact_zero_gain_and_original_negative_margin():
    reference = law(((2.0, 1.0),))
    target = reference.support.atoms[0]
    mu = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    mc = torch.tensor([[3.0, 0.0]], dtype=torch.float64)
    coeff = adapter().coefficients(0)
    state = mu.clone()
    metrics, weights, mean = direct.current_reference_metrics(state, mu, mc, target, reference, "atom-0", coeff.alpha, coeff.sigma)
    cf = coeff.A * state + coeff.kappa * mu
    out = cf + 2 * coeff.kappa * (mc - mu)
    result = direct.proposition5_metrics(state, mu, mc, cf, out, target, "atom-0", reference,
                                         coeff, weights, mean, torch.zeros(1), torch.ones(1, dtype=torch.bool), 2.0)
    assert result["direct_prop5_applicable"]
    assert result["direct_prop5_log_probability_gain"].item() == 0
    assert result["direct_prop5_variation_l2"].item() == 0
    assert result["direct_prop5_lower_bound"].item() < 0
    assert torch.isnan(result["direct_prop5_log_odds_gain"]).all()


def test_integration_exception_retains_independently_computed_endpoint_gain(monkeypatch):
    reference = law()
    target = reference.support.atoms[0]
    state = torch.tensor([[0.2, 0.1]], dtype=torch.float64)
    mu, mc = state * 0, state * -0.1
    coeff = adapter().coefficients(0)
    _, weights, mean = direct.current_reference_metrics(state, mu, mc, target, reference, "atom-0", coeff.alpha, coeff.sigma)
    cf = coeff.A * state + coeff.kappa * mu
    out = cf + 2 * coeff.kappa * (mc - mu)
    def failed(*args, **kwargs):
        raise RuntimeError("synthetic integral interruption")
    monkeypatch.setattr(direct_integration, "integration_metrics", failed)
    result = direct.proposition5_metrics(state, mu, mc, cf, out, target, "atom-0", reference,
                                         coeff, weights, mean, torch.zeros(1), torch.ones(1, dtype=torch.bool), 2.0)
    assert result["direct_prop5_status"] == "failed_integration"
    assert torch.isfinite(result["direct_prop5_log_probability_gain"]).all()
    assert torch.isnan(result["direct_prop5_variation_l2"]).all()


@pytest.mark.parametrize("noise,clean", [(False, True), (True, True), (False, False)])
def test_theorem7_uses_actual_endpoint_and_keeps_inapplicable_pairs(noise, clean):
    reference = law()
    scheduler = adapter(noisy=noise, clean=clean)
    coeff = scheduler.coefficients(1)
    target = reference.support.atoms[0]
    state = torch.tensor([[0.8, 0.2]], dtype=torch.float64)
    eu, ec = state * 0, state * 0.2
    mu, mc, _, mg = direct.clean_estimates(state, eu, ec, coeff.alpha, coeff.sigma, 2.0, latent_ndim=1)
    observed = mg if clean else 0.02 * state + 0.97 * mg
    current, _, _ = direct.current_reference_metrics(state, mu, mc, target, reference, "atom-0", coeff.alpha, coeff.sigma)
    terminal = scheduler.terminal_diagnostics(state, observed, eu, ec, 2.0, 1, target)
    result = direct.theorem7_metrics(mu, mc, target, observed, reference, current, terminal, 2.0, 1.0)
    torch.testing.assert_close(result["direct_theorem7_endpoint_error_l2"], (observed - target).norm(dim=1))
    assert torch.isfinite(result["direct_theorem7_bound_l2"]).all()
    assert bool(result["direct_theorem7_applicable"].item()) == (clean and not noise)
    if noise or not clean:
        assert torch.isnan(result["direct_theorem7_applicable_slack_l2"]).all()
        assert not result["direct_theorem7_condition_certified"].item()


def narrow_transition(*, integrate=False, payload_callback=None, config=None):
    reference = law(((-1.0, 0.0), (1.0, 0.0)))
    target = reference.support.atoms[0]
    state = torch.tensor([[-1.0, 0.0]], dtype=torch.float64)
    mu, mc = state.clone(), torch.tensor([[2.0, 0.0]], dtype=torch.float64)
    coeff = UpdateCoefficients(0, 1, 0, 1.0, 0.05, 400.0,
                               1.0, 1e-4, 0.5, 0.5, 0.0, True, True, "synthetic_affine")
    _, weights, mean = direct.current_reference_metrics(
        state, mu, mc, target, reference, "atom-0", coeff.alpha, coeff.sigma,
    )
    result = direct.proposition5_metrics(
        state, mu, mc, state, mc, target, "atom-0", reference, coeff,
        weights, mean, torch.zeros(1, dtype=torch.float64),
        torch.ones(1, dtype=torch.bool), 2.0,
        integrate=integrate, payload_callback=payload_callback, config=config,
    )
    return reference, result


def test_original_variation_narrow_transition_signed_integral_and_negative_lower_bound():
    from utils.experiments.theory.candidate_integration import IntegrationConfig

    captured = []
    reference, core = narrow_transition(payload_callback=lambda *args: captured.append(args))
    assert len(captured) == 1
    payload, base = captured[0]
    observed_gain = core["direct_prop5_log_probability_gain"].clone()
    # Integrate the saved payload alone, without any generated state/predictions.
    result = direct_integration.integrate_proposition5_payload(
        reference.support, payload, base,
        config=IntegrationConfig(absolute_tolerance=1e-8, relative_tolerance=1e-8),
    )
    assert torch.equal(result["direct_prop5_log_probability_gain"], observed_gain)
    assert torch.isnan(core["direct_prop5_variation_l2"]).all()
    assert result["direct_prop5_variation_l2"].item() == pytest.approx(4 / 3, abs=2e-7)
    assert result["direct_prop5_margin_l2"].item() == pytest.approx(-4 / 3, abs=2e-7)
    assert result["direct_prop5_log_probability_gain"].item() < 0
    assert result["direct_prop5_lower_bound"].item() < 0
    torch.testing.assert_close(
        result["direct_prop5_signed_integral"] * result["direct_prop5_prefactor"],
        result["direct_prop5_integrated_gain"],
    )
    assert "not_certified" in result["direct_prop5_certification_status"]


def test_exhausted_integral_keeps_endpoint_sign_and_never_certifies_condition():
    from utils.experiments.theory.candidate_integration import IntegrationConfig

    captured = []
    reference, core = narrow_transition(payload_callback=lambda *args: captured.append(args))
    result = direct_integration.integrate_proposition5_payload(
        reference.support, *captured[0],
        config=IntegrationConfig(max_evaluations=15, absolute_tolerance=1e-12, relative_tolerance=1e-12),
    )
    assert result["direct_prop5_quadrature_budget_exhausted"].item()
    assert result["direct_prop5_integral_status"] == ["numerically_unresolved"]
    assert result["direct_prop5_condition_status"] == ["numerically_unresolved"]
    assert torch.equal(result["direct_prop5_log_probability_gain"], core["direct_prop5_log_probability_gain"])
    assert result["direct_prop5_gain_status"] == ["negative"]


@pytest.mark.parametrize("singleton", [False, True])
def test_shared_pass_has_all_current_states_raw_terminal_and_separate_integral_receipt(singleton):
    reference = law(((2.0, 1.0),) if singleton else ((2.0, 1.0), (3.0, 0.0)))
    scheduler = adapter()
    target = reference.support.atoms[0]
    z = torch.zeros(2, 3, 2, dtype=torch.float64)
    z[:, 0] = torch.tensor([[0.2, 0.3], [0.5, 0.1]], dtype=torch.float64)
    u = torch.zeros(2, 2, 2, dtype=torch.float64)
    c = torch.full_like(u, 0.1)
    for step in range(2):
        coeff = scheduler.coefficients(step)
        _, _, _, mg = direct.clean_estimates(
            z[:, step], u[:, step], c[:, step], coeff.alpha, coeff.sigma, 2.0, latent_ndim=1,
        )
        z[:, step + 1] = coeff.A * z[:, step] + coeff.kappa * mg
    metadata = {"seeds": [4, 9], "scientific_config_hash": "run", "original_index": "007",
                "record_id": "record", "target_image_sha256": "image-sha",
                "tensor_file_sha256": {"target_latent": "latent-file-sha"},
                "stored_prediction_type": "epsilon"}
    captured = []
    result = direct.measure_direct_record(
        (z, u, c, target), metadata, reference,
        {"num_inference_steps": 2, "guidance_scale": 2.0}, adapter=scheduler,
        terminal_sscd=[0.2, 0.9], integration=False,
        payload_callback=lambda *args: captured.append(args),
    )
    assert result["audit"]["complete"] and result["audit"]["core_complete"]
    assert result["audit"]["integration_complete"] == singleton
    assert result["audit"]["no_pending_integrals"] == singleton
    assert len(captured) == (0 if singleton else 1)
    assert len(result["initial"]) == 2
    assert len(result["trajectory"]) == 4
    assert set(result["trajectory"].step_index) == {0, 1}
    assert result["trajectory"].direct_unconditional_reference_error_l2.notna().all()
    assert set(result["initial"].target_id) == {"image-sha"}
    assert set(result["initial"].target_latent_sha256) == {"latent-file-sha"}
    expected = (z[:, -1] - target).norm(dim=1)
    torch.testing.assert_close(
        torch.tensor(result["terminal"].direct_theorem7_endpoint_error_l2.tolist()), expected,
        check_dtype=False,
    )
    assert result["terminal"].terminal_sscd.tolist() == [0.2, 0.9]
    assert set(result["trajectory"].input_source) == {"generated_state"}


def test_target_outside_manifest_law_preserves_computable_reference_comparisons():
    reference = law()
    target = torch.tensor([7.0, 1.0], dtype=torch.float64)
    state = torch.tensor([[0.1, 0.2]], dtype=torch.float64)
    metrics, _, _ = direct.current_reference_metrics(
        state, state, state + 1, target, reference, None, 0.8, 0.6,
    )
    assert not metrics["direct_reference_target_positive_mass"]
    assert metrics["direct_non_target_mass"].item() == 1
    assert metrics["direct_target_log_probability"].item() == -math.inf
    assert torch.isfinite(metrics["direct_unconditional_reference_error_l2"]).all()
    assert metrics["direct_target_radius_l2"] == pytest.approx(math.sqrt(50))


def test_exact_prompt_ambiguity_uses_distinct_atoms_and_preserves_absent_target_scope():
    from utils.experiments.theory.reference_law import _prompt_scope

    reference = reference_law_from_atoms(
        [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), torch.tensor([1.0, 0.0])],
        ["a", "alias-a", "None"],
    )
    audit = _prompt_scope([
        {"prompt_raw": "same", "candidate_id": "a"},
        {"prompt_raw": "same", "candidate_id": "alias-a"},
        {"prompt_raw": "different", "candidate_id": "a"},
        {"prompt_raw": "different", "candidate_id": "None"},
        {"prompt_raw": "missing", "candidate_id": None, "target_atom_sha256": "outside"},
        {"prompt_raw": " different", "candidate_id": "a"},
    ], reference)
    assert audit["exact_prompt_count"] == 4
    assert audit["ambiguous_exact_prompt_count"] == 1
    assert audit["source_records_without_positive_reference_atom"] == 1
    assert audit["record_policy"] == "retain_all_records_without_assumption_based_selection"


def test_finite_payload_and_nullable_scalar_parquet_resume_round_trip(tmp_path):
    import pandas as pd
    from utils.common.io import atomic_torch_save, atomic_write_frame_parquet, safe_torch_load

    captured = []
    reference, core = narrow_transition(payload_callback=lambda *args: captured.append(args))
    payload, base = captured[0]
    # The strict tensor cache stores integrator inputs only. NaN placeholders
    # and legitimate infinite log annotations belong in scalar Parquet.
    payload_path = tmp_path / "integrator.pt"
    atomic_torch_save(payload, payload_path)
    base["direct_test_undefined"] = torch.tensor([torch.nan], dtype=torch.float64)
    base["direct_test_negative_infinity"] = torch.tensor([-torch.inf], dtype=torch.float64)
    base["direct_test_positive_infinity"] = torch.tensor([torch.inf], dtype=torch.float64)
    scalar_path = tmp_path / "scalars.parquet"
    atomic_write_frame_parquet(pd.DataFrame(direct._rows(base, [{"seed": 11}])), scalar_path)
    saved = pd.read_parquet(scalar_path)
    restored = {column: saved[column].tolist() for column in saved if column != "seed"}
    result = direct_integration.integrate_proposition5_payload(
        reference.support, safe_torch_load(payload_path), restored,
    )
    assert math.isnan(result["direct_test_undefined"][0])
    assert result["direct_test_negative_infinity"] == [-math.inf]
    assert result["direct_test_positive_infinity"] == [math.inf]
    assert result["direct_prop5_log_probability_gain"] == core["direct_prop5_log_probability_gain"].tolist()
    assert torch.isfinite(result["direct_prop5_variation_l2"]).all()
    assert result["direct_prop5_status"] == "matched_finite_law_comparison"


def test_payload_persistence_failure_does_not_erase_computed_endpoint_gain():
    from utils.common.io import CacheIOError

    def failed_receipt(*args):
        raise CacheIOError("synthetic payload storage failure")

    _, expected = narrow_transition()
    _, actual = narrow_transition(payload_callback=failed_receipt)
    assert actual["direct_prop5_status"] == "failed_integration"
    assert actual["direct_prop5_failure_phase"] == "payload_persistence"
    assert actual["direct_prop5_integral_status"] == ["failed"]
    assert actual["direct_prop5_applicable"]
    assert torch.equal(actual["direct_prop5_log_probability_gain"], expected["direct_prop5_log_probability_gain"])
    assert torch.equal(actual["direct_prop5_log_odds_gain"], expected["direct_prop5_log_odds_gain"])
    assert torch.isnan(actual["direct_prop5_variation_l2"]).all()
    assert "synthetic payload storage failure" in actual["direct_prop5_integration_error"]


@pytest.mark.parametrize("mean_source", [
    "initial_unconditional_reference_monte_carlo",
    "minimum_snr_unconditional_reference_monte_carlo",
])
def test_reference_mean_receipt_and_device_roundtrip_preserve_the_exact_atom_law(mean_source):
    from utils.experiments.theory.reference_law import ReferenceLaw
    from utils.experiments.theory.support import _tensor_hash
    reference = law(((2., 1.), (6., -1.)), weights=[1., 3.])
    estimate = torch.tensor([1., 2.], dtype=torch.float64)
    receipt = {"source": mean_source, "vector_sha256": _tensor_hash(estimate), "sample_count": 10000}
    metadata = reference.metadata | {"theory_mean": receipt}
    selected = ReferenceLaw(reference.support, metadata, "selected-mean-fixture", estimate)
    query = torch.tensor([[.3, -.1]], dtype=torch.float64)
    torch.testing.assert_close(selected.mean_vector, reference.mean_vector, rtol=0, atol=0)
    torch.testing.assert_close(selected.theory_mean_vector, estimate, rtol=0, atol=0)
    torch.testing.assert_close(selected.posterior_mean(query, .6, .8), reference.posterior_mean(query, .6, .8), rtol=0, atol=0)
    moved = selected.to("cpu")
    torch.testing.assert_close(moved.theory_mean_vector, estimate, rtol=0, atol=0)
    assert moved.theory_mean_metadata == receipt
    torch.testing.assert_close(moved.to_payload()["estimated_mean_vector"], estimate, rtol=0, atol=0)
    with pytest.raises(ValueError, match="lacks its saved vector"):
        ReferenceLaw(reference.support, metadata, "selected-mean-fixture")
    with pytest.raises(ValueError, match="differs from its receipt"):
        ReferenceLaw(reference.support, metadata, "selected-mean-fixture", estimate + 1)
    assert selected.target_radius("atom-0") == reference.target_radius("atom-0")
    unconditional = torch.tensor([[4., 3.]], dtype=torch.float64)
    conditional = torch.tensor([[2., 1.]], dtype=torch.float64)
    target = reference.support.atoms[0]
    old, _, _ = direct.current_reference_metrics(query, unconditional, conditional, target,
                                                 reference, "atom-0", .6, .8)
    current, _, _ = direct.current_reference_metrics(query, unconditional, conditional, target,
                                                     selected, "atom-0", .6, .8)
    for field in ("direct_target_log_probability", "direct_target_log_complement",
                  "direct_unconditional_reference_error_l2", "direct_radius_tail_l2", "direct_lemma6_rhs_l2"):
        torch.testing.assert_close(current[field], old[field], rtol=0, atol=0)
    torch.testing.assert_close(current["direct_reference_to_bank_mean_error_l2"], old["direct_reference_mean_offset_l2"], rtol=0, atol=0)
    assert not torch.equal(current["direct_reference_mean_offset_l2"], old["direct_reference_mean_offset_l2"])
