"""Offline mathematical checks for cache-only theory reductions."""

import math

import pytest
import torch

from utils.experiments.theory.metrics import (
    branch_metrics,
    clean_estimates,
    initial_distribution_diagnostics,
)
from utils.experiments.theory.scheduler_adapter import SchedulerAdapter
from utils.experiments.theory.support import FiniteSupport
from utils.models.prediction_conversion import native_prediction_to_epsilon
from utils.models.schedule_metadata import build_schedule_metadata


def scheduler_fixture(name="ddim", native="epsilon", **kwargs):
    import diffusers
    from diffusers import DDIMScheduler, DDPMScheduler

    kind = DDIMScheduler if name == "ddim" else DDPMScheduler
    scheduler = kind(
        num_train_timesteps=31, clip_sample=False, prediction_type=native, **kwargs
    )
    schedule = build_schedule_metadata(
        scheduler, name, num_inference_steps=6, device="cpu"
    )
    adapter = SchedulerAdapter(
        schedule, recorded_diffusers_version=diffusers.__version__
    )
    return scheduler, adapter


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_branch_promotion_units_identity_and_evidence(dtype):
    z = torch.tensor([[[[1024.0, 1.125], [0.125, -2.0]]]], dtype=dtype)
    u = torch.tensor([[[[1023.0, 1.0], [0.25, -1.0]]]], dtype=dtype)
    c = torch.tensor([[[[1022.0, 0.5], [0.125, -0.5]]]], dtype=dtype)
    target = torch.zeros((1, 2, 2), dtype=dtype)
    result = branch_metrics(z, u, c, target, 0.1, math.sqrt(0.99), -2, center=target)
    expected = (z.double() - math.sqrt(0.99) * c.double()) / 0.1
    assert result["e_c_rmse"].dtype == torch.float64
    assert torch.allclose(
        result["e_c_rmse"], torch.linalg.vector_norm(expected, dim=(1, 2, 3)) / 2
    )
    assert torch.allclose(result["e_c_l2"], 2 * result["e_c_rmse"])
    assert torch.allclose(
        result["effective_target_identity_residual"],
        torch.zeros(1, dtype=torch.float64),
        atol=1e-9,
    )
    assert result["effective_target_residual_evidence"] == "exact_cache_algebra"
    assert result["population_forward_loss"] is None
    assert "forward_corruption" in result["population_forward_loss_status"]
    assert "conditional_loss" not in result
    assert result["projection_status"].startswith("degenerate")
    assert torch.isnan(result["a_parallel"]).all()
    assert torch.isfinite(result["e_c_rmse"]).all()


def test_invalid_sigma_preserves_branch_measurements():
    z = torch.ones(2, 1, 2, 2)
    result = branch_metrics(z, z, z, z[0], 1.0, 0.0, 7.5, center=z[0])
    assert torch.isfinite(result["e_c_rmse"]).all()
    assert torch.isnan(result["effective_target_noise_residual_mse"]).all()
    assert result["effective_target_identity_status"] != "valid"
    result = branch_metrics(z, z, z, z[0], 0.0, 1.0, 7.5)
    assert torch.isnan(result["e_c_rmse"]).all()
    assert torch.isfinite(result["state_error_rmse"]).all()


def test_injection_keeps_negative_direction_and_off_target():
    target = torch.tensor([[[1.0, 0.0]]])
    z = torch.zeros(1, 1, 1, 2)
    u = torch.zeros_like(z)
    c = torch.tensor([[[[1.0, -2.0]]]])
    result = branch_metrics(
        z,
        u,
        c,
        target,
        math.sqrt(0.5),
        math.sqrt(0.5),
        2.0,
        center=torch.zeros_like(target),
    )
    assert result["a_parallel"].item() == pytest.approx(-2)
    assert result["off_target"].item() == pytest.approx(4)
    assert result["target_direction_cosine"].item() < 0


@pytest.mark.parametrize("name", ["ddim", "ddpm"])
@pytest.mark.parametrize("native", ["epsilon", "v_prediction", "sample"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_matched_scheduler_parity_canonical_native_types(name, native, dtype):
    scheduler, adapter = scheduler_fixture(name, native)
    generator = torch.Generator().manual_seed(812)
    z = torch.randn((2, 1, 2, 2), generator=generator, dtype=dtype)
    u = torch.randn(z.shape, generator=generator, dtype=dtype)
    c = torch.randn(z.shape, generator=generator, dtype=dtype)
    g = -1.5
    for k, t in enumerate(scheduler.timesteps):
        # Same RNG seed -> exactly the same stochastic innovation, no separate
        # chain. Production adapter never draws this comparator noise.
        kwargs = (
            {"eta": 0.0}
            if name == "ddim"
            else {"generator": torch.Generator().manual_seed(17)}
        )
        observed = scheduler.step(u + g * (c - u), t, z, **kwargs).prev_sample
        kwargs = (
            {"eta": 0.0}
            if name == "ddim"
            else {"generator": torch.Generator().manual_seed(17)}
        )
        expected_u = scheduler.step(u, t, z, **kwargs).prev_sample
        eu = native_prediction_to_epsilon(u, z, t, scheduler)
        ec = native_prediction_to_epsilon(c, z, t, scheduler)
        diagnostics, matched, _ = adapter.matched_update(z, observed, eu, ec, g, k)
        tolerance = diagnostics["update_rounding_sensitivity_rmse"] + 2e-5
        error = (matched - expected_u.double()).square().mean(dim=(1, 2, 3)).sqrt()
        assert torch.all(error <= tolerance)
        if dtype == torch.float64:
            assert torch.allclose(matched, expected_u.double(), atol=2e-5, rtol=2e-5)
        assert diagnostics["scheduler_status"] == "valid_affine"
        if name == "ddpm" and int(t) > 0:
            assert torch.isnan(diagnostics["deterministic_replay_rmse"]).all()
        else:
            assert torch.all(diagnostics["deterministic_replay_rmse"] <= tolerance)


@pytest.mark.parametrize("spacing", ["leading", "trailing", "linspace"])
@pytest.mark.parametrize("set_alpha_to_one", [False, True])
def test_ddim_actual_destination_and_terminal_alpha(spacing, set_alpha_to_one):
    scheduler, adapter = scheduler_fixture(
        "ddim", timestep_spacing=spacing, set_alpha_to_one=set_alpha_to_one
    )
    for k, t in enumerate(scheduler.timesteps):
        coeff = adapter.coefficients(k)
        assert coeff.destination_timestep == int(t) - 31 // 6
        cumulative = (
            scheduler.alphas_cumprod[coeff.destination_timestep]
            if coeff.destination_timestep >= 0
            else scheduler.final_alpha_cumprod
        )
        assert coeff.destination_alpha**2 == pytest.approx(float(cumulative))
    with pytest.raises(IndexError):
        adapter.coefficients(6)  # There is no terminal branch prediction.
    if spacing == "leading":
        assert adapter.coefficients(5).destination_sigma == pytest.approx(
            0 if set_alpha_to_one else math.sqrt(1 - float(scheduler.alphas_cumprod[0]))
        )


@pytest.mark.parametrize("name", ["ddim", "ddpm"])
@pytest.mark.parametrize("thresholding", [False, True])
def test_nonlinear_drift_supported_but_affine_bound_inapplicable(name, thresholding):
    import diffusers
    from diffusers import DDIMScheduler, DDPMScheduler

    cls = DDIMScheduler if name == "ddim" else DDPMScheduler
    scheduler = cls(
        num_train_timesteps=31, clip_sample=not thresholding, thresholding=thresholding
    )
    schedule = build_schedule_metadata(
        scheduler, name, num_inference_steps=6, device="cpu"
    )
    adapter = SchedulerAdapter(
        schedule, recorded_diffusers_version=diffusers.__version__
    )
    z = torch.tensor([[[[3.0, -4.0], [2.0, -1.0]]]], dtype=torch.float64)
    u, c = 0.5 * z, -0.25 * z
    t = scheduler.timesteps[0]
    kwargs = (
        {"eta": 0.0}
        if name == "ddim"
        else {"generator": torch.Generator().manual_seed(2)}
    )
    observed = scheduler.step(u + 7.5 * (c - u), t, z, **kwargs).prev_sample
    kwargs = (
        {"eta": 0.0}
        if name == "ddim"
        else {"generator": torch.Generator().manual_seed(2)}
    )
    expected = scheduler.step(u, t, z, **kwargs).prev_sample
    diagnostics, matched, _ = adapter.matched_update(z, observed, u, c, 7.5, 0)
    assert torch.allclose(matched, expected, atol=2e-5, rtol=2e-5)
    assert not diagnostics["affine_applicable"]
    assert torch.isnan(diagnostics["predicted_shift_rmse"]).all()
    assert adapter.endpoint_bound({}, 0, 7.5, 4)["operational_bound_rmse"] is None


@pytest.mark.parametrize("variance", ["learned", "learned_range", "fixed_large_log"])
def test_unsupported_variance_explicit(variance):
    _, adapter = scheduler_fixture("ddpm", variance_type=variance)
    assert adapter.coefficients(0).status.startswith("unsupported")
    assert adapter.endpoint_bound({}, 0, 7.5, 4)["operational_bound_rmse"] is None


@pytest.mark.parametrize("guidance", [-3, 0, 0.5, 1, 7.5])
def test_prospective_endpoint_bound_includes_target_offset(guidance):
    _, adapter = scheduler_fixture("ddim", set_alpha_to_one=False, steps_offset=1)
    k = len(adapter.timesteps) - 1
    coeff = adapter.coefficients(k)
    z = torch.tensor([[[[3.0, 2.0]]]], dtype=torch.float64)
    u, c = torch.ones_like(z), -torch.ones_like(z)
    target = torch.tensor([[[2.0, -1.0]]], dtype=torch.float64)
    metrics = branch_metrics(z, u, c, target, coeff.alpha, coeff.sigma, guidance)
    bound = adapter.endpoint_bound(metrics, k, guidance, 2)
    mg = clean_estimates(z, u, c, coeff.alpha, coeff.sigma, guidance)[3]
    endpoint = coeff.A * z + coeff.B * mg
    observed = (endpoint - target).square().mean().sqrt()
    assert bound["operational_bound_rmse"].item() >= observed.item() - 1e-12
    assert bound["bound_target_offset_component"].item() > 0
    assert bound["bound_noise_component"] == 0


def test_known_variance_noise_bound_has_no_endpoint_dependency():
    _, adapter = scheduler_fixture("ddpm")
    metrics = {
        "state_error_rmse": 0.0,
        "e_u_rmse": 0.0,
        "e_c_rmse": 0.0,
        "target_rmse": 0.0,
    }
    bound = adapter.endpoint_bound(metrics, 0, 7.5, 4)
    from scipy.stats import chi2

    expected = adapter.coefficients(0).noise_std * math.sqrt(chi2.ppf(0.95, 4) / 4)
    assert bound["bound_noise_component"] == pytest.approx(expected)
    assert bound["noise_bound_coverage_scope"] == "per_update"
    import inspect

    assert "z_next" not in inspect.signature(adapter.endpoint_bound).parameters


def support_fixture(**kwargs):
    atoms = torch.tensor(
        [[[[0.0, 0.0]]], [[[1.0, 0.0]]], [[[0.0, 2.0]]]], dtype=torch.float64
    )
    bank = FiniteSupport.from_candidates(
        [(str(i), x) for i, x in enumerate(atoms)], **kwargs
    )
    return atoms, bank


@pytest.mark.parametrize("candidate_chunk", [1, 2, 100])
def test_support_bruteforce_raw_l2_and_chunking(candidate_chunk):
    atoms, bank = support_fixture(candidate_chunk=candidate_chunk, query_chunk=1)
    z = torch.tensor([[[[0.9, 0.1]]], [[[0.0, 1.5]]]], dtype=torch.float64)
    a, s = 0.8, 0.6
    result = bank.evaluate(z, "1", a, s, unconditional=torch.ones_like(z))
    logits = -((z[:, None] - a * atoms[None]).square().sum(dim=(2, 3, 4))) / (2 * s * s)
    probability = logits.softmax(1)
    target_lp = logits[:, 1] - torch.logsumexp(logits, 1)
    assert torch.allclose(result["target_log_probability"], target_lp, atol=1e-12)
    assert torch.allclose(
        result["target_log_odds"],
        logits[:, 1] - torch.logsumexp(logits[:, [0, 2]], 1),
        atol=1e-12,
    )
    assert torch.allclose(
        result["posterior_entropy"],
        -(probability * probability.log()).sum(1),
        atol=1e-12,
    )
    mean = probability @ atoms.reshape(3, -1)
    assert torch.allclose(
        result["posterior_mean_target_rmse"],
        (mean - atoms[1].flatten()).norm(dim=1) / math.sqrt(2),
    )
    assert not torch.allclose(target_lp, (logits / 2).log_softmax(1)[:, 1])
    assert torch.all(result["surrogate_triangle_slack_rmse"] >= -1e-12)
    assert torch.all(
        result["posterior_weighted_target_radius_rmse"]
        <= result["surrogate_concentration_bound_rmse"] + 1e-12
    )


def test_support_duplicate_atoms_aliases_and_identity_conflicts():
    target = torch.zeros(1, 2, 2)
    bank = FiniteSupport.from_candidates(
        [("01", target), ("alias", target.clone()), ("different", target + 1)]
    )
    assert bank.size == 2 and bank.aliases["01"] == bank.aliases["alias"]
    assert bank.weights.tolist() == [0.5, 0.5]
    with pytest.raises(ValueError, match="conflicting"):
        FiniteSupport.from_candidates([("01", target), ("01", target + 1)])
    with pytest.raises(ValueError, match="conflicting"):
        FiniteSupport.from_candidates(
            [("01", target), ("02", target + 1)],
            identities={"01": "same-image", "02": "same-image"},
        )


def test_support_extreme_snr_and_true_zero_noise_exclusion():
    atoms, bank = support_fixture(candidate_chunk=1)
    values = bank.evaluate(atoms[1:2], "1", 1.0, 1e-8)
    assert values["target_log_probability"].item() == pytest.approx(0)
    assert math.isfinite(values["target_log_odds"].item())
    assert values["posterior_entropy"].item() == pytest.approx(0)
    invalid = bank.evaluate(atoms[1:2], "1", 1.0, 0.0)
    assert torch.isnan(invalid["target_log_probability"]).all()
    assert "sigma" in invalid["support_status"]


@pytest.mark.parametrize("candidate_chunk", [1, 2, 100])
def test_feedback_bounds_signed_gain_and_bruteforce(candidate_chunk):
    _, bank = support_fixture(candidate_chunk=candidate_chunk)
    z_u = torch.tensor([[[[0.1, 0.3]]], [[[0.5, 1.0]]]], dtype=torch.float64)
    z_g = torch.tensor([[[[1.0, 0.1]]], [[[-1.0, 2.0]]]], dtype=torch.float64)
    result = bank.feedback(z_u, z_g, "1", 0.8, 0.6)
    expected = (
        bank.evaluate(z_g, "1", 0.8, 0.6)["target_log_odds"]
        - bank.evaluate(z_u, "1", 0.8, 0.6)["target_log_odds"]
    )
    assert torch.allclose(result["log_odds_gain"], expected, atol=1e-12)
    assert torch.all(result["margin_min"] <= result["log_odds_gain"] + 1e-12)
    assert torch.all(result["log_odds_gain"] <= result["margin_max"] + 1e-12)
    assert result["margin_min"][0] > 0 and result["log_odds_gain"][0] > 0
    assert result["log_odds_gain"][1] < 0
    assert result["margin_bound_covered"].tolist() == [1.0, 1.0]
    tie = bank.feedback(z_u, z_u, "1", 0.8, 0.6)
    assert tie["positive_gain"].tolist() == [0.0, 0.0]


def test_gaussian_distribution_shift_is_total_dimension_analytic():
    values = initial_distribution_diagnostics(
        torch.ones(2, 2, 2), 0.1, math.sqrt(0.99), 1.0
    )
    assert values["initial_signal_energy"] == pytest.approx(8 * 0.01 / 0.99)
    assert values["initial_gaussian_kl"] > 0.5 * values["initial_signal_energy"]
    assert "quantized" in values["initial_distribution_status"]


def test_invalid_matched_state_keeps_valid_guided_posterior():
    atoms, bank = support_fixture()
    result = bank.feedback(
        torch.full_like(atoms[:1], torch.nan), atoms[:1], "0", 0.8, 0.6
    )
    assert result["feedback_status"] == "invalid_nonfinite_matched_or_guided_state"
    assert torch.isnan(result["margin_min"]).all()
    assert torch.isfinite(result["guided_target_log_probability"]).all()
    invalid = bank.evaluate(torch.full_like(atoms[:1], torch.nan), "0", 0.8, 0.6)
    assert invalid["support_status"] == "invalid_nonfinite_query"


def test_support_deduplicates_signed_zero():
    bank = FiniteSupport.from_candidates(
        [("a", torch.tensor([0.0])), ("b", torch.tensor([-0.0]))]
    )
    assert bank.size == 1 and bank.aliases["a"] == bank.aliases["b"]


def test_target_offset_term_is_necessary_when_all_preupdate_errors_vanish():
    _, adapter = scheduler_fixture("ddim", set_alpha_to_one=False, steps_offset=1)
    step = len(adapter.timesteps) - 1
    coeff = adapter.coefficients(step)
    target = torch.tensor([[[3.0, -2.0]]], dtype=torch.float64)
    z = target[None].clone()
    epsilon = (z - coeff.alpha * target) / coeff.sigma
    metrics = branch_metrics(
        z, epsilon, epsilon, target, coeff.alpha, coeff.sigma, -4.0
    )
    bound = adapter.endpoint_bound(metrics, step, -4.0, target.numel())
    observed = abs(coeff.A + coeff.B - 1) * target.square().mean().sqrt()
    assert observed > 0
    assert bound["operational_bound_rmse"].item() == pytest.approx(
        observed.item(), abs=1e-13
    )
    assert bound["bound_state_component"].item() == 0
    assert bound["bound_unconditional_component"].item() == pytest.approx(0, abs=1e-13)


def test_bad_deterministic_replay_is_exposed():
    scheduler, adapter = scheduler_fixture()
    z = torch.zeros(1, 1, 2, 2)
    u = torch.ones_like(z)
    c = -u
    t = scheduler.timesteps[0]
    observed = scheduler.step(u + 7.5 * (c - u), t, z, eta=0.0).prev_sample
    result, _, _ = adapter.matched_update(z, observed + 100.0, u, c, 7.5, 0)
    assert result["deterministic_replay_rmse"].item() > 99
    assert not result["deterministic_replay_within_sensitivity"].item()
