"""Offline integration tests with real tiny cache tensors and frozen metadata.

The fixture uses scheduler arithmetic only. It never loads a model or downloads
weights. It deliberately retains low-SSCD experiment seeds, duplicate raw
prompts, leading-zero IDs, and nonselected candidate targets.
"""

from __future__ import annotations

import copy
import hashlib
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from utils.common.io import (
    atomic_torch_save,
    atomic_write_json,
    canonical_hash,
    file_sha256,
    read_json,
)
from utils.data.selection import build_target_pair_selection
from utils.experiments.cache import (
    generation_paths,
    list_completed_records,
    publish_completion_marker,
    save_generation_tensors,
)
from utils.experiments.sscd import (
    SSCDPaths,
    SSCD_SCHEMA_VERSION,
    SCORE_DEFINITION,
    sscd_configuration_hash,
)
from utils.experiments.theory import cache_reader, reduce as reducer
from utils.experiments.theory.contracts import (
    TheoryError,
    analysis_parent,
    find_analysis_bundle,
    validate_source_metadata,
)
from utils.experiments.theory.plotting import (
    DEFAULT_FIGURES,
    FIGURE_ORDER,
    render_bundle,
    validate_bundle,
)
from utils.experiments.theory.statements import (
    STATEMENT_IDS,
    load_statement_registry,
    paper_unavailable_metrics,
    posterior_feedback_condition,
    synchronization_bound,
    terminal_reproduction_bound,
    validate_statement_registry,
)
from utils.models.schedule_metadata import build_schedule_metadata

CONFIG = dict(
    model_name="sdv1",
    scheduler_name="ddim",
    guidance_scale=7.5,
    num_inference_steps=3,
    num_seeds=3,
)
IDS = ("00001", "00002", "00003", "00004")


def _forbidden(*args, **kwargs):
    raise AssertionError(
        "Model, upstream inference or raw tensor access is forbidden here"
    )


@pytest.fixture
def saved_cache(tmp_path, monkeypatch):
    import diffusers
    from diffusers import DDIMScheduler

    scheduler = DDIMScheduler(num_train_timesteps=12, clip_sample=False)
    schedule = build_schedule_metadata(
        scheduler, "ddim", num_inference_steps=3, device="cpu"
    )
    records = pd.DataFrame(
        [
            dict(
                model_name="sdv1",
                original_index=index,
                record_id=f"record-{index}",
                source_row_number=i,
                prompt="  duplicated raw prompt\n",
                kind="N",
                target_image_sha256=hashlib.sha256(index.encode()).hexdigest(),
            )
            for i, index in enumerate(IDS)
        ]
    )
    role_rows, paths_by_role, runs, sscd_by_role = {}, {}, {}, {}
    for role, seed_start in (("experiment", 0), ("reference", 3)):
        paths = generation_paths(tmp_path, **CONFIG, seed_start=seed_start)
        paths.create()
        paths_by_role[role] = paths
        seeds = list(range(seed_start, seed_start + 3))
        science = dict(
            model_cli_name="sdv1",
            dataset_model="sdv1",
            model_id="synthetic/model",
            model_revision="a" * 40,
            vae_id="synthetic/vae",
            vae_revision="b" * 40,
            scheduler={
                "name": "ddim",
                "class": schedule["scheduler_class"],
                "config": schedule["scheduler_config"],
            },
            guidance_scale=7.5,
            num_inference_steps=3,
            num_seeds=3,
            seeds=seeds,
            stored_prediction_type="epsilon",
            native_prediction_type="epsilon",
            trajectory_order="noise_to_image",
            latent_shape=[1, 2, 2],
            target_preprocessing={"policy": "synthetic_scaled_latent"},
            target_latent_definition="synthetic_posterior_mode_times_scale",
            package_versions={"diffusers": diffusers.__version__},
        )
        run = dict(
            scientific_config=science, scientific_config_hash=canonical_hash(science)
        )
        runs[role] = run
        atomic_write_json(paths.run_config, run)
        atomic_torch_save(schedule, paths.schedule)
        score_paths = SSCDPaths(paths.run_directory)
        score_paths.create()
        sscd = dict(
            generation_scientific_config_hash=run["scientific_config_hash"],
            num_seeds=3,
            seeds=seeds,
            sscd_checkpoint_sha256="c" * 64,
            sscd_preprocessing_hash="d" * 64,
        )
        sscd["configuration_hash"] = sscd_configuration_hash(sscd)
        sscd_by_role[role] = sscd
        atomic_write_json(score_paths.config_json, sscd)
        observations = []
        for position, row in enumerate(records.to_dict("records")):
            index = row["original_index"]
            target = torch.tensor([[[0.2, -0.1], [0.4, -0.2]]]) * (position + 1)
            state = torch.stack(
                [
                    torch.randn(1, 2, 2, generator=torch.Generator().manual_seed(seed))
                    for seed in seeds
                ]
            )
            states, us, cs = [state.clone()], [], []
            for t in scheduler.timesteps:
                u = 0.03 * state
                c = (
                    u
                    + 0.03 * (position + 1)
                    + torch.arange(3).reshape(3, 1, 1, 1) * 0.01
                )
                us.append(u.clone())
                cs.append(c.clone())
                state = scheduler.step(u + 7.5 * (c - u), t, state, eta=0.0).prev_sample
                states.append(state.clone())
            latents, u, c = (
                torch.stack(states, 1),
                torch.stack(us, 1),
                torch.stack(cs, 1),
            )
            tensors = dict(
                latent=latents,
                unconditional_noise_predictions=u,
                conditional_noise_predictions=c,
                target_latent=target,
            )
            hashes = save_generation_tensors(
                paths,
                index,
                latents=latents,
                unconditional_predictions=u,
                conditional_predictions=c,
                target_latent=target,
            )
            metadata = dict(
                row,
                prompt_raw=row["prompt"],
                scientific_config_hash=run["scientific_config_hash"],
                tensor_file_sha256=hashes,
                seeds=seeds,
                num_inference_steps=3,
                tensor_shapes={
                    key: list(value.shape) for key, value in tensors.items()
                },
                tensor_dtypes={key: "float32" for key in tensors},
            )
            for key in (
                "model_id",
                "model_revision",
                "vae_id",
                "vae_revision",
                "stored_prediction_type",
                "native_prediction_type",
                "target_preprocessing",
                "latent_shape",
            ):
                metadata[key] = science[key]
            publish_completion_marker(paths, index, metadata)
            # Selection separates reference clouds. Actual experimental scores
            # include failures even for every selected prompt.
            values = (
                ([0.1, 0.12, 0.08] if position < 2 else [0.91, 0.87, 0.93])
                if role == "reference"
                else [0.97, 0.03, 0.42]
            )
            scores = torch.tensor(values, dtype=torch.float32)
            score_hash = atomic_torch_save(scores, score_paths.score_path(index))
            marker = dict(
                schema_version=SSCD_SCHEMA_VERSION,
                original_index=index,
                record_id=row["record_id"],
                source_row_number=position,
                num_seeds=3,
                seeds=seeds,
                sscd_configuration_hash=sscd["configuration_hash"],
                generation_latent_sha256=hashes["latent"],
                generation_scientific_config_hash=run["scientific_config_hash"],
                target_image_sha256=row["target_image_sha256"],
                prompt_raw=row["prompt"],
                score_shape=[3],
                score_dtype="float32",
                terminal_latent_index=-1,
                similarity=SCORE_DEFINITION,
                score_sha256=score_hash,
            )
            atomic_write_json(score_paths.marker_path(index), marker)
            distances = (latents[:, -1] - target).flatten(1).norm(dim=1)
            for offset, seed in enumerate(seeds):
                observations.append(
                    dict(
                        row,
                        seed=seed,
                        l2_norm=float(distances[offset]),
                        sscd=float(scores[offset]),
                        observation_status="complete",
                        observation_error="",
                    )
                )
        role_rows[role] = pd.DataFrame(observations)
    selection = build_target_pair_selection(
        tmp_path,
        **CONFIG,
        overwrite=False,
        paired_frame=role_rows["reference"],
        records_frame=records,
        reference_run_config=runs["reference"],
        sscd_config=sscd_by_role["reference"],
    )
    assert {"00003", "00004"}.issubset(selection.included_indices)
    experiment = paths_by_role["experiment"]
    proximity_dir = (
        tmp_path
        / "outputs"
        / experiment.run_directory.parent.name
        / "proximity"
        / experiment.run_directory.name
    )
    proximity_dir.mkdir(parents=True)
    frame = role_rows["experiment"].copy()
    frame["include_prompt"] = frame.original_index.isin(selection.included_indices)
    # Deliberately reverse rows: joins must use actual seed, never row order.
    frame.iloc[::-1].to_csv(proximity_dir / "proximity.csv", index=False)
    atomic_write_json(
        proximity_dir / "run_config.json",
        dict(
            selection_hash=selection.sha256,
            generation_scientific_config_hash=runs["experiment"][
                "scientific_config_hash"
            ],
            sscd_configuration_hash=sscd_by_role["experiment"]["configuration_hash"],
            seeds=[0, 1, 2],
            distance="euclidean_l2",
        ),
    )
    from utils.models import loading, latent
    from utils.experiments import generation, sscd as sscd_module

    for module, name in (
        (loading, "load_model_components"),
        (loading, "load_vae_from_generation_config"),
        (latent, "encode_target_latent"),
        (latent, "decode_generated_latents"),
        (generation, "generate_webster_trajectories"),
        (sscd_module, "run_sscd"),
    ):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, _forbidden)
    monkeypatch.setattr(torch.nn.Module, "__call__", _forbidden)
    monkeypatch.setattr(torch.jit, "load", _forbidden)
    return dict(
        root=tmp_path,
        paths=paths_by_role,
        selection=selection,
        proximity=proximity_dir,
        records=records,
    )


def _reduce(fixture, **kwargs):
    return reducer.run_theory(fixture["root"], **CONFIG, **kwargs)


def test_real_cache_reader_string_ids_seed_roles_and_same_seed_scores(saved_cache):
    sources = cache_reader.discover_sources(saved_cache["root"], **CONFIG)
    assert all(
        isinstance(record.original_index, str)
        and record.original_index.startswith("000")
        for record in sources.selected
    )
    assert sources.runs["experiment"]["scientific_config"]["seeds"] == [0, 1, 2]
    assert sources.runs["reference"]["scientific_config"]["seeds"] == [3, 4, 5]
    for record in sources.selected:
        z, u, c, target = cache_reader.load_record(sources.experiment, record)
        assert z.shape == (3, 4, 1, 2, 2) and u.shape == c.shape == (3, 3, 1, 2, 2)
        assert cache_reader.load_scores(sources, record).tolist() == pytest.approx(
            [0.97, 0.03, 0.42]
        )


def test_reduction_bundle_retains_failures_units_and_every_registered_figure(
    saved_cache,
):
    bundle = _reduce(saved_cache)
    manifest = validate_bundle(bundle)
    assert find_analysis_bundle(saved_cache["root"], **CONFIG) == bundle
    initial, endpoint = (
        pd.read_parquet(bundle / "initial_metrics.parquet"),
        pd.read_parquet(bundle / "endpoint_metrics.parquet"),
    )
    expected = len(saved_cache["selection"].included_indices) * 3
    assert len(initial) == len(endpoint) == expected
    assert set(initial.seed) == {0, 1, 2} and initial["sscd"].min() < 0.05
    assert initial.step_index.eq(0).all() and endpoint.step_index.eq(2).all()
    assert np.allclose(endpoint.terminal_l2, 2 * endpoint.terminal_rmse)
    assert endpoint.paper_terminal_bound_l2.isna().all()
    assert endpoint.population_forward_loss.isna().all()
    assert endpoint.effective_target_residual_evidence.eq("exact_cache_algebra").all()
    assert set(manifest["statement_ids"]) == set(STATEMENT_IDS)
    assert set(manifest["default_figure_ids"]) == DEFAULT_FIGURES
    assert {path.stem for path in (bundle / "plotdata").glob("*.parquet")} == set(
        FIGURE_ORDER
    )
    summary = read_json(bundle / "summary.json")
    assert summary["effective_initial_unconditional_seed_count"] == 3
    assert summary["experiment_sample_count"] == expected
    assert read_json(bundle / "support_metadata.json")["support_size"] == 4
    assert read_json(bundle / "center_metadata.json")["source_sample_count"] == 3
    # The fixed center's distance uses one center throughout the trajectory.
    trajectories = pd.concat(
        [pd.read_parquet(p) for p in (bundle / "trajectory_metrics").glob("*.parquet")]
    )
    assert trajectories.groupby("original_index").size().eq(9).all()
    assert trajectories.seed_role.eq("experiment").all()
    assert trajectories.is_final_update.eq(trajectories.step_index.eq(2)).all()
    # All target/seed identities remain in both outcome groups; no favorable
    # condition or terminal SSCD is allowed to change scientific eligibility.
    assert trajectories.groupby(["original_index", "seed"]).size().eq(3).all()
    assert set(trajectories.original_index) == set(
        saved_cache["selection"].included_indices
    )
    assert set(trajectories.seed) == {0, 1, 2}
    assert trajectories.groupby(
        "seed"
    ).terminal_sscd.first().to_dict() == pytest.approx({0: 0.97, 1: 0.03, 2: 0.42})
    eligible = trajectories.loc[trajectories.feedback_eligible]
    assert len(eligible) == expected * 2
    assert eligible.candidate_variation_l2.ge(0).all()
    assert np.allclose(
        eligible.candidate_condition_margin_l2,
        eligible.branch_gap_l2
        - eligible.candidate_conditional_reference_error_l2
        - eligible.candidate_unconditional_reference_error_l2
        - eligible.candidate_variation_l2,
    )
    assert np.allclose(
        eligible.candidate_condition_margin_rmse * 2,
        eligible.candidate_condition_margin_l2,
    )
    assert np.allclose(
        eligible.candidate_log_probability_gain,
        eligible.candidate_guided_log_probability
        - eligible.candidate_matched_log_probability,
    )
    assert not trajectories.loc[trajectories.is_final_update, "feedback_eligible"].any()
    # The current-level candidate reference remains available for terminal
    # terms even though the final feedback comparison is inapplicable.
    assert endpoint.candidate_unconditional_reference_error_l2.notna().all()
    baseline = pd.read_parquet(bundle / "plotdata" / "unconditional_center.parquet")
    assert len(baseline) == 3 and not baseline.duplicated(["run_id", "seed"]).any()
    assert baseline.step_index.eq(0).all()
    for table in (
        "feedback_by_step",
        "feedback_by_prompt",
        "synchronization",
        "paired_synchronization",
        "phases",
    ):
        assert (bundle / "summaries" / f"{table}.parquet").is_file()
    assert (bundle / "applicability_report.json").is_file()


def test_copied_bundle_plots_without_raw_tensors_and_preserves_scalar_hashes(
    saved_cache, monkeypatch, tmp_path
):
    bundle = _reduce(saved_cache)
    first = render_bundle(bundle)
    copy_bundle = tmp_path / "scalar-copy"
    shutil.copytree(
        bundle, copy_bundle, ignore=shutil.ignore_patterns("*.pt", "figures")
    )
    before = {
        p.relative_to(copy_bundle).as_posix(): file_sha256(p)
        for p in copy_bundle.rglob("*")
        if p.is_file()
    }
    for module in (torch,):
        monkeypatch.setattr(module, "load", _forbidden)
    monkeypatch.setattr(cache_reader, "load_record", _forbidden)
    monkeypatch.setattr(cache_reader, "load_schedule", _forbidden)
    monkeypatch.setattr(reducer, "run_theory", _forbidden)
    from utils.experiments.theory import centers, feedback, support
    from utils.common import io

    monkeypatch.setattr(io, "safe_torch_load", _forbidden)
    monkeypatch.setattr(centers, "choose_center", _forbidden)
    monkeypatch.setattr(support.FiniteSupport, "evaluate", _forbidden)
    monkeypatch.setattr(feedback, "proposition5_feedback", _forbidden)
    second = render_bundle(copy_bundle)
    assert set(first["figures"]) == set(second["figures"]) == DEFAULT_FIGURES
    for name in DEFAULT_FIGURES:
        assert file_sha256(bundle / "figures" / f"{name}.png") == file_sha256(
            copy_bundle / "figures" / f"{name}.png"
        )
        assert (copy_bundle / "figures" / f"{name}.pdf").is_file()
    assert before == {name: file_sha256(copy_bundle / name) for name in before}


def test_atomic_record_failure_then_resume_existing_shards(saved_cache, monkeypatch):
    original = reducer._record_metrics
    selected = sorted(saved_cache["selection"].included_indices)
    failed_index = selected[-1]
    calls = []

    def interrupted(sources, record, *args, **kwargs):
        calls.append(record.original_index)
        if record.original_index == failed_index:
            raise RuntimeError("synthetic interruption after one published shard")
        return original(sources, record, *args, **kwargs)

    monkeypatch.setattr(reducer, "_record_metrics", interrupted)
    with pytest.raises(TheoryError, match="no complete manifest"):
        _reduce(saved_cache)
    parent = analysis_parent(saved_cache["root"], **CONFIG)
    bundles = [p for p in parent.iterdir() if p.is_dir()]
    assert len(bundles) == 1
    bundle = bundles[0]
    assert not (bundle / "manifest.json").exists()
    assert failed_index in (bundle / "failed.csv").read_text()
    first_hashes = {
        p.name: file_sha256(p)
        for p in (bundle / "trajectory_metrics").glob("*.parquet")
    }
    calls.clear()

    def resumed(sources, record, *args, **kwargs):
        calls.append(record.original_index)
        return original(sources, record, *args, **kwargs)

    monkeypatch.setattr(reducer, "_record_metrics", resumed)
    assert _reduce(saved_cache) == bundle
    assert calls == [failed_index]
    assert all(
        file_sha256(bundle / "trajectory_metrics" / name) == digest
        for name, digest in first_hashes.items()
    )
    assert (
        failed_index in (bundle / "failed.csv").read_text()
    )  # failure history survives success
    validate_bundle(bundle)


@pytest.mark.parametrize(
    "mutation", ["score_seed", "target_identity", "missing_score", "missing_seed"]
)
def test_real_metadata_and_same_seed_join_fail_closed(saved_cache, mutation):
    paths = saved_cache["paths"]["experiment"]
    index = sorted(saved_cache["selection"].included_indices)[0]
    if mutation == "target_identity":
        marker = read_json(paths.record_path(index))
        marker["target_image_sha256"] = "f" * 64
        atomic_write_json(paths.record_path(index), marker)
        with pytest.raises(TheoryError, match="target_image_sha256"):
            cache_reader.discover_sources(saved_cache["root"], **CONFIG)
    elif mutation == "missing_seed":
        path = saved_cache["proximity"] / "proximity.csv"
        frame = pd.read_csv(path, dtype={"original_index": str})
        frame = frame[~(frame.original_index.eq(index) & frame.seed.eq(1))]
        frame.to_csv(path, index=False)
        with pytest.raises(TheoryError, match="seed coverage"):
            cache_reader.discover_sources(saved_cache["root"], **CONFIG)
    else:
        sources = cache_reader.discover_sources(saved_cache["root"], **CONFIG)
        record = next(r for r in sources.selected if r.original_index == index)
        if mutation == "missing_score":
            SSCDPaths(paths.run_directory).score_path(index).unlink()
            with pytest.raises(Exception, match="SSCD score is missing"):
                cache_reader.load_scores(sources, record)
        else:
            sources.proximity.loc[
                sources.proximity.original_index.eq(index), "sscd"
            ] = [0.03, 0.97, 0.42]
            with pytest.raises(TheoryError, match="Same-seed SSCD/proximity mismatch"):
                cache_reader.load_scores(sources, record)


def test_reference_block_mismatch_and_stale_frozen_selection_fail(saved_cache):
    ref = saved_cache["paths"]["reference"]
    config = read_json(ref.run_config)
    config["scientific_config"]["seeds"] = [4, 5, 6]
    config["scientific_config_hash"] = canonical_hash(config["scientific_config"])
    atomic_write_json(ref.run_config, config)
    with pytest.raises(TheoryError, match="reference seed block"):
        cache_reader.discover_sources(saved_cache["root"], **CONFIG)


def test_scientific_hash_changes_with_center_and_source_tensor_staleness_is_rejected(
    saved_cache,
):
    first = _reduce(saved_cache)
    second = _reduce(saved_cache, center="zero")
    assert first != second
    record = list_completed_records(saved_cache["paths"]["experiment"])[-1]
    path = saved_cache["paths"]["experiment"].latent_path(record.original_index)
    with path.open("ab") as handle:
        handle.write(b"changed-source-evidence")
    with pytest.raises(TheoryError, match="Stale source"):
        _reduce(saved_cache)


def test_plot_rejects_scalar_tampering_and_root_rejects_metadata_inventory_change(
    saved_cache,
):
    bundle = _reduce(saved_cache)
    path = bundle / "plotdata" / "initial_recovery.parquet"
    frame = pd.read_parquet(path)
    frame.iloc[1:].to_parquet(path, index=False)
    with pytest.raises(TheoryError, match="Changed/incompatible numerical artifact"):
        validate_bundle(bundle)
    ref = saved_cache["paths"]["reference"]
    atomic_write_json(ref.record_directory / "extra.json", {"new": "completion"})
    with pytest.raises(TheoryError, match="inventory changed"):
        validate_source_metadata(bundle, saved_cache["root"])


def test_manuscript_exact_inventory_and_guarded_population_formulas():
    registry = load_statement_registry()
    assert [entry["semantic_id"] for entry in registry["statements"]] == list(
        STATEMENT_IDS
    )
    assert [entry["label"] for entry in registry["statements"]] == [
        "Theorem 1",
        "Lemma 2",
        "Corollary 3",
        "Lemma 4",
        "Proposition 5",
        "Lemma 6",
        "Theorem 7",
    ]
    incomplete = copy.deepcopy(registry)
    incomplete["statements"].pop(3)
    with pytest.raises(ValueError, match="seven revised results"):
        validate_statement_registry(incomplete)
    unavailable = paper_unavailable_metrics(
        7.5, branch_gap_l2=4, conditional_target_error_l2=1
    )
    assert (
        unavailable["paper_feedback_lhs_l2"] == 4
        and unavailable["paper_feedback_rhs_l2"] is None
    )
    assert unavailable["paper_conditional_reference_error_l2"] is None
    identified = dict(
        identified_training_distribution=True, single_target_verified=True
    )
    assert synchronization_bound(1, 2, 10, 0.9, **identified)[
        "bound_l2"
    ] == pytest.approx(4)
    assert (
        terminal_reproduction_bound(
            1, 2, 10, 0.9, guidance=7.5, terminal_clean_update=True
        )["bound_l2"]
        is None
    )
    bound = terminal_reproduction_bound(
        1, 2, 10, 0.9, guidance=7.5, terminal_clean_update=True, **identified
    )
    assert bound["bound_l2"] == pytest.approx(27)
    for g, clean in [(-2, True), (7.5, False)]:
        assert (
            terminal_reproduction_bound(
                1, 2, 10, 0.9, guidance=g, terminal_clean_update=clean, **identified
            )["bound_l2"]
            is None
        )
    condition = posterior_feedback_condition(
        4, 1, 2, 0.5, guidance=7.5, kappa=0.1, destination_sigma=0.2, **identified
    )
    assert condition["slack_l2"] == 0.5 and condition["strictly_satisfied"]
    assert (
        posterior_feedback_condition(
            4, 1, 2, 0.5, guidance=7.5, kappa=0.1, destination_sigma=0, **identified
        )["slack_l2"]
        is None
    )
