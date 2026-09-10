"""Offline checks for the standalone, reference-fixed ensemble experiment."""

import json

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from PIL import Image

from scripts import forward_corruptions_generated_states as experiment
from utils.common.io import file_sha256
from utils.experiments.cache import CompletedGenerationRecord, GenerationPaths
from utils.experiments.sscd import SSCDPaths
from utils.models.sampling import make_initial_noise


def _states(target, *, offset=0.0):
    noise = make_initial_noise(range(20), target.shape)
    states = torch.stack(
        (noise, target + noise * 0.3, target + noise * 0.01 + offset), dim=1
    )
    return states.to(torch.float16)


def _measure(panel="memorized", offset=0.0):
    target = torch.ones(1, 2, 2)
    return experiment.measure_states(
        panel=panel,
        record_id=panel,
        original_index="1" if panel == "memorized" else "2",
        seeds=list(range(20)),
        timesteps=torch.tensor([19, 1]),
        alpha=torch.tensor([0.1, 0.8]),
        sigma=torch.tensor([0.99, 0.6]),
        target=target,
        states=_states(target, offset=offset),
    )


def test_measurements_use_input_states_raw_l2_fixed_noise_and_final_output():
    frame = _measure()
    noise = make_initial_noise(range(20), (1, 2, 2)).double()
    target = torch.ones(1, 2, 2).double()
    states = _states(target.float())
    assert len(frame) == 60
    for step, (alpha, sigma) in enumerate(
        zip(
            torch.tensor([0.1, 0.8, 1.0]).double(),
            torch.tensor([0.99, 0.6, 0.0]).double(),
            strict=True,
        )
    ):
        rows = frame.loc[frame.step_index.eq(step)].sort_values("generation_seed")
        expected_forward = ((alpha - 1) * target + sigma * noise).flatten(1).norm(dim=1)
        expected_generated = (states[:, step].double() - target).flatten(1).norm(dim=1)
        assert rows.forward_distance.to_numpy() == pytest.approx(
            expected_forward.numpy()
        )
        assert rows.generated_distance.to_numpy() == pytest.approx(
            expected_generated.numpy()
        )
    final = frame.loc[frame.is_final_clean_state]
    assert final.timestep.eq(-1).all()
    assert final.forward_distance.eq(0).all()
    assert final.denoising_progress.eq(1).all()
    # Intermediate input distances are not replaced by the final output distances.
    assert (
        not frame.loc[frame.step_index.eq(1), "generated_distance"]
        .reset_index(drop=True)
        .equals(final.generated_distance.reset_index(drop=True))
    )


def test_progress_total_covers_all_seed_timestep_observations():
    updates = []
    target = torch.zeros(1, 2, 2)
    experiment.measure_states(
        panel="memorized",
        record_id="a",
        original_index="1",
        seeds=range(20),
        timesteps=torch.tensor([19, 1]),
        alpha=torch.tensor([0.1, 0.8]),
        sigma=torch.tensor([0.99, 0.6]),
        target=target,
        states=_states(target),
        progress=updates.append,
    )
    assert updates == [20, 20, 20]


@pytest.mark.parametrize(
    "corruption", ["initial", "missing_seed", "missing_final", "nan"]
)
def test_measurement_never_silently_drops_bad_evaluation_samples(corruption):
    target = torch.ones(1, 2, 2)
    states = _states(target)
    if corruption == "initial":
        states[:, 0] += 1
    elif corruption == "missing_seed":
        states = states[:-1]
    elif corruption == "missing_final":
        states = states[:, :-1]
    else:
        states[0, 1, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        experiment.measure_states(
            panel="memorized",
            record_id="a",
            original_index="1",
            seeds=range(20),
            timesteps=torch.tensor([19, 1]),
            alpha=torch.tensor([0.1, 0.8]),
            sigma=torch.tensor([0.99, 0.6]),
            target=target,
            states=states,
        )


def test_reference_ranking_uses_worst_seed_scores_not_categories_or_mean_alone():
    candidates = [
        {"record_id": "reliable", "sscd_min": 0.9, "sscd_mean": 0.91, "sscd_max": 0.92},
        {"record_id": "spiky", "sscd_min": 0.1, "sscd_mean": 0.95, "sscd_max": 0.99},
        {
            "record_id": "normal-TV",
            "sscd_min": -0.1,
            "sscd_mean": -0.01,
            "sscd_max": 0.01,
        },
        {
            "record_id": "normal-N",
            "sscd_min": -0.2,
            "sscd_mean": -0.02,
            "sscd_max": 0.5,
        },
    ]
    chosen = experiment.rank_reference_pairs(
        candidates, dict.fromkeys(experiment.PANELS)
    )
    assert chosen["memorized"]["record_id"] == "reliable"
    assert chosen["normal"]["record_id"] == "normal-TV"
    requested = {"memorized": "spiky", "normal": "normal-N"}
    assert {
        k: v["record_id"]
        for k, v in experiment.rank_reference_pairs(candidates, requested).items()
    } == requested
    with pytest.raises(ValueError, match="lack complete reference"):
        experiment.rank_reference_pairs(
            candidates, {"memorized": "missing", "normal": "normal-N"}
        )


def test_seed_blocks_settings_and_paths(tmp_path):
    parser = experiment.build_parser()
    arguments = parser.parse_args([])
    assert arguments.device is None
    assert parser.parse_args(["--device", "cpu"]).device == "cpu"
    settings = experiment.requested_settings(arguments)
    assert settings["evaluation_seeds"] == list(range(20))
    assert settings["reference_seeds"] == list(range(20, 40))
    logs, output = experiment.experiment_paths(tmp_path, settings)
    assert "logs/sdv1_ddim_g7.5_T50_N20/experiment_S0_N20/" in str(logs)
    assert str(logs).endswith("reference_S20_N20/reference_ranked")
    assert (
        "outputs/sdv1_ddim_g7.5_T50_N20/forward_corruptions_generated_states/"
        in str(output)
    )
    for args in (
        ["--reference-seed-start", "19"],
        ["--memorized-record-id", "only"],
        ["--T", "8"],
    ):
        with pytest.raises(ValueError):
            experiment.requested_settings(parser.parse_args(args))
    for args in (
        ["--plot", "--overwrite"],
        ["--N", "19"],
        ["--model", "sdv2"],
        ["--scheduler", "ddpm"],
        ["--g", "8"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(args)


def test_interpretation_reports_overlap_at_intermediate_steps_not_only_endpoints():
    summary = experiment.summarize_observations(
        pd.concat([_measure(), _measure("normal", 1)])
    )
    summary.loc[summary.step_index.eq(1), ["forward_min", "generated_min"]] = 1.0
    summary.loc[summary.step_index.eq(1), ["forward_max", "generated_max"]] = 3.0
    report = experiment.summarize_interpretation(summary)
    for panel in experiment.PANELS:
        assert report["panels"][panel]["overlapping_step_indices"] == [1]
        assert report["panels"][panel]["intermediate_steps"] == 1
    assert "not confidence intervals" in report["bands"]
    assert "not a formal distributional test" in report["interpretation"]


def _mock_runs(tmp_path, monkeypatch):
    runs = {}
    records = []
    for seed_start in (0, 20):
        paths = GenerationPaths(tmp_path / f"source_{seed_start}")
        paths.create()
        paths.schedule.write_bytes(b"fixed source schedule")
        sscd = SSCDPaths(paths.run_directory)
        sscd.record_directory.mkdir()
        seeds = tuple(range(seed_start, seed_start + 20))
        runs[seed_start] = SimpleNamespace(
            paths=paths,
            sscd_paths=sscd,
            configuration={"seeds": list(seeds)},
            seeds=seeds,
            timesteps=torch.arange(17, -1, -2),
            alpha=torch.linspace(0.1, 0.99, 9),
            sigma=(1 - torch.linspace(0.1, 0.99, 9).square()).sqrt(),
            scheduler_final_alpha_cumprod=0.999,
            sscd_configuration={"seeds": list(seeds)},
        )
        for index, panel in enumerate(experiment.PANELS, start=1):
            marker = {
                "record_id": panel,
                "original_index": str(index),
                "source_row_number": index,
                "prompt_raw": f"prompt {panel}",
                "target_image_path": f"training/{index}.png",
                "target_image_sha256": str(index) * 64,
            }
            sscd.marker_path(str(index)).write_text(json.dumps(marker))
            if seed_start == 20:
                records.append(
                    CompletedGenerationRecord(
                        str(index), index, paths.record_path(str(index)), marker
                    )
                )
    monkeypatch.setattr(
        experiment, "load_run", lambda root, **kw: runs[kw["seed_start"]]
    )
    monkeypatch.setattr(experiment, "validate_run_pair", lambda *_: None)
    monkeypatch.setattr(experiment, "list_completed_records", lambda _: records)
    calls = []

    def scores(run, record):
        calls.append(("score", run.seeds[0], record.metadata["record_id"]))
        # Evaluation scores are deliberately reversed: labels must stay reference-fixed.
        high = (record.metadata["record_id"] == "memorized") == (run.seeds[0] == 20)
        return (
            torch.linspace(0.9, 0.95, 20) if high else torch.linspace(-0.05, 0.05, 20)
        )

    def states(run, record):
        calls.append(("states", run.seeds[0], record.metadata["record_id"]))
        target = torch.ones(1, 2, 2)
        noise = make_initial_noise(run.seeds, target.shape)
        cached = torch.stack(
            [noise] + [target + noise / (step + 1) for step in range(1, 10)],
            dim=1,
        ).half()
        return cached, target, dict(record.metadata)

    monkeypatch.setattr(experiment, "load_scores", scores)
    monkeypatch.setattr(experiment, "load_states", states)
    plotted = []
    monkeypatch.setattr(
        experiment,
        "plot_saved_results",
        lambda csv, output: plotted.append(pd.read_csv(csv)),
    )
    monkeypatch.setattr(
        experiment,
        "ensure_decoded_states",
        lambda root, logs, **kwargs: logs / "decoded_states.pt",
    )
    monkeypatch.setattr(experiment, "plot_saved_decoded_states", lambda *_: None)
    return calls, plotted


def test_end_to_end_logging_reference_freeze_cache_hit_and_plot_only(
    tmp_path, monkeypatch
):
    calls, plotted = _mock_runs(tmp_path, monkeypatch)
    args = experiment.build_parser().parse_args(["--T", "9"])
    logs, output = experiment.run_experiment(tmp_path, args)
    assert calls[:2] == [("score", 20, "memorized"), ("score", 20, "normal")]
    observed = pd.read_csv(logs / "observations.csv")
    final = pd.read_csv(logs / "final_samples.csv")
    assert len(observed) == 400 and len(final) == 40
    assert set(final.generation_seed) == set(range(20))
    assert final.loc[final.panel.eq("memorized"), "target_sscd"].max() < 0.1
    assert final.loc[final.panel.eq("normal"), "target_sscd"].min() > 0.8
    targets = torch.load(logs / "target_latents.pt", weights_only=True)
    assert set(targets) == set(experiment.PANELS)
    assert len(plotted) == 1
    assert set(path.name for path in logs.iterdir()) == {
        *experiment.ARTIFACTS,
        "completion.json",
    }
    fingerprints = {path: path.read_bytes() for path in logs.iterdir()}

    def forbidden(*args, **kwargs):
        raise AssertionError("completed or plot-only run must not load source caches")

    monkeypatch.setattr(experiment, "load_run", forbidden)
    assert experiment.run_experiment(tmp_path, args) == (logs, output)
    args.plot = True
    assert experiment.run_experiment(tmp_path, args) == (logs, output)
    assert len(plotted) == 3
    assert all(path.read_bytes() == content for path, content in fingerprints.items())
    (logs / "observations.csv").write_text("changed")
    with pytest.raises(ValueError, match="missing or changed"):
        experiment.run_experiment(tmp_path, args)


def test_unavailable_chosen_evaluation_aborts_without_reselecting(
    tmp_path, monkeypatch
):
    calls, _ = _mock_runs(tmp_path, monkeypatch)

    def unavailable(*args, **kwargs):
        raise ValueError("missing chosen evaluation trajectory")

    monkeypatch.setattr(experiment, "load_states", unavailable)
    args = experiment.build_parser().parse_args(["--T", "9"])
    with pytest.raises(ValueError, match="missing chosen"):
        experiment.run_experiment(tmp_path, args)
    assert calls == [("score", 20, "memorized"), ("score", 20, "normal")]
    logs, _ = experiment.experiment_paths(tmp_path, experiment.requested_settings(args))
    assert not (logs / "completion.json").exists()


def test_gallery_step_indices_span_initialization_to_final_output():
    assert experiment.gallery_step_indices(50) == [0, 6, 11, 17, 22, 28, 33, 39, 44, 50]
    assert experiment.gallery_step_indices(9) == list(range(10))
    for steps in (10, 20, 100):
        indices = experiment.gallery_step_indices(steps)
        assert len(indices) == len(set(indices)) == 10
        assert indices == sorted(indices)
        assert indices[0] == 0 and indices[-1] == steps


@pytest.mark.parametrize("steps", [0, 1, 8, -1, True, 50.0])
def test_gallery_step_indices_require_ten_distinct_states(steps):
    with pytest.raises(ValueError, match="T >= 9"):
        experiment.gallery_step_indices(steps)


def _gallery_inputs():
    target = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    seed = 7
    noise = make_initial_noise([seed], target.shape)[0]
    states = torch.stack(
        [noise] + [target * (step / 50) + noise * 0.3 for step in range(1, 51)]
    ).half()
    alpha = torch.linspace(0.1, 0.99, 50)
    sigma = (1 - alpha.square()).sqrt()
    return target, seed, noise, states, alpha, sigma


def test_gallery_latents_are_matched_input_states_not_clean_predictions():
    target, seed, noise, states, alpha, sigma = _gallery_inputs()
    indices = experiment.gallery_step_indices(50)
    original_states = states.clone()
    forward, generated = experiment.gallery_latents(
        target, states, seed=seed, step_indices=indices, alpha=alpha, sigma=sigma
    )
    expected_forward = torch.stack(
        [
            alpha[step].double() * target.double()
            + sigma[step].double() * noise.double()
            for step in indices[:-1]
        ]
        + [target.double()]
    ).float()
    torch.testing.assert_close(forward, expected_forward, rtol=0, atol=0)
    torch.testing.assert_close(generated, states[indices].float(), rtol=0, atol=0)
    assert forward.dtype == generated.dtype == torch.float32
    assert forward.shape == generated.shape == (10, 1, 2, 2)
    for row, step in enumerate(indices[:-1]):
        recovered_noise = (forward[row] - alpha[step] * target) / sigma[step]
        torch.testing.assert_close(recovered_noise, noise, rtol=1e-6, atol=1e-6)
    assert torch.equal(forward[-1], target)
    assert not torch.equal(generated[-1], target)
    assert not torch.equal(forward[0], generated[0])
    assert torch.equal(states, original_states)


@pytest.mark.parametrize("corruption", ["initial", "missing_final", "nan"])
def test_gallery_latents_reject_misaligned_or_corrupt_cached_inputs(corruption):
    target, seed, _, states, alpha, sigma = _gallery_inputs()
    if corruption == "initial":
        states[0] += 1
    elif corruption == "missing_final":
        states = states[:-1]
    else:
        states[1, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        experiment.gallery_latents(
            target,
            states,
            seed=seed,
            step_indices=experiment.gallery_step_indices(50),
            alpha=alpha,
            sigma=sigma,
        )


def _decoded_cache(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    completion = logs / "completion.json"
    completion.write_text(json.dumps({"artifacts": {}}))
    payload = logs / "decoded_states.pt"
    torch.save({"minimal_payload": torch.zeros(1)}, payload)
    marker = logs / "decoded_states.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "measurement_completion_sha256": file_sha256(completion),
                "payload_sha256": file_sha256(payload),
            }
        )
    )
    return logs, payload, marker


def _forbid_gallery_inference(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("gallery cache hit must not access source caches or VAE")

    for name in ("load_run", "load_states", "load_vae_from_generation_config"):
        monkeypatch.setattr(experiment, name, forbidden)


def test_decoded_gallery_creation_uses_forty_cached_inputs_then_reuses_rgb(
    tmp_path, monkeypatch
):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "completion.json").write_text(json.dumps({"artifacts": {}}))
    paths = GenerationPaths(tmp_path / "source")
    paths.create()
    paths.schedule.write_bytes(b"unchanged source schedule")
    target, seed, _, states, alpha, sigma = _gallery_inputs()
    seeds = tuple(range(seed, seed + 20))
    scientific = {
        "model_id": "fixed-model",
        "model_revision": "model-commit",
        "vae_id": "fixed-model",
        "vae_revision": "vae-commit",
    }
    evaluation = SimpleNamespace(
        paths=paths,
        configuration={"scientific_config": scientific, "seeds": list(seeds)},
        seeds=seeds,
        timesteps=torch.arange(99, -1, -2),
        alpha=alpha,
        sigma=sigma,
    )
    targets = {
        panel: target * multiplier
        for panel, multiplier in zip(experiment.PANELS, (1.0, -2.0), strict=True)
    }
    cached_states = {
        panel: states.unsqueeze(0).expand(20, -1, -1, -1, -1).clone()
        for panel in experiment.PANELS
    }
    markers = {
        panel: {"tensor_file_sha256": {"latent": str(index) * 64}}
        for index, panel in enumerate(experiment.PANELS, start=1)
    }
    configuration = {
        "settings": {"num_inference_steps": 50, "evaluation_seeds": list(seeds)},
        "evaluation_generation_configuration": evaluation.configuration,
        "evaluation_schedule_sha256": file_sha256(paths.schedule),
        "pairs": {
            panel: {
                "record_id": panel,
                "original_index": str(index),
                "source_row_number": index,
                "evaluation_generation_marker": markers[panel],
            }
            for index, panel in enumerate(experiment.PANELS, start=1)
        },
    }
    (logs / "run_config.json").write_text(json.dumps(configuration))
    torch.save(targets, logs / "target_latents.pt")
    original = {path: path.read_bytes() for path in [*logs.iterdir(), paths.schedule]}
    source_calls = []

    def load_run(root, **kwargs):
        source_calls.append(("run", root, kwargs))
        return evaluation

    def load_states(run, record):
        panel = experiment.PANELS[int(record.original_index) - 1]
        source_calls.append(("states", panel))
        assert run is evaluation
        return cached_states[panel], targets[panel], markers[panel]

    monkeypatch.setattr(experiment, "load_run", load_run)
    monkeypatch.setattr(experiment, "load_states", load_states)
    runtime = SimpleNamespace(device="cpu")
    monkeypatch.setattr(
        experiment,
        "select_runtime",
        lambda device, **kwargs: runtime if device == "cpu" else None,
    )
    vae = object()
    load_calls = []

    def load_vae(configuration, **kwargs):
        load_calls.append((configuration, kwargs))
        return SimpleNamespace(vae=vae, device="cpu")

    monkeypatch.setattr(experiment, "load_vae_from_generation_config", load_vae)
    decoded_batches = []

    def decode(batch, loaded_vae, device, *, microbatch_size, progress_callback):
        assert loaded_vae is vae and device == "cpu" and microbatch_size == 4
        decoded_batches.append(batch.clone())
        progress_callback(len(batch))
        return [
            Image.new("RGB", (16, 16), (index, index * 2, index * 3))
            for index in range(len(batch))
        ]

    monkeypatch.setattr(experiment, "decode_generated_latents", decode)
    payload_path = experiment.ensure_decoded_states(tmp_path, logs, device="cpu")
    assert payload_path == logs / "decoded_states.pt"
    assert source_calls == [
        ("run", tmp_path, {"num_inference_steps": 50, "seed_start": seed}),
        ("states", "memorized"),
        ("states", "normal"),
    ]
    assert load_calls == [(evaluation.configuration, {"runtime": runtime})]
    assert len(decoded_batches) == 2
    indices = experiment.gallery_step_indices(50)
    for panel, batch in zip(experiment.PANELS, decoded_batches, strict=True):
        forward, generated = experiment.gallery_latents(
            targets[panel],
            cached_states[panel][0],
            seed=seed,
            step_indices=indices,
            alpha=alpha,
            sigma=sigma,
        )
        torch.testing.assert_close(
            batch, torch.cat((forward, generated)), rtol=0, atol=0
        )
    payload = torch.load(payload_path, weights_only=True)
    assert payload["generation_seed"] == seed
    assert payload["step_indices"] == indices
    assert payload["display_timesteps"] == [50 - step for step in indices]
    assert payload["scheduler_timesteps"][-1] == -1
    assert payload["alpha_t"][-1] == 1.0 and payload["sigma_t"][-1] == 0.0
    assert payload["vae"] == scientific
    for panel in experiment.PANELS:
        for name, start in (("forward_images", 0), ("generated_images", 10)):
            images = payload["panels"][panel][name]
            assert images.dtype == torch.uint8
            assert images.shape == (10, 256, 256, 3)
            for index, image in enumerate(images, start=start):
                assert torch.equal(
                    image[0, 0], torch.tensor([index, index * 2, index * 3])
                )
    marker = json.loads((logs / "decoded_states.json").read_text())
    assert marker == {
        "schema_version": 1,
        "measurement_completion_sha256": file_sha256(logs / "completion.json"),
        "payload_sha256": file_sha256(payload_path),
    }
    assert all(path.read_bytes() == content for path, content in original.items())
    _forbid_gallery_inference(monkeypatch)
    assert (
        experiment.ensure_decoded_states(tmp_path, logs, plot_only=True) == payload_path
    )
    assert len(decoded_batches) == 2


def test_decoded_gallery_cache_hit_is_self_contained(tmp_path, monkeypatch):
    logs, payload, _ = _decoded_cache(tmp_path)
    original = {path: path.read_bytes() for path in logs.iterdir()}
    _forbid_gallery_inference(monkeypatch)
    for plot_only in (False, True):
        assert (
            experiment.ensure_decoded_states(tmp_path, logs, plot_only=plot_only)
            == payload
        )
    assert all(path.read_bytes() == content for path, content in original.items())


@pytest.mark.parametrize("corruption", ["payload", "completion", "schema"])
def test_decoded_gallery_cache_rejects_changed_artifacts(
    tmp_path, monkeypatch, corruption
):
    logs, payload, marker = _decoded_cache(tmp_path)
    _forbid_gallery_inference(monkeypatch)
    if corruption == "payload":
        payload.write_bytes(b"corrupted gallery payload")
    elif corruption == "completion":
        (logs / "completion.json").write_text(
            json.dumps({"artifacts": {}, "new": True})
        )
    else:
        metadata = json.loads(marker.read_text())
        metadata["schema_version"] = 2
        marker.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="rerun with --overwrite"):
        experiment.ensure_decoded_states(tmp_path, logs, plot_only=True)


@pytest.mark.parametrize("missing", ["decoded_states.pt", "decoded_states.json"])
def test_plot_only_missing_gallery_cache_requests_decode_once(
    tmp_path, monkeypatch, missing
):
    logs, _, _ = _decoded_cache(tmp_path)
    (logs / missing).unlink()
    _forbid_gallery_inference(monkeypatch)
    with pytest.raises(ValueError, match="run without --plot"):
        experiment.ensure_decoded_states(tmp_path, logs, plot_only=True)


@pytest.mark.parametrize("mode", [[], ["--plot"], ["--overwrite"]])
def test_plot_experiment_always_renders_saved_measurements_then_gallery(
    tmp_path, monkeypatch, mode
):
    arguments = experiment.build_parser().parse_args([*mode, "--device", "cpu"])
    logs = tmp_path / "logs"
    output = tmp_path / "output"
    decoded = logs / "decoded_states.pt"
    calls = []
    monkeypatch.setattr(
        experiment,
        "plot_saved_results",
        lambda csv, directory: calls.append(("measurements", csv, directory)),
    )

    def ensure(root, directory, **kwargs):
        calls.append(("ensure_gallery", root, directory, kwargs))
        return decoded

    monkeypatch.setattr(experiment, "ensure_decoded_states", ensure)
    monkeypatch.setattr(
        experiment,
        "plot_saved_decoded_states",
        lambda path, directory: calls.append(("gallery", path, directory)),
    )
    experiment._plot_experiment(tmp_path, logs, output, arguments)
    assert calls == [
        ("measurements", logs / "observations.csv", output),
        (
            "ensure_gallery",
            tmp_path,
            logs,
            {
                "plot_only": arguments.plot,
                "overwrite": arguments.overwrite,
                "device": "cpu",
            },
        ),
        ("gallery", decoded, output),
    ]
