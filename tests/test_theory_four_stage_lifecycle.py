"""Unexecuted four-stage lifecycle and optional-inference isolation regressions."""
from pathlib import Path

import pytest

from utils.common.io import atomic_write_json, read_json
from utils.experiments.theory.contracts import FOUR_STAGE_OPTION_KEYS, TheoryError, numerical_config
from utils.experiments.theory.paper_contracts import (
    recompute_command, saved_plot_configuration, saved_scientific_configuration,
)


def config(**changes):
    return numerical_config(model_name="sdv1", scheduler_name="ddim", **({"mean_source": "cached-targets"} | changes))


def test_optional_network_settings_do_not_change_base_scientific_configuration():
    default = config()
    enabled = config(counterfactual_unconditional=True, counterfactual_steps="3,0,3")
    assert not set(FOUR_STAGE_OPTION_KEYS).intersection(default)
    assert enabled["counterfactual_steps"] == [0, 3]
    assert {key: value for key, value in enabled.items() if key not in FOUR_STAGE_OPTION_KEYS} == default
    assert "--counterfactual-unconditional --counterfactual-steps 0,3" in recompute_command(enabled)


@pytest.mark.parametrize("options", [
    {"counterfactual_unconditional": 1},
    {"counterfactual_steps": [0]},
    {"counterfactual_unconditional": True, "counterfactual_steps": []},
    {"counterfactual_unconditional": True, "counterfactual_steps": [True]},
    {"counterfactual_unconditional": True, "counterfactual_steps": [-1]},
    {"counterfactual_unconditional": True, "counterfactual_steps": [49]},
    {"counterfactual_unconditional": True, "counterfactual_steps": "0,1.5"},
])
def test_counterfactual_configuration_rejects_undefined_or_implicit_probes(options):
    with pytest.raises(TheoryError):
        config(**options)


def test_portable_plot_inherits_separate_saved_supplement_and_nondefault_loss(tmp_path):
    base = config(num_loss_seeds=93, loss_seed=17)
    enabled = config(num_loss_seeds=93, loss_seed=17,
                     counterfactual_unconditional=True, counterfactual_steps=[2])
    saved = {"scientific_config": base,
             "supplemental_config": {key: enabled[key] for key in FOUR_STAGE_OPTION_KEYS}}
    atomic_write_json(tmp_path / "run_config.json", saved)
    assert saved_scientific_configuration(saved) == enabled
    assert saved_plot_configuration(tmp_path, requested=config(), portable=True) == enabled
    with pytest.raises(TheoryError, match="conflict.*num_loss_seeds.*Recompute"):
        saved_plot_configuration(tmp_path, requested=config(), explicit_keys={"num_loss_seeds"})
    assert read_json(tmp_path / "run_config.json")["scientific_config"] == base


def test_root_forwards_opt_in_to_theory_without_upstream_work(tmp_path):
    from tests.test_theory_plot_cli import _option, _run_all_stub
    result, calls = _run_all_stub(tmp_path, ["--model", "sdv1", "--scheduler", "ddim",
        "--recompute-experiments", "--counterfactual-unconditional", "--counterfactual-steps", "0,4"])
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1 and calls[0][0] == "theory_validation.sh"
    assert "--counterfactual-unconditional" in calls[0]
    assert _option(calls[0], "--counterfactual-steps") == "0,4"


def test_primary_preparation_precedes_integration_and_optional_stage_is_separate(tmp_path, monkeypatch):
    from utils.experiments.theory import (four_stage_reduce, counterfactual_probes,
                                         numerical_reduce, gaussian_controls, paper_reduce, paper_plotting)
    from tests.test_theory_four_stage_figures import four_stage_config
    from tests.test_theory_paper_pipeline import measured_fixture
    from utils.experiments.theory import reduce as shared_reduce
    monkeypatch.setattr(shared_reduce, "_resolve_theory_devices", lambda _device: ("cuda:0",))
    sequence = []
    measurements = measured_fixture(tmp_path)
    settings = four_stage_config() | {"counterfactual_unconditional": True, "counterfactual_steps": [0]}
    base = {key: value for key, value in settings.items() if key not in FOUR_STAGE_OPTION_KEYS}
    def prepare(root, **options):
        sequence.append("fast-primary")
        assert options["config"] == base
    def numerics(root, **options):
        sequence.append("required-integration-and-numerics")
        assert options["config"] == base
        return measurements
    def supplement(root, *, result, **options):
        sequence.append("four-stage-collection")
        assert options["allow_compute"]
        return result
    def optional(root, *, result, **options):
        sequence.append("optional-network")
        assert options["config"]["counterfactual_unconditional"]
        return result  # Separate backend tests exercise inference and receipts.
    monkeypatch.setattr(gaussian_controls, "run_gaussian_control_analysis", lambda *a, result, **k: result)
    monkeypatch.setattr(four_stage_reduce, "prepare_four_stage_primary", prepare)
    monkeypatch.setattr(numerical_reduce, "run_precision_analysis", numerics)
    monkeypatch.setattr(four_stage_reduce, "run_four_stage_analysis", supplement)
    monkeypatch.setattr(counterfactual_probes, "run_counterfactual_analysis", optional)
    def publish(stage, **options):
        from utils.experiments.figure_paths import publication_directory
        from utils.experiments.theory.paper_contracts import PaperPaths
        assert options["formats"] == ("pdf",)
        assert not options["figure_directory"].is_relative_to(stage)
        assert options["publication_directory"] == publication_directory(PaperPaths.build(tmp_path, **settings).output_directory)
        atomic_write_json(stage / "figure_manifest.json", {"files": {}, "figures": [], "complete": True})
    monkeypatch.setattr(paper_plotting, "render_paper", publish)
    output = paper_reduce.run_paper(tmp_path, **settings)
    assert sequence == ["fast-primary", "required-integration-and-numerics", "four-stage-collection", "optional-network"]
    saved = read_json(output / "run_config.json")
    assert saved["scientific_config"] == base
    assert saved["supplemental_config"]["counterfactual_steps"] == [0]
    assert saved["source_analysis"]["path"] == str(measurements["directory"])
    assert (output / "initial_baseline_summary.csv").is_file()
    assert (output / "initial_baseline_summary.json").is_file()


def test_cli_derived_recompute_inherits_saved_draw_count_without_changing_explicit_root(tmp_path, monkeypatch):
    from scripts import theory_validation as cli
    from utils.experiments.theory import paper_reduce
    from utils.experiments.theory.paper_contracts import PaperPaths
    saved = config(num_loss_seeds=93, loss_seed=17)
    bundle = PaperPaths.build(tmp_path, **saved).output_directory
    atomic_write_json(bundle / "run_config.json", {"scientific_config": saved})
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    observed = []
    def measure(root, **options):
        observed.append(options)
        return bundle
    monkeypatch.setattr(paper_reduce, "run_paper", measure)
    assert cli.main(["--model", "sdv1", "--scheduler", "ddim", "--recompute-experiments", "--loss-seed", "23"]) == 0
    assert observed[0]["num_loss_seeds"] == 93
    assert observed[0]["loss_seed"] == 23


def test_explicit_center_override_drops_inherited_independent_baseline(tmp_path, monkeypatch):
    from scripts import theory_validation as cli
    from utils.experiments.theory import paper_reduce
    from utils.experiments.theory.paper_contracts import PaperPaths
    saved = config(center="cached-baseline", cached_baseline=tmp_path / "saved.pt")
    bundle = PaperPaths.build(tmp_path, **saved).output_directory
    atomic_write_json(bundle / "run_config.json", {"scientific_config": saved})
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    observed = []
    def measure(root, **options):
        observed.append(options)
        return bundle
    monkeypatch.setattr(paper_reduce, "run_paper", measure)
    assert cli.main(["--model", "sdv1", "--scheduler", "ddim",
                     "--recompute-experiments", "--center", "reference-initial"]) == 0
    assert observed[0]["center"] == "reference-initial"
    assert "cached_baseline" not in observed[0]



@pytest.mark.parametrize("options", [("--center", "zero"), ("--no-mu",)])
def test_cli_rejects_zero_center_before_measurement(options, monkeypatch):
    from scripts import theory_validation as cli
    from utils.experiments.theory import paper_reduce
    def forbidden(*args, **kwargs):
        pytest.fail("retired zero centering reached measurement")
    monkeypatch.setattr(paper_reduce, "run_paper", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main(list(options))
    assert error.value.code == 2


def test_new_analysis_upgrades_legacy_zero_center_without_changing_saved_receipt(tmp_path, monkeypatch):
    from scripts import theory_validation as cli
    from utils.experiments.theory import paper_reduce
    from utils.experiments.theory.paper_contracts import PaperPaths
    saved = config(center="zero", num_loss_seeds=93, mean_source="reference-min-snr")
    bundle = PaperPaths.build(tmp_path, **saved).output_directory
    receipt = {"scientific_config": saved}
    atomic_write_json(bundle / "run_config.json", receipt)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    observed = []
    def measure(root, **options):
        observed.append(options)
        return bundle
    monkeypatch.setattr(paper_reduce, "run_paper", measure)
    assert cli.main(["--model", "sdv1", "--scheduler", "ddim", "--recompute-experiments"]) == 0
    assert observed[0]["center"] == "reference-initial"
    assert observed[0]["mean_source"] == "reference-min-snr"
    assert observed[0]["num_loss_seeds"] == 93
    assert read_json(bundle / "run_config.json") == receipt


def test_historical_zero_center_remains_readable_and_retry_uses_reference_center(tmp_path):
    import shlex
    saved = config(center="zero", mean_source="reference-min-snr")
    receipt = {"scientific_config": saved}
    atomic_write_json(tmp_path / "run_config.json", receipt)
    assert saved_plot_configuration(tmp_path, requested=config(), portable=True) == saved
    tokens = shlex.split(recompute_command(saved))
    assert tokens[tokens.index("--center") + 1] == "reference-initial"
    assert tokens[tokens.index("--mean-source") + 1] == "reference-min-snr"
    assert read_json(tmp_path / "run_config.json") == receipt


def test_direct_paper_compute_rejects_legacy_zero_center_before_cuda_or_publication(tmp_path, monkeypatch):
    from utils.experiments.theory import paper_reduce, reduce as shared_reduce
    def forbidden(*args, **kwargs):
        pytest.fail("retired zero centering reached CUDA setup")
    monkeypatch.setattr(shared_reduce, "_resolve_theory_devices", forbidden)
    with pytest.raises(TheoryError, match="Zero centering is retired"):
        paper_reduce.run_paper(tmp_path, **config(center="zero"))
    assert not list(tmp_path.iterdir())


def test_shell_refine_center_can_inherit_saved_baseline_path(tmp_path):
    from tests.test_theory_plot_cli import _run_all_stub
    result, calls = _run_all_stub(tmp_path, ["--model", "sdv1", "--scheduler", "ddim",
                                            "--refine-numerics", "--center", "cached-baseline"])
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    assert "--refine-numerics" in calls[0]
    assert "--cached-baseline" not in calls[0]


def test_numerical_policy_can_change_without_changing_measurements():
    from utils.experiments.theory.paper_reduce import _same_measurement_configuration
    certified = config(numerical_max_decimal_products=200_000_000,
                       numerical_max_variation_nodes=257, numerical_variation_absolute_width=1e-8)
    ordinary = config(numerical_max_decimal_products=0)
    assert _same_measurement_configuration(certified, ordinary)
    assert _same_measurement_configuration(ordinary, certified)


@pytest.mark.parametrize("change", [
    {"guidance_scale": 8.0}, {"num_loss_seeds": 93}, {"loss_seed": 17},
    {"num_mean_samples": 12000}, {"num_seeds": 40},
    {"counterfactual_unconditional": True, "counterfactual_steps": [0]},
])
def test_numerical_policy_migration_cannot_hide_changed_observations(change):
    from utils.experiments.theory.paper_reduce import _same_measurement_configuration
    previous = config(numerical_max_decimal_products=2_000_000)
    requested = config(numerical_max_decimal_products=0, **change)
    assert not _same_measurement_configuration(previous, requested)
