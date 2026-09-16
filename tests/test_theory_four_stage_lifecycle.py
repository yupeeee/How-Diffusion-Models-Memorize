"""Unexecuted four-stage lifecycle and optional-inference isolation regressions."""
from pathlib import Path

import pytest

from utils.common.io import atomic_write_json, read_json
from utils.experiments.theory.contracts import FOUR_STAGE_OPTION_KEYS, TheoryError, numerical_config
from utils.experiments.theory.paper_contracts import (
    recompute_command, saved_plot_configuration, saved_scientific_configuration,
)


def config(**changes):
    return numerical_config(model_name="sdv1", scheduler_name="ddim", **changes)


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
                     "--recompute-experiments", "--center", "zero"]) == 0
    assert observed[0]["center"] == "zero"
    assert "cached_baseline" not in observed[0]


def test_shell_refine_center_can_inherit_saved_baseline_path(tmp_path):
    from tests.test_theory_plot_cli import _run_all_stub
    result, calls = _run_all_stub(tmp_path, ["--model", "sdv1", "--scheduler", "ddim",
                                            "--refine-numerics", "--center", "cached-baseline"])
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    assert "--refine-numerics" in calls[0]
    assert "--cached-baseline" not in calls[0]
