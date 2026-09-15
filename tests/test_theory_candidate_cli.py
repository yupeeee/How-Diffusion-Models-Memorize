"""Paper routing supersedes saved discovery-suite choices."""

import pytest
from tests.test_theory_plot_cli import _run_all_stub, _option


@pytest.mark.parametrize("mode", ["--plot", "--recompute-experiments", None])
def test_paper_suite_reaches_only_theory_and_every_preflight(tmp_path, mode):
    result, calls = _run_all_stub(
        tmp_path, ([mode] if mode else []) + ["--figure-suite", "paper"]
    )
    assert result.returncode == 0, result.stderr
    theory = [c for c in calls if c[0] == "theory_validation.sh"]
    assert len(theory) == (8 if mode == "--plot" else 4)
    assert all(_option(c, "--figure-suite") == "paper" for c in theory)
    assert all(
        "--figure-suite" not in c for c in calls if c[0] != "theory_validation.sh"
    )
    if mode == "--recompute-experiments":
        assert len(calls) == 4


def test_default_is_paper_regardless_of_old_saved_suite(tmp_path, monkeypatch):
    from scripts import theory_validation
    from tests.test_theory_paper_plotting import compact_fixture
    from utils.experiments.theory.paper_contracts import PaperPaths
    from utils.experiments.theory import paper_contracts, candidate_contracts
    from utils.common.io import atomic_write_json

    config = dict(model_name="sdv1", scheduler_name="ddim", num_seeds=2)
    bundle = PaperPaths.build(tmp_path, **config).output_directory
    compact_fixture(bundle)
    atomic_write_json(
        bundle.parent.parent / "theory_suite.json", {"figure_suite": "candidates"}
    )
    monkeypatch.setattr(theory_validation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        candidate_contracts,
        "selected_suite",
        lambda *a, **k: pytest.fail("Read retired suite selection"),
    )
    calls = []
    monkeypatch.setattr(
        paper_contracts, "render_saved_paper", lambda b, **k: calls.append(b)
    )
    assert theory_validation.main(["--plot", "--N", "2"]) == 0
    assert calls == [bundle]


@pytest.mark.parametrize("suite", ["main", "candidates", "best"])
def test_retired_or_invalid_suite_fails_before_any_stage(tmp_path, suite):
    result, calls = _run_all_stub(tmp_path, ["--plot", "--figure-suite", suite])
    assert result.returncode != 0
    assert calls == []


def test_default_root_routes_paper_and_diagnostics_only_to_theory(tmp_path):
    result, calls = _run_all_stub(tmp_path, ["--plot", "--diagnostics"])
    assert result.returncode == 0, result.stderr
    for call in calls:
        if call[0] == "theory_validation.sh":
            assert _option(call, "--figure-suite") == "paper"
            assert "--diagnostics" in call
        else:
            assert "--diagnostics" not in call


def test_saved_proximity_mismatch_requests_normal_pipeline(
    tmp_path, monkeypatch, capsys
):
    from scripts import theory_validation
    from tests.test_theory_paper_plotting import compact_fixture
    from utils.experiments.theory.paper_contracts import PaperPaths
    from utils.experiments.theory.contracts import TheoryError

    bundle = PaperPaths.build(
        tmp_path, model_name="sdv1", scheduler_name="ddim", num_seeds=2
    ).output_directory
    compact_fixture(bundle)
    monkeypatch.setattr(theory_validation, "PROJECT_ROOT", tmp_path)

    def mismatch(*args):
        raise TheoryError("saved proximity configuration differs at: selection_hash")

    monkeypatch.setattr(theory_validation, "validate_saved_proximity", mismatch)
    assert (
        theory_validation.main(["--N", "2", "--validate-only", "--validate-proximity"])
        == 1
    )
    message = capsys.readouterr().err
    assert "selection_hash" in message
    assert "./run_all.sh --model sdv1 --scheduler ddim" in message
    assert "--recompute-experiments" not in message
    assert not list(bundle.rglob("*.png"))
