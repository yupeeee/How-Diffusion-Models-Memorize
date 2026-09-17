"""Source-authored checks for publication routing; no scientific inputs needed."""

from pathlib import Path

import pytest

from utils.experiments.figure_paths import publication_directory


@pytest.mark.parametrize("family", ["proximity", "theory"])
@pytest.mark.parametrize("role", ["experiment_S0_N20", "reference_S20_N20"])
def test_publication_directory_preserves_run_family_and_role(tmp_path, family, role):
    run = "sdv1_ddim_g7.5_T50_N20"
    source = tmp_path / "outputs" / run / family / role
    assert publication_directory(source) == tmp_path / "figures" / run / family / role
    assert not (tmp_path / "figures").exists()


def test_publication_directory_preserves_additional_namespace(tmp_path):
    relative = Path("sdv2_ddim_g7.5_T50_N20/proximity/reference_S20_N20/gmm")
    assert publication_directory(tmp_path / "outputs" / relative) == tmp_path / "figures" / relative


@pytest.mark.parametrize("relative", ["bundle", "outputs/run", "outputs/run/generation/experiment_S0_N20", "outputs/run/theory/../other"])
def test_publication_directory_rejects_noncanonical_paths(tmp_path, relative):
    with pytest.raises(ValueError):
        publication_directory(tmp_path / relative)
