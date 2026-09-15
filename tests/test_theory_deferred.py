"""Endpoint preparation never invokes deferred path quadrature or loses references."""

import pandas as pd
import pytest

from tests.test_theory_reduction import CONFIG, saved_cache as _saved_cache
from utils.common.io import read_json
from utils.experiments.theory import reduce as reducer
from utils.experiments.theory.contracts import analysis_parent

saved_cache = _saved_cache


def _forbidden(*args, **kwargs):
    raise AssertionError("Deferred stage must not evaluate path integration")


def test_deferred_base_has_distinct_identity_preserves_current_reference_and_resumes(
    saved_cache, monkeypatch
):
    root = saved_cache["root"]
    ordinary = reducer.run_theory(root, device="cpu", **CONFIG)
    before = pd.concat(
        [
            pd.read_parquet(path)
            for path in sorted((ordinary / "trajectory_metrics").glob("*.parquet"))
        ],
        ignore_index=True,
    )
    monkeypatch.setattr(reducer, "proposition5_feedback", _forbidden)
    deferred = reducer.run_theory(
        root, device="cpu", defer_path_integration=True, **CONFIG
    )
    assert deferred != ordinary
    assert deferred.parent.name == "theory_measurements"
    manifest = read_json(deferred / "manifest.json")
    identity = read_json(deferred / "analysis_identity.json")
    assert manifest["complete"] and manifest["defer_path_integration"]
    assert not manifest["path_integration_complete"]
    assert identity["numerical_policy"]["defer_path_integration"]
    assert "paper_measurements.py" in identity["reducer_sources"]
    after = pd.concat(
        [
            pd.read_parquet(path)
            for path in sorted((deferred / "trajectory_metrics").glob("*.parquet"))
        ],
        ignore_index=True,
    )
    keys = ["run_id", "original_index", "record_id", "target_id", "seed", "step_index"]
    preserved = [
        *keys,
        "conditional_target_error_l2",
        "unconditional_target_error_l2",
        "candidate_unconditional_reference_error_l2",
        "candidate_radius_l2",
        "target_log_complement",
        "target_log_probability",
    ]
    pd.testing.assert_frame_equal(before[preserved], after[preserved], check_exact=True)
    nonterminal = after.step_index < CONFIG["num_inference_steps"] - 1
    assert after.loc[nonterminal, "candidate_condition_status"].eq("not_computed").all()
    assert (
        after.loc[nonterminal, "feedback_status"]
        .eq("path_integration_deferred_for_endpoint_stage")
        .all()
    )
    assert after.loc[nonterminal, "candidate_variation_l2"].isna().all()
    assert (
        after.loc[~nonterminal, "candidate_unconditional_reference_error_l2"]
        .notna()
        .all()
    )
    assert after.loc[~nonterminal, "target_log_complement"].notna().all()
    assert "terminal_accounting_status" in after
    assert after.loc[~nonterminal, "terminal_accounting_term0_rmse"].notna().all()
    monkeypatch.setattr(reducer, "_record_metrics", _forbidden)
    assert (
        reducer.run_theory(root, device="cpu", defer_path_integration=True, **CONFIG)
        == deferred
    )


def test_backing_path_and_boolean_policy_are_explicit(tmp_path):
    assert analysis_parent(tmp_path, **CONFIG).name == "theory_measurements"
    with pytest.raises(reducer.TheoryError, match="boolean"):
        reducer.run_theory(tmp_path, defer_path_integration="yes", **CONFIG)
