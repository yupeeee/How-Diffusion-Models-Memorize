"""Regression checks for bounded device staging and packed scalar transfers."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from tests.test_theory_reduction import CONFIG, saved_cache as _saved_cache
from utils.experiments.theory import reduce as reducer
from utils.experiments.theory.cache_reader import discover_sources, load_schedule
from utils.experiments.theory.centers import choose_center
from utils.experiments.theory.scheduler_adapter import SchedulerAdapter

saved_cache = _saved_cache


def test_scalar_packing_preserves_types_nulls_and_large_integers(monkeypatch):
    original = torch.Tensor.cpu
    transfers = []

    def counted(value, *args, **kwargs):
        transfers.append((value.dtype, tuple(value.shape)))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted)
    result = reducer._scalar_rows(
        {
            "error": torch.tensor([1.5, float("nan")], dtype=torch.float64),
            "scale": torch.tensor(2.0, dtype=torch.float64),
            "integer": torch.tensor([2**60 + 1, 2**60 + 3]),
            "flag": torch.tensor([True, False]),
            "status": "unavailable_population_loss",
            "loss": None,
        },
        2,
    )
    assert result == [
        dict(
            error=1.5,
            scale=2.0,
            integer=2**60 + 1,
            flag=True,
            status="unavailable_population_loss",
            loss=None,
        ),
        dict(
            error=None,
            scale=2.0,
            integer=2**60 + 3,
            flag=False,
            status="unavailable_population_loss",
            loss=None,
        ),
    ]
    assert len(transfers) == 3  # One packed transfer per dtype, not per column.
    assert transfers[0] == (torch.float64, (2, 2))
    with pytest.raises(reducer.TheoryError, match="unexpected shape"):
        reducer._scalar_rows({"not_scalar_per_query": torch.ones(2, 4)}, 2)


def test_time_blocks_preserve_all_seed_step_endpoint_metrics(saved_cache, monkeypatch):
    sources = discover_sources(saved_cache["root"], **CONFIG)
    schedule = load_schedule(sources)
    support, _, _ = reducer.build_support(sources, query_chunk_size=2)
    center, _ = choose_center(sources, schedule)
    adapter = SchedulerAdapter(schedule)
    record = sources.selected[0]
    results = []
    for block in (1, 2, 8):
        monkeypatch.setattr(reducer, "TIME_BLOCK_SIZE", block)
        result = reducer._record_metrics(
            sources,
            record,
            schedule,
            adapter,
            support,
            center,
            None,
            device="cpu",
            query_chunk_size=2,
        )
        results.append(result[:3])
    for result in results[1:]:
        for expected, actual in zip(results[0], result, strict=True):
            pd.testing.assert_frame_equal(expected, actual, check_exact=True)
    trajectory, initial, endpoint = results[0]
    assert set(zip(trajectory.seed, trajectory.step_index)) == {
        (seed, step) for seed in range(3) for step in range(3)
    }
    assert initial.step_index.eq(0).all()
    assert endpoint.step_index.eq(2).all()
    assert len(endpoint) == 3
