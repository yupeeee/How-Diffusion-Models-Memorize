"""Reference validation shares one canonical anchor across spawned readers."""

from __future__ import annotations

import pytest
import torch

from tests.test_theory_reduction import CONFIG, saved_cache as saved_cache
from utils.experiments.theory import cache_reader, centers
from utils.experiments.theory.contracts import TheoryError


@pytest.fixture
def reference_sources(saved_cache):
    sources = cache_reader.discover_sources(saved_cache["root"], **CONFIG)
    return sources, cache_reader.load_schedule(sources)


def test_spawned_reference_center_matches_serial_exactly(reference_sources):
    sources, schedule = reference_sources
    serial_center, serial_metadata = centers.reference_center(sources, schedule)
    # Discovery order must not change the frozen canonical record.
    sources.records["reference"].reverse()
    messages = []
    parallel_center, parallel_metadata = centers.reference_center(
        sources,
        schedule,
        devices=(torch.device("cpu"), torch.device("cpu")),
        progress=messages.append,
    )
    assert torch.equal(parallel_center, serial_center)
    assert parallel_metadata == serial_metadata
    assert parallel_metadata["canonical_original_index"] == "00001"
    assert parallel_metadata["source_sample_count"] == CONFIG["num_seeds"]
    assert parallel_metadata["verified_reference_record_count"] == 4
    assert any("2 CPU reader workers" in message for message in messages)
    assert messages[-1] == "reference initial consistency 4/4"


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("initial", "Repeated initial states differ"),
        ("epsilon", "Repeated initial unconditional outputs inconsistent"),
        ("hash", "Invalid generation record 00003"),
    ],
)
def test_spawned_reader_rejects_invalid_noncanonical_record(
    reference_sources, failure, message
):
    sources, schedule = reference_sources
    # 00003 belongs to the spawned worker, away from the canonical record and
    # the local reader. Refitting an anchor per shard would hide this mismatch.
    index = "00003"
    if failure == "epsilon":
        path = sources.reference.noise_prediction_path(index)
        u, c = torch.load(path, weights_only=True, map_location="cpu")
        u[:, 0] += 1
        torch.save((u, c), path)
    elif failure == "initial":
        path = sources.reference.latent_path(index)
        z = torch.load(path, weights_only=True, map_location="cpu")
        z[:, 0] += 0.125
        torch.save(z, path)
    else:
        with sources.reference.latent_path(index).open("ab") as stream:
            stream.write(b"changed-after-discovery")
    with pytest.raises(TheoryError, match=message):
        centers.reference_center(
            sources,
            schedule,
            devices=(torch.device("cpu"), torch.device("cpu")),
            # Semantic failures deliberately bypass hashes to reach comparison.
            verify_hashes=failure == "hash",
        )


def test_single_canonical_record_does_not_spawn(reference_sources, monkeypatch):
    sources, schedule = reference_sources
    sources.records["reference"] = sources.records["reference"][:1]

    def forbidden(*args, **kwargs):
        raise AssertionError("No reference comparison tasks need a process")

    monkeypatch.setattr(centers, "ProcessPoolExecutor", forbidden)
    center, metadata = centers.reference_center(
        sources, schedule, devices=(torch.device("cpu"),) * 4
    )
    assert center.device.type == "cpu"
    assert metadata["verified_reference_record_count"] == 1
    assert metadata["source_sample_count"] == CONFIG["num_seeds"]
    assert metadata["repeated_epsilon_max_abs_difference"] == 0
    assert metadata["repeated_epsilon_max_abs_tolerance"] == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_subprocess_anchor_transport_preserves_storage_dtype(dtype, monkeypatch):
    z = torch.tensor([[[[0.125, -2.0], [3.5, 1024.0]]]], dtype=dtype)
    u = z / 4

    def validate(paths, records, canonical_z, canonical_u, **kwargs):
        assert canonical_z.dtype == dtype
        assert canonical_u.dtype == dtype
        assert canonical_z.device.type == canonical_u.device.type == "cpu"
        assert torch.equal(canonical_z, z)
        assert torch.equal(canonical_u, u)
        return 2, 0.0, 0.001

    monkeypatch.setattr(centers, "_reference_initial_shard", validate)
    result = centers._reference_initial_subprocess(
        None,
        (),
        z.view(torch.uint8).numpy(),
        u.view(torch.uint8).numpy(),
        z.dtype,
        u.dtype,
        verify_hashes=True,
        worker_count=2,
    )
    assert result == (2, 0.0, 0.001)
