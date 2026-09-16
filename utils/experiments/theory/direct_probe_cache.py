"""Read-only resolution of completed learned-probe tasks.

Two explicitly pinned transitions are supported: the original batching repair
and additive zero-baseline scalars for Gaussian reference probes. The latter
reuses only unaffected conditional/forward observations; Gaussian-reference
rows must be recomputed because old scalar rows cannot provide zero norms.

The cached-prediction batching fix changes orchestration source hashes without
changing any successfully published observation. Its predecessor could only
complete a cached task when the whole bank fit in its first microbatch; larger
banks failed before the completion marker was written. This one audited source
revision is therefore compatible after every other identity field and the
completed payload have been checked. No general formula-version fallback is
permitted, and no saved marker or numerical row is rewritten.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from utils.common.io import CacheIOError, canonical_hash, file_sha256, read_json

PROBE_SCHEMA_VERSION = 1
PROBE_SOURCE_KEY = "experiments/theory/direct_probes.py"
CACHED_BATCHING_PREDECESSOR_SHA256 = (
    "5622ba2da9ae2d709300d2c87af59dde13f12035f94907fc7e37810c2ed5e93c"
)
# Frozen to the audited fixed orchestrator revision.
# A future orchestrator edit must not inherit compatibility automatically.
AUDITED_FIXED_SOURCE_SHA256 = (
    "ca9fb0ae4e102ff504084825221a8e516d30af79980b86b5aa333ba7690fe6ce"
)
COMPATIBILITY_REASON = (
    "cached_prediction_microbatch_closure_fix; "
    "verified_complete_predecessor_tasks_have_unchanged_observations"
)


MATH_SOURCE_KEY = "experiments/theory/direct_probe_math.py"
ZERO_BASELINE_MATH_PREDECESSOR_SHA256 = (
    "5caebe520b6c23c1210ba31cdb966ea32c6306b1958eae6f54a586b9303ae40c"
)
# Pinned after the additive Gaussian-reference-only math change is audited.
# An unrelated future math/orchestrator edit must not inherit compatibility.
AUDITED_ZERO_BASELINE_MATH_SHA256 = (
    "d1d3faa961ab4928d9bfb3783f2473de538325d09adf171e9cc01641d7d2e437"
)
AUDITED_ZERO_BASELINE_PROBE_SHA256 = (
    "dcc4d47b77caa67c1e4e2532693120994fd0535ff1dd08873347e42a0ea54c0a"
)
ZERO_BASELINE_UNCHANGED_TABLES = frozenset({
    "forward_loss_draws", "gaussian_conditional", "forward_unconditional_loss",
})
ZERO_BASELINE_COMPATIBILITY_REASON = (
    "additive_zero_baseline_gaussian_reference_metrics_only; "
    "unchanged_conditional_and_forward_observations"
)


def _verified_completed_task(directory, task, path_for_task):
    data, marker = map(Path, path_for_task(directory, task))
    if any(not path.is_file() or path.is_symlink() for path in (data, marker)):
        return None
    if any(parent.is_symlink() for path in (data, marker) for parent in path.parents):
        return None
    try:
        saved = read_json(marker)
        if (
            type(saved.get("schema_version")) is not int
            or saved["schema_version"] != PROBE_SCHEMA_VERSION
            or saved.get("complete") is not True
            or saved.get("task_hash") != task["task_hash"]
            or saved.get("identity") != task["identity"]
            or type(saved.get("rows")) is not int
            or saved["rows"] != task["count"]
            or saved.get("sha256") != file_sha256(data)
        ):
            return None
    except (CacheIOError, OSError, ValueError):
        return None
    return task["task_hash"], data, marker


def resolve_completed_task(
    directory: str | Path,
    task: dict,
    *,
    path_for_task: Callable,
) -> tuple[str, Path, Path] | None:
    """Resolve an exact or narrowly compatible task, without mutating its files.

    The returned hash is the actual saved task hash. Callers must preserve it
    in source receipts and retain the original per-row ``task_hash`` values,
    rather than relabelling predecessor observations as newly computed rows.
    ``path_for_task(directory, task)`` is supplied by the probe orchestrator to
    keep path ownership and symlink policy in that module without an import
    cycle.
    """
    identity = task.get("identity")
    if not isinstance(identity, dict):
        return None
    if (
        type(task.get("count")) is not int
        or task["count"] < 1
        or identity.get("count") != task["count"]
        or identity.get("table") != task.get("table")
        or canonical_hash(identity) != task.get("task_hash")
    ):
        return None
    resolved = _verified_completed_task(directory, task, path_for_task)
    if resolved is not None:
        return resolved
    sources = identity.get("source_code")
    if not isinstance(sources, dict) or not isinstance(sources.get(PROBE_SOURCE_KEY), str):
        return None
    predecessor_sources = []
    if sources[PROBE_SOURCE_KEY] == AUDITED_FIXED_SOURCE_SHA256:
        predecessor_sources.append(sources | {PROBE_SOURCE_KEY: CACHED_BATCHING_PREDECESSOR_SHA256})
    if (sources[PROBE_SOURCE_KEY] == AUDITED_ZERO_BASELINE_PROBE_SHA256
            and sources.get(MATH_SOURCE_KEY) == AUDITED_ZERO_BASELINE_MATH_SHA256
            and task["table"] in ZERO_BASELINE_UNCHANGED_TABLES):
        # Only gaussian_reference_metrics gained columns. Its Gaussian-reference
        # tasks are intentionally excluded; exact source receipts for every
        # other function, policy, input, seed, checkpoint and law stay required.
        predecessor_sources.append(sources | {MATH_SOURCE_KEY: ZERO_BASELINE_MATH_PREDECESSOR_SHA256})
    for prior_sources in predecessor_sources:
        prior_identity = identity | {"source_code": prior_sources}
        prior_task = task | {"identity": prior_identity, "task_hash": canonical_hash(prior_identity)}
        resolved = _verified_completed_task(directory, prior_task, path_for_task)
        if resolved is not None:
            return resolved
    return None
