"""Read-only resolution of completed learned-probe tasks.

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
    if sources[PROBE_SOURCE_KEY] != AUDITED_FIXED_SOURCE_SHA256:
        return None
    prior_identity = identity | {
        "source_code": sources | {PROBE_SOURCE_KEY: CACHED_BATCHING_PREDECESSOR_SHA256}
    }
    prior_task = task | {
        "identity": prior_identity,
        "task_hash": canonical_hash(prior_identity),
    }
    return _verified_completed_task(directory, prior_task, path_for_task)
