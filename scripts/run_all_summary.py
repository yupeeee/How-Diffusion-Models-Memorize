#!/usr/bin/env python3
"""Metadata-only accounting for final files produced by one run_all invocation.

Only project data/, logs/, outputs/, figures/ and checkpoints/ are counted.
Unchanged reused files, deletions, external model-hub caches, symlinks and
transient lock/staging files are excluded. Persisted .archives are included.
Sizes are logical file lengths, not allocated disk blocks; an updated file
contributes its final size, not its size delta. No artifact content is opened.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
import tempfile


SNAPSHOT_VERSION = 1
ROOT_NAMES = ("data", "logs", "outputs", "figures", "checkpoints")


class SummaryError(RuntimeError):
    """Accounting is unavailable; a partial result must not be reported as exact."""


def _absolute(path):
    return Path(path).expanduser().absolute()


def _transient(name):
    if name in {".lock", ".locks", ".figure_locks"}:
        return True
    if name.endswith((".lock", ".tmp", ".transaction.json", ".pending.json")):
        return True
    return name.startswith(".") and any(token in name for token in (".stage-", ".backup-", ".tmp-"))


def snapshot_files(root, *, exclude=()):
    """Return relative paths and (size, mtime_ns) receipts without reading content.

    Missing scope roots are empty. Permission or metadata failures anywhere in
    an existing tree invalidate the whole snapshot. Symlink entries, special
    files and transient staging/locking names are never traversed or counted.
    Exclusions are filesystem paths (relative paths follow the current cwd).
    """
    root = _absolute(root)
    excluded = {_absolute(path) for path in exclude}
    files, pending, errors = {}, [], []
    for name in ROOT_NAMES:
        directory = root / name
        if directory in excluded:
            continue
        try:
            mode = directory.stat(follow_symlinks=False).st_mode
        except FileNotFoundError:
            continue
        except OSError as error:
            errors.append(f"{name}: {error}")
            continue
        if stat.S_ISLNK(mode):
            continue
        if not stat.S_ISDIR(mode):
            errors.append(f"{name}: expected an artifact directory")
            continue
        pending.append(directory)
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = directory / entry.name
                    if path in excluded or _transient(entry.name):
                        continue
                    try:
                        metadata = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        # A concurrently deleted file has no final produced size.
                        continue
                    except OSError as error:
                        errors.append(f"{path.relative_to(root).as_posix()}: {error}")
                        continue
                    if stat.S_ISLNK(metadata.st_mode):
                        continue
                    if stat.S_ISDIR(metadata.st_mode):
                        pending.append(path)
                    elif stat.S_ISREG(metadata.st_mode):
                        files[path.relative_to(root).as_posix()] = {
                            "size": metadata.st_size, "mtime_ns": metadata.st_mtime_ns}
        except FileNotFoundError:
            continue
        except OSError as error:
            errors.append(f"{directory.relative_to(root).as_posix()}: {error}")
    if errors:
        shown = "; ".join(errors[:8])
        if len(errors) > 8:
            shown += f"; and {len(errors) - 8} more metadata errors"
        raise SummaryError("Artifact metadata could not be read completely: " + shown)
    return files


def write_snapshot(root, state):
    """Atomically save the baseline; exclude the state file even inside a scope root."""
    root, state = _absolute(root), _absolute(state)
    if state.is_symlink():
        raise SummaryError(f"Refusing a symlink accounting state: {state}")
    receipt = {"schema_version": SNAPSHOT_VERSION, "project_root": str(root),
               "scope_roots": list(ROOT_NAMES), "files": snapshot_files(root, exclude=(state,))}
    temporary = None
    try:
        state.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=state.parent,
                                         prefix="." + state.name + ".tmp-", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(receipt, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, state)
    except OSError as error:
        raise SummaryError(f"Cannot save accounting state {state}: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return receipt


def _read_snapshot(state):
    try:
        if state.is_symlink() or not stat.S_ISREG(state.stat(follow_symlinks=False).st_mode):
            raise SummaryError(f"Accounting state is not a regular file: {state}")
        with state.open(encoding="utf-8") as handle:
            receipt = json.load(handle)
    except (OSError, ValueError) as error:
        raise SummaryError(f"Cannot read accounting state {state}: {error}") from error
    if (not isinstance(receipt, dict) or receipt.get("schema_version") != SNAPSHOT_VERSION
            or receipt.get("scope_roots") != list(ROOT_NAMES)
            or not isinstance(receipt.get("project_root"), str)
            or not Path(receipt["project_root"]).is_absolute()
            or not isinstance(receipt.get("files"), dict)):
        raise SummaryError("Accounting state has an unsupported or incomplete schema")
    for name, metadata in receipt["files"].items():
        path = Path(name)
        if (path.is_absolute() or ".." in path.parts or len(path.parts) < 2
                or path.parts[0] not in ROOT_NAMES or not isinstance(metadata, dict)
                or set(metadata) != {"size", "mtime_ns"}
                or any(isinstance(metadata[key], bool) or not isinstance(metadata[key], int)
                       for key in ("size", "mtime_ns")) or metadata["size"] < 0):
            raise SummaryError("Accounting state contains an invalid file receipt")
    return receipt


def report_snapshot(state):
    """Account for final sizes of new or metadata-changed files, never deletions."""
    state = _absolute(state)
    receipt = _read_snapshot(state)
    before = receipt["files"]
    after = snapshot_files(receipt["project_root"], exclude=(state,))
    produced_bytes = new_files = updated_files = 0
    for name, metadata in after.items():
        previous = before.get(name)
        if previous is None:
            new_files += 1
        elif previous == metadata:
            continue
        else:
            updated_files += 1
        produced_bytes += metadata["size"]
    return {"produced_bytes": produced_bytes, "produced_files": new_files + updated_files,
            "new_files": new_files, "updated_files": updated_files,
            "deleted_files": sum(name not in after for name in before),
            "scope_roots": list(ROOT_NAMES)}


def format_bytes(value):
    """Format a nonnegative logical byte count in IEC units."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Byte count must be a nonnegative integer")
    if value < 1024:
        return f"{value} B"
    amount = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB")
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024
    raise AssertionError("Unreachable IEC formatting state")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot", help="save metadata before pipeline work")
    snapshot.add_argument("--project-root", required=True, type=Path)
    snapshot.add_argument("--state", required=True, type=Path)
    report = commands.add_parser("report", help="report new or updated final artifact sizes")
    report.add_argument("--state", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "snapshot":
            write_snapshot(args.project_root, args.state)
        else:
            result = report_snapshot(args.state)
            size, count = result["produced_bytes"], result["produced_files"]
            print(f"Total produced file size: {format_bytes(size)} ({size:,} bytes; {count:,} new or updated files)")
    except (SummaryError, OSError) as error:
        print(f"run_all summary unavailable: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
