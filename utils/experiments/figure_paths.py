"""Publication destinations independent of numerical cache locations.

This module only manipulates paths; it never creates directories or reads data.
"""

from pathlib import Path


def publication_directory(source_directory: str | Path) -> Path:
    """Mirror a canonical cache role below the project's ``figures`` root.

    For example, ``outputs/<run>/theory/experiment_S0_N20`` maps to
    ``figures/<run>/theory/experiment_S0_N20``. Additional namespaces are
    preserved, so seed roles and selection strategies cannot overwrite one
    another. Callers staging cache bundles must pass the final cache path,
    rather than the temporary staging path. Portable bundles use a canonical
    cache path derived from their saved configuration and the project root.

    Path/symlink validation and directory creation remain the publisher's job.
    """
    source = Path(source_directory)
    if ".." in source.parts:
        raise ValueError(f"Unsafe publication source path: {source}")
    source = source.absolute()
    for parent in source.parents:
        if parent.name != "outputs":
            continue
        relative = source.relative_to(parent)
        if len(relative.parts) < 3 or relative.parts[1] not in {"proximity", "theory"}:
            continue
        return parent.parent / "figures" / relative
    raise ValueError(
        "Publication source must be outputs/<experiment>/{proximity,theory}/<role>: "
        + str(source)
    )
