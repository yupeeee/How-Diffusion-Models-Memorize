"""Unexecuted regression for the theory CLI's exact dependency-warning exception."""
from pathlib import Path
import runpy
import sys
import warnings


def test_spawned_theory_entry_filters_only_the_ignored_hub_symlink_warning(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "theory_validation.py"
    monkeypatch.setattr(sys, "path", sys.path.copy())
    message = (
        "The `local_dir_use_symlinks` argument is deprecated and ignored in "
        "`hf_hub_download`. Downloading to a local directory does not use symlinks anymore."
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        # This is how multiprocessing imports the script. It must configure
        # warnings without entering main(), measuring anything, or downloading.
        runpy.run_path(str(script), run_name="__mp_main__")
        for text, category, module in (
            (message, UserWarning, "huggingface_hub.utils._validators"),
            ("An unrelated Hub warning", UserWarning, "huggingface_hub.utils._validators"),
            (message, UserWarning, "another_dependency"),
            (message, RuntimeWarning, "huggingface_hub.utils._validators"),
        ):
            warnings.warn_explicit(text, category, "synthetic_warning_source.py", 1, module=module)
    assert [(str(item.message), item.category) for item in captured] == [
        ("An unrelated Hub warning", UserWarning),
        (message, UserWarning),
        (message, RuntimeWarning),
    ]
