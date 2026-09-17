"""Opt-in single-format publication; test source only, authored without execution."""
from pathlib import Path

import pytest

from utils.experiments import plotting


class SavedFigure:
    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    def savefig(self, path, **options):
        self.calls.append(dict(options))
        Path(path).write_bytes(b"updated figure")
        if self.fail:
            raise RuntimeError("export failed")


@pytest.mark.parametrize("publisher", [plotting.publish_figures, plotting._publish_figures])
def test_pdf_only_updates_pdf_and_leaves_existing_png_untouched(tmp_path, publisher):
    directory = tmp_path / "figures"
    directory.mkdir()
    png, pdf = directory / "panel.png", directory / "panel.pdf"
    png.write_bytes(b"existing PNG")
    pdf.write_bytes(b"existing PDF")
    previous_png = (png.read_bytes(), png.stat().st_mtime_ns)
    figure, events = SavedFigure(), []
    publisher(tmp_path, [(figure, {"pdf": "figures/panel.pdf"})], formats=("pdf",),
              progress=lambda name, status: events.append((name, status)))
    assert figure.calls == [dict(format="pdf", dpi=150, bbox_inches="tight", pad_inches=.05)]
    assert (png.read_bytes(), png.stat().st_mtime_ns) == previous_png
    assert pdf.read_bytes() == b"updated figure"
    assert events == [("figures/panel.pdf", "saving"), ("figures/panel.pdf", "saved")]
    assert sorted(path.name for path in directory.iterdir()) == ["panel.pdf", "panel.png"]


def test_default_remains_strictly_paired_and_explicit_png_only_is_supported(tmp_path):
    figure = SavedFigure()
    with pytest.raises(plotting.PlottingError, match="exactly match requested"):
        plotting.publish_figures(tmp_path / "invalid", [(figure, {"pdf": "panel.pdf"})])
    assert not (tmp_path / "invalid").exists() and not figure.calls
    plotting.publish_figures(tmp_path / "png", [(figure, {"png": "panel.png"})], formats=("png",))
    assert [call["format"] for call in figure.calls] == ["png"]
    assert list((tmp_path / "png").iterdir()) == [tmp_path / "png" / "panel.png"]


@pytest.mark.parametrize("formats", [(), ("svg",), ("pdf", "pdf"), "pdf", None, (1,)])
def test_invalid_format_selection_is_rejected_before_any_writes(tmp_path, formats):
    output, figure = tmp_path / "invalid", SavedFigure()
    with pytest.raises(plotting.PlottingError, match="nonempty, unique subset"):
        plotting.publish_figures(output, [(figure, {"pdf": "panel.pdf"})], formats=formats)
    assert not output.exists() and not figure.calls


@pytest.mark.parametrize("names", [
    {}, {"png": "panel.png"}, {"pdf": "panel.pdf", "png": "panel.png"},
    {"pdf": "panel.png"}, {"pdf": "../outside.pdf"}, {"pdf": "/absolute.pdf"},
])
@pytest.mark.parametrize("publisher", [plotting.publish_figures, plotting._publish_figures])
def test_single_format_rejects_extra_missing_or_unsafe_paths(tmp_path, names, publisher):
    output, figure = tmp_path / "invalid", SavedFigure()
    with pytest.raises(plotting.PlottingError):
        publisher(output, [(figure, names)], formats=("pdf",))
    assert not output.exists() and not figure.calls


def test_pdf_only_failure_preserves_entire_requested_set_and_unrequested_pngs(tmp_path):
    pending = []
    for stem, fail in (("first", False), ("second", True)):
        (tmp_path / f"{stem}.png").write_bytes(b"original PNG")
        (tmp_path / f"{stem}.pdf").write_bytes(b"original PDF")
        pending.append((SavedFigure(fail=fail), {"pdf": f"{stem}.pdf"}))
    with pytest.raises(RuntimeError, match="export failed"):
        plotting.publish_figures(tmp_path, pending, formats=("pdf",))
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == {
        "first.png": b"original PNG", "second.png": b"original PNG",
        "first.pdf": b"original PDF", "second.pdf": b"original PDF"}


def test_pdf_only_install_failure_rolls_back_all_requested_pdfs(tmp_path, monkeypatch):
    pending = []
    for stem in ("first", "second"):
        (tmp_path / f"{stem}.png").write_bytes(b"original PNG")
        (tmp_path / f"{stem}.pdf").write_bytes(b"original PDF")
        pending.append((SavedFigure(), {"pdf": f"{stem}.pdf"}))
    replace = plotting.os.replace
    interrupted = False
    def fail_second_install(source, destination):
        nonlocal interrupted
        if Path(destination).name == "second.pdf" and not interrupted:
            interrupted = True
            raise OSError("installation interrupted")
        return replace(source, destination)
    monkeypatch.setattr(plotting.os, "replace", fail_second_install)
    with pytest.raises(OSError, match="installation interrupted"):
        plotting.publish_figures(tmp_path, pending, formats=("pdf",))
    assert interrupted
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == {
        "first.png": b"original PNG", "second.png": b"original PNG",
        "first.pdf": b"original PDF", "second.pdf": b"original PDF"}


def test_pdf_only_rejects_export_options_for_unrequested_png(tmp_path):
    with pytest.raises(plotting.PlottingError, match="unknown output"):
        plotting.publish_figures(tmp_path / "invalid", [(SavedFigure(), {"pdf": "panel.pdf"})],
                                 formats=("pdf",), export_options={"panel.png": {"dpi": 150}})
    assert not (tmp_path / "invalid").exists()


def test_pdf_only_parent_symlink_cannot_escape_output_directory(tmp_path):
    output, outside = tmp_path / "output", tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "figures").symlink_to(outside, target_is_directory=True)
    with pytest.raises(plotting.PlottingError, match="symlink"):
        plotting.publish_figures(output, [(SavedFigure(), {"pdf": "figures/panel.pdf"})], formats=("pdf",))
    assert not list(outside.iterdir())
