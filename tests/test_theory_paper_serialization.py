"""Caption serialization and export-progress regressions; author runs these tests."""
import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_theory_paper_plotting import compact_fixture
from utils.common.io import read_json
from utils.experiments import plotting as shared
from utils.experiments.theory import paper_plotting as plotting
from utils.experiments.theory.paper_registry import paper_registry


def test_caption_keeps_numpy_audit_flags_as_json_booleans():
    entry = paper_registry()[0]
    entry.update(status="available", scientific_hash="saved-science",
                 measurement_metadata={"counts": {"rows": np.int64(2), "resolved": np.bool_(True)}},
                 display_audit={"condition_zero_overlap": np.bool_(False),
                                "nested": [np.int64(3), np.float32(.5), np.array([True, False])],
                                "zero_dimensional": np.array(True)})
    caption = plotting._caption(entry)
    display = json.loads(next(line.removeprefix("Display: ") for line in caption.splitlines()
                              if line.startswith("Display: ")))
    assert display == {"condition_zero_overlap": False, "nested": [3, .5, [True, False]],
                       "zero_dimensional": True}
    assert type(display["condition_zero_overlap"]) is bool
    assert 'Counts: {"resolved": true, "rows": 2}' in caption
    assert isinstance(entry["display_audit"]["condition_zero_overlap"], np.bool_)


def test_render_serializes_captions_before_export_and_reports_each_file(tmp_path, monkeypatch, capsys):
    root = compact_fixture(tmp_path / "paper")
    closed, captioned, exports = [], [], []
    monkeypatch.setattr(plotting, "_draw", lambda *args: (object(), {
        "condition_zero_overlap": np.bool_(True), "count": np.int64(2),
        "nested": [np.array([False, True])]}))
    monkeypatch.setattr(plotting.plt, "close", closed.append)
    real_caption = plotting._caption
    def caption(entry):
        result = real_caption(entry)
        captioned.append(entry["stem"])
        return result
    monkeypatch.setattr(plotting, "_caption", caption)
    monkeypatch.setattr(plotting, "_finalize_presentations", lambda *args: {})
    def publish(output, figures, *, progress, export_options, formats=("png", "pdf")):
        assert formats == ("png", "pdf")
        assert export_options == {}
        assert len(captioned) == len(paper_registry())
        for _figure, names in figures:
            for relative in names.values():
                progress(relative, "saving")
                path = Path(output) / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"export fixture")
                progress(relative, "saved")
                exports.append(relative)
    monkeypatch.setattr(plotting, "publish_figures", publish)
    manifest = plotting.render_paper(root, formats=("png", "pdf"))
    saved = read_json(root / "figure_manifest.json")
    assert len(exports) == 12 and len(closed) == 6
    for entry in manifest["figures"]:
        assert entry["display_audit"] == {"condition_zero_overlap": True, "count": 2,
                                           "nested": [[False, True]]}
        assert type(entry["display_audit"]["condition_zero_overlap"]) is bool
    assert [entry["display_audit"] for entry in saved["figures"]] == [entry["display_audit"] for entry in manifest["figures"]]
    assert saved["files"] == manifest["files"]
    text = capsys.readouterr().err
    assert "[Theory] Preparing figures" in text
    assert "[Theory] Exporting figures" in text
    assert "figures/initial_loss_recovery.png" in text
    assert "Writing figure captions and manifest" in text


def test_unknown_audit_type_fails_before_export_and_closes_figure(tmp_path, monkeypatch):
    root = compact_fixture(tmp_path / "paper")
    figure, closed = object(), []
    monkeypatch.setattr(plotting, "_draw", lambda *args: (figure, {"invalid": object()}))
    monkeypatch.setattr(plotting.plt, "close", closed.append)
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid metadata reached PNG/PDF export")
    monkeypatch.setattr(plotting, "publish_figures", forbidden)
    with pytest.raises(TypeError, match="cannot convert"):
        plotting.render_paper(root, formats=("png", "pdf"))
    assert closed == [figure]
    assert not (root / "figure_manifest.json").exists()


@pytest.mark.parametrize("fail_pdf", [False, True])
def test_shared_export_progress_counts_completed_files_and_preserves_rollback(tmp_path, fail_pdf):
    names = {"png": "main/pair.png", "pdf": "main/pair.pdf"}
    for relative in names.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"previous")
    class SavedFigure:
        def savefig(self, path, *, format, **options):
            Path(path).write_bytes(b"replacement")
            if fail_pdf and format == "pdf":
                raise RuntimeError("failed PDF export")
    events = []
    def report(relative, status):
        events.append((relative, status))
    if fail_pdf:
        with pytest.raises(RuntimeError, match="failed PDF"):
            shared.publish_figures(tmp_path, [(SavedFigure(), names)], progress=report)
        assert events == [(names["png"], "saving"), (names["png"], "saved"), (names["pdf"], "saving")]
    else:
        shared.publish_figures(tmp_path, [(SavedFigure(), names)], progress=report)
        assert events == [(names["png"], "saving"), (names["png"], "saved"),
                          (names["pdf"], "saving"), (names["pdf"], "saved")]
    assert all((tmp_path / relative).read_bytes() == (b"previous" if fail_pdf else b"replacement")
               for relative in names.values())
    assert {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*") if path.is_file()} == set(names.values())
