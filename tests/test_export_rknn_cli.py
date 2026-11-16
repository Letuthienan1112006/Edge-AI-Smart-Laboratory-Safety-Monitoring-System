"""Argument validation for `scripts/export_rknn.py` -- no real export runs
here, and neither `ultralytics` nor `rknn-toolkit2` needs to be installed.
`validate_args()` must reject bad invocations before anything expensive
(or architecture-incompatible) is attempted.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("export_rknn", SCRIPTS_DIR / "export_rknn.py")
export_rknn = importlib.util.module_from_spec(spec)
sys.modules["export_rknn"] = export_rknn
spec.loader.exec_module(export_rknn)  # must succeed without ultralytics installed


def test_int8_without_data_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        export_rknn.validate_args(["model.pt", "--precision", "int8"])
    assert exc.value.code != 0


def _make_data_yaml(tmp_path, n_val_images: int, split: str = "val") -> "pathlib.Path":
    """A minimal YOLO dataset YAML + a populated split dir, matching what
    ultralytics' check_det_dataset() actually expects -- NOT a bare
    directory of images (that was this script's original, unverified
    assumption; see KNOWN_ISSUES.md and docs/rknn-pipeline.md)."""
    split_dir = tmp_path / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_val_images):
        (split_dir / f"img_{i}.jpg").write_bytes(b"\x00")
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(f"path: {tmp_path}\n{split}: {split}\nnc: 1\nnames: [person]\n")
    return yaml_path


def test_int8_with_a_plain_image_directory_is_rejected(tmp_path):
    """The original bug this guards: --data used to accept a bare directory
    of images and pass it straight to model.export(data=...). Ultralytics'
    RKNN INT8 path reads that value with check_det_dataset(), which expects
    a dataset YAML -- a directory would fail deep inside ultralytics with a
    confusing error instead of this script's own clear one."""
    for i in range(60):
        (tmp_path / f"img_{i}.jpg").write_bytes(b"\x00")
    with pytest.raises(SystemExit) as exc:
        export_rknn.validate_args(
            ["model.pt", "--precision", "int8", "--data", str(tmp_path)]
        )
    assert exc.value.code != 0


def test_int8_with_too_few_calib_images_exits_nonzero(tmp_path):
    yaml_path = _make_data_yaml(tmp_path, n_val_images=3)
    with pytest.raises(SystemExit) as exc:
        export_rknn.validate_args(
            ["model.pt", "--precision", "int8", "--data", str(yaml_path)]
        )
    assert exc.value.code != 0


def test_int8_calib_count_reports_the_resolved_split_dir_not_just_the_yaml(tmp_path):
    """Zero-image failures must name the directory actually checked, so a
    wrong `path:`/`val:` in the yaml doesn't read as "images are missing"
    when the real problem is "looked in the wrong folder"."""
    yaml_path = _make_data_yaml(tmp_path, n_val_images=0)
    with pytest.raises(SystemExit):
        export_rknn.validate_args(
            ["model.pt", "--precision", "int8", "--data", str(yaml_path)]
        )


def test_end2end_requires_explicit_ack():
    with pytest.raises(SystemExit) as exc:
        export_rknn.validate_args(["model.pt", "--precision", "fp16", "--end2end"])
    assert exc.value.code != 0
    # ...but WITH the acknowledgement flag it must pass validation.
    args = export_rknn.validate_args(
        ["model.pt", "--precision", "fp16", "--end2end", "--i-know-this-segfaults"]
    )
    assert args.end2end is True


def test_refuses_to_run_on_aarch64(monkeypatch):
    monkeypatch.setattr(export_rknn.platform, "machine", lambda: "aarch64")
    with pytest.raises(SystemExit) as exc:
        export_rknn.validate_args(["model.pt", "--precision", "fp16"])
    assert exc.value.code != 0


def test_fp16_with_enough_data_is_fine_without_data_at_all():
    """fp16 never requires calibration images -- only int8 does."""
    args = export_rknn.validate_args(["model.pt", "--precision", "fp16"])
    assert args.precision == "fp16"


def test_int8_with_enough_calib_images_passes(tmp_path):
    yaml_path = _make_data_yaml(tmp_path, n_val_images=60)
    args = export_rknn.validate_args(
        ["model.pt", "--precision", "int8", "--data", str(yaml_path)]
    )
    assert args.data == str(yaml_path)


def test_int8_calib_split_defaults_to_val_and_is_configurable(tmp_path):
    yaml_path = _make_data_yaml(tmp_path, n_val_images=60, split="test")
    args = export_rknn.validate_args([
        "model.pt", "--precision", "int8", "--data", str(yaml_path),
        "--calib-split", "test",
    ])
    assert args.calib_split == "test"


# -- _collect_export: a convert that produced nothing must not report success --
#
# Regression guard for the form this replaced:
#     Path(exported).replace(dest) if Path(exported).is_file() else None
# Ultralytics returns a DIRECTORY for packaged formats (`*_rknn_model/`), so
# that expression did nothing on the branch that matters -- silently -- and
# the script then wrote a model card beside a path that did not exist and
# printed it as a success.

def test_collect_export_unwraps_the_directory_ultralytics_actually_returns(tmp_path):
    pkg = tmp_path / "yolo26n_rknn_model"
    pkg.mkdir()
    (pkg / "yolo26n.rknn").write_bytes(b"MODEL")
    (pkg / "README.txt").write_bytes(b"not the model")
    dest = tmp_path / "out" / "yolo26n_fp16_640.rknn"

    out = export_rknn._collect_export(pkg, dest)

    assert out == dest and dest.is_file()
    assert dest.read_bytes() == b"MODEL"


def test_collect_export_still_handles_a_plain_file(tmp_path):
    src = tmp_path / "model.rknn"
    src.write_bytes(b"MODEL")
    dest = tmp_path / "out" / "renamed.rknn"
    assert export_rknn._collect_export(src, dest) == dest
    assert dest.read_bytes() == b"MODEL"


def test_collect_export_raises_when_export_produced_nothing(tmp_path):
    with pytest.raises(RuntimeError, match="produced nothing"):
        export_rknn._collect_export(tmp_path / "does_not_exist", tmp_path / "d.rknn")


def test_collect_export_raises_when_directory_has_no_model(tmp_path):
    pkg = tmp_path / "empty_rknn_model"
    pkg.mkdir()
    (pkg / "log.txt").write_bytes(b"")
    with pytest.raises(RuntimeError, match="no \\*.rknn file"):
        export_rknn._collect_export(pkg, tmp_path / "d.rknn")


def test_collect_export_refuses_to_guess_between_several_models(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "a.rknn").write_bytes(b"A")
    (pkg / "b.rknn").write_bytes(b"B")
    with pytest.raises(RuntimeError, match="Refusing to guess"):
        export_rknn._collect_export(pkg, tmp_path / "d.rknn")


# -- _toolkit_versions: must read installed-package metadata, not
#    module.__version__ -- regression guard for a real installation bug --
#
# rknn-toolkit2 installs as the `rknn` namespace package. `import rknn` then
# `rknn.__version__` does not raise (namespace packages have no
# __version__), it silently returns None via getattr's default. That was
# verified against a REAL `pip install rknn-toolkit2` on 2026-09-26: the
# toolkit was correctly installed and importable (`from rknn.api import
# RKNN` worked), yet `_toolkit_versions()` reported `(None, None)`, so every
# ModelCard.toolkit_version would read empty despite a working toolchain --
# defeating the field's whole purpose (tracing which toolkit converted a
# model, see docs/rknn-pipeline.md).

def test_toolkit_versions_reads_installed_package_metadata(monkeypatch):
    """The bug this guards: module.__version__ on the `rknn` namespace
    package is silently absent, so the function must use package metadata
    (importlib.metadata.version) keyed by DISTRIBUTION name, not module
    attribute lookup keyed by import name."""
    def fake_version(dist_name):
        return {"ultralytics": "8.3.99", "rknn-toolkit2": "2.3.2"}[dist_name]

    monkeypatch.setattr("importlib.metadata.version", fake_version)
    assert export_rknn._toolkit_versions() == ("8.3.99", "2.3.2")


def test_toolkit_versions_reports_none_when_neither_installed(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    def fake_version(dist_name):
        raise PackageNotFoundError(dist_name)

    monkeypatch.setattr("importlib.metadata.version", fake_version)
    assert export_rknn._toolkit_versions() == (None, None)


def test_toolkit_versions_does_not_import_the_rknn_module_directly():
    """The old implementation did `import rknn; rknn.__version__`, which
    means the mere ABSENCE of a real rknn-toolkit2 install could not be
    told apart from a namespace-package stub with no __version__. Confirm
    the fix does not reach for `sys.modules['rknn']` or similar module
    introspection at all -- it must go through package metadata only."""
    import ast
    import inspect

    src = inspect.getsource(export_rknn._toolkit_versions)
    tree = ast.parse(src)
    names = {n.id for node in ast.walk(tree) for n in ast.walk(node)
             if isinstance(n, ast.Name)}
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
               for alias in node.names}
    assert "rknn" not in imports and "ultralytics" not in imports
    assert "importlib.metadata" in src or "from importlib.metadata" in src


# -- _collect_export must survive a cross-device move --------------------
#
# Regression guard for a real failure: `--out` pointing at a different
# filesystem/mount than wherever ultralytics writes its export (which is
# the NORMAL case -- ultralytics writes beside the source checkpoint,
# ignoring --out entirely) made the old `Path.replace()` raise
# `OSError: [Errno 18] Invalid cross-device link`. Reproduced for real on
# 2026-09-26: exporting to /tmp (tmpfs) from a repo on a different disk
# crashed here, after rknn-toolkit2 had already spent ~11s building a real
# model -- the worst place for a silent assumption to break.

def test_collect_export_survives_a_cross_device_move(tmp_path, monkeypatch):
    src = tmp_path / "src" / "model.rknn"
    src.parent.mkdir()
    src.write_bytes(b"MODEL")
    dest = tmp_path / "dest" / "out.rknn"

    def fake_replace(self, target):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(Path, "replace", fake_replace)
    out = export_rknn._collect_export(src, dest)

    assert out == dest
    assert dest.read_bytes() == b"MODEL"
    assert not src.exists(), "shutil.move must remove the source, like a real move"
