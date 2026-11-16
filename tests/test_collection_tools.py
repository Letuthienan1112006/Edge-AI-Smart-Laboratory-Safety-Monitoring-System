"""Collection tooling: resume logic of the guided recorder and the by-session dataset builder.

These run the real scripts as subprocesses on temp dirs. They need PyYAML (the scripts do), so they
skip on an interpreter without it (the project venv has it).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                          capture_output=True, text=True, cwd=ROOT)


def _plan(tmp_path: Path) -> Path:
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump({"batch": "T", "scenes": [
        {"id": "01_a", "seconds": 5, "split": "train", "text": "a"},
        {"id": "02_b", "seconds": 5, "split": "test", "text": "b"}]}))
    return p


def _session(root: Path, name: str, labels: dict[str, str | None]) -> None:
    (root / name / "images").mkdir(parents=True)
    (root / name / "labels").mkdir(parents=True)
    for stem, content in labels.items():
        (root / name / "images" / f"{stem}.jpg").write_bytes(b"x")
        if content is not None:
            (root / name / "labels" / f"{stem}.txt").write_text(content)


def test_directory_without_finished_marker_is_not_counted_as_recorded(tmp_path):
    out = tmp_path / "sessions"
    (out / "T_01_a").mkdir(parents=True)                       # a crashed attempt: folder only
    r = _run("collect_guided.py", "--plan", str(_plan(tmp_path)), "--out", str(out), "--list")
    assert "[ ] T_01_a" in r.stdout
    (out / "T_01_a" / "scene.json").write_text("{}")           # finished marker
    r = _run("collect_guided.py", "--plan", str(_plan(tmp_path)), "--out", str(out), "--list")
    assert "[x] T_01_a" in r.stdout and "[ ] T_02_b" in r.stdout


def test_build_splits_by_session_and_never_shares_a_session(tmp_path):
    lab = tmp_path / "labeled"
    _session(lab, "T_01_a", {"f0": "0 0.5 0.5 0.2 0.2\n", "f1": ""})   # an empty label file is valid
    _session(lab, "T_02_b", {"g0": "1 0.5 0.5 0.1 0.1\n"})
    out = tmp_path / "ds"
    r = _run("build_lab_dataset.py", "--labeled", str(lab), "--plan", str(_plan(tmp_path)), "--out", str(out))
    assert r.returncode == 0, r.stderr + r.stdout
    train = {p.name.split("__")[0] for p in (out / "images" / "train").iterdir()}
    test = {p.name.split("__")[0] for p in (out / "images" / "test").iterdir()}
    assert train == {"T_01_a"} and test == {"T_02_b"}
    assert len(list((out / "images" / "train").iterdir())) == 2


def test_build_refuses_a_missing_label_file_and_leaves_no_output(tmp_path):
    lab = tmp_path / "labeled"
    _session(lab, "T_01_a", {"f0": None})
    _session(lab, "T_02_b", {"g0": "0 0.5 0.5 0.1 0.1\n"})
    out = tmp_path / "ds"
    r = _run("build_lab_dataset.py", "--labeled", str(lab), "--plan", str(_plan(tmp_path)), "--out", str(out))
    assert r.returncode != 0 and "no label file" in r.stdout and not out.exists()


def test_build_refuses_an_unknown_class_id(tmp_path):
    lab = tmp_path / "labeled"
    _session(lab, "T_01_a", {"f0": "7 0.5 0.5 0.1 0.1\n"})
    _session(lab, "T_02_b", {"g0": "0 0.5 0.5 0.1 0.1\n"})
    r = _run("build_lab_dataset.py", "--labeled", str(lab), "--plan", str(_plan(tmp_path)), "--out", str(tmp_path / "ds"))
    assert r.returncode != 0 and "bad line" in r.stdout


def test_build_refuses_when_the_test_split_is_empty(tmp_path):
    lab = tmp_path / "labeled"
    _session(lab, "T_01_a", {"f0": "0 0.5 0.5 0.1 0.1\n"})
    r = _run("build_lab_dataset.py", "--labeled", str(lab), "--plan", str(_plan(tmp_path)), "--out", str(tmp_path / "ds"))
    assert r.returncode != 0 and "test split is empty" in (r.stdout + r.stderr)


# -- Roboflow import + pilot holdout --------------------------------------------

def _rf_export(tmp_path: Path, names: list[str], lines_by_img: dict[str, list[str]], splits=("train", "valid", "test")) -> Path:
    root = tmp_path / "rf"
    root.mkdir()
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names}))
    for i, (stem, lines) in enumerate(lines_by_img.items()):
        sp = splits[i % len(splits)]
        (root / sp / "images").mkdir(parents=True, exist_ok=True)
        (root / sp / "labels").mkdir(parents=True, exist_ok=True)
        (root / sp / "images" / f"{stem}_jpg.rf.{i:012x}.jpg").write_bytes(b"x")
        (root / sp / "labels" / f"{stem}_jpg.rf.{i:012x}.txt").write_text("\n".join(lines) + "\n")
    return root


def test_import_remaps_classes_by_name_not_by_position(tmp_path):
    # Roboflow order (alphabetical): lab_coat=0, person=1  -> project order person=0, lab_coat=1
    root = _rf_export(tmp_path, ["lab_coat", "person"],
                      {"boot1__0": ["1 0.5 0.5 0.4 0.8", "0 0.5 0.3 0.1 0.05"]})
    out = tmp_path / "lab"
    r = _run("import_roboflow.py", str(root), "--out", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    rows = (out / "boot1" / "labels" / "boot1__0.txt").read_text().split("\n")
    assert rows[0].startswith("0 ") and rows[1].startswith("1 ")      # person -> 0, lab_coat -> 1


def test_import_refuses_an_unknown_class_name(tmp_path):
    root = _rf_export(tmp_path, ["person", "helmet"], {"boot1__0": ["1 0.5 0.5 0.1 0.1"]})
    # (and `goggles`, dropped from the project, is now unknown too -- see test_import_refuses_the_dropped_goggles_class)
    r = _run("import_roboflow.py", str(root), "--out", str(tmp_path / "lab"))
    assert r.returncode != 0 and "helmet" in (r.stdout + r.stderr)


def test_import_ignores_roboflow_splits_and_groups_by_session(tmp_path):
    root = _rf_export(tmp_path, ["person"], {f"bootA__{i}": ["0 0.5 0.5 0.2 0.2"] for i in range(6)} |
                      {f"bootB__{i}": ["0 0.5 0.5 0.2 0.2"] for i in range(3)})
    out = tmp_path / "lab"
    assert _run("import_roboflow.py", str(root), "--out", str(out)).returncode == 0
    assert len(list((out / "bootA" / "images").iterdir())) == 6 and len(list((out / "bootB" / "images").iterdir())) == 3


def test_import_converts_polygons_and_merges_coat_panels(tmp_path):
    root = _rf_export(tmp_path, ["person", "lab_coat"],
                      {"s1__0": ["0 0.5 0.5 0.6 0.9",
                                 "1 0.30 0.40 0.38 0.40 0.38 0.65 0.30 0.65",          # polygon: left panel
                                 "1 0.62 0.40 0.72 0.40 0.72 0.65 0.62 0.65"]})        # polygon: right panel
    out = tmp_path / "lab"
    assert _run("import_roboflow.py", str(root), "--out", str(out), "--merge-coats").returncode == 0
    rows = [l.split() for l in (out / "s1" / "labels" / "s1__0.txt").read_text().splitlines()]
    coats = [r for r in rows if r[0] == "1"]
    assert len(coats) == 1 and float(coats[0][3]) > 0.4          # one box spanning both panels


def test_pilot_holdout_puts_the_last_block_in_test_with_a_gap(tmp_path):
    lab = tmp_path / "labeled"
    _session(lab, "T_01_a", {f"T_01_a__{i:03d}": "0 0.5 0.5 0.1 0.1\n" for i in range(40)})
    out = tmp_path / "ds"
    r = _run("build_lab_dataset.py", "--labeled", str(lab), "--plan", str(_plan(tmp_path)), "--out", str(out),
             "--pilot-time-holdout", "0.2", "--test", "")
    assert r.returncode == 0 and "PILOT SPLIT" in r.stdout, r.stdout + r.stderr
    train = sorted(p.stem for p in (out / "images" / "train").iterdir())
    test = sorted(p.stem for p in (out / "images" / "test").iterdir())
    assert train and test and train[-1] < test[0]
    assert int(test[0].split("__")[-1]) - int(train[-1].split("__")[-1]) > 1      # a gap, not adjacent frames


def test_import_refuses_the_dropped_goggles_class(tmp_path):
    root = _rf_export(tmp_path, ["person", "goggles"], {"boot1__0": ["1 0.5 0.5 0.1 0.1"]})
    r = _run("import_roboflow.py", str(root), "--out", str(tmp_path / "lab"))
    assert r.returncode != 0 and "goggles" in (r.stdout + r.stderr)


def test_import_refuses_gloves_unless_it_is_explicitly_dropped(tmp_path):
    root = _rf_export(tmp_path, ["gloves", "lab_coat", "person"],
                      {"boot1__0": ["2 0.5 0.5 0.4 0.8", "0 0.5 0.3 0.1 0.05", "1 0.4 0.4 0.2 0.3"]})
    assert _run("import_roboflow.py", str(root), "--out", str(tmp_path / "a")).returncode != 0
    r = _run("import_roboflow.py", str(root), "--out", str(tmp_path / "lab"), "--drop-classes", "gloves")
    assert r.returncode == 0, r.stdout + r.stderr
    rows = [l.split()[0] for l in (tmp_path / "lab" / "boot1" / "labels" / "boot1__0.txt").read_text().splitlines()]
    assert sorted(rows) == ["0", "1"]            # gloves box gone; person -> 0, lab_coat -> 1
    assert "1 box(es)" in r.stdout
