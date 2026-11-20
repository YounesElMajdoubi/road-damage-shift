import csv
import json
from collections import defaultdict

import pytest
import yaml
from PIL import Image

from roadshift.data.convert import MARKER, discover_countries, run


def _read_list(root, name):
    return (root / name).read_text().split()


def test_discovery_handles_nested_extraction(raw_root):
    found = discover_countries(raw_root, ["Japan", "Czech", "United_States", "Norway"])
    assert set(found) == {"Japan", "Czech", "United_States"}
    assert found["Czech"][0].xml_dir.parts[-4] == "Czech"


def test_end_to_end(raw_root, tmp_path, cfg):
    out = tmp_path / "out"
    res = run(cfg, raw_root, out)

    # Sous-échantillonnage : Czech plafonné à 30 ; Norway absent mais optionnel.
    rows = list(csv.DictReader((out / "manifest.csv").open()))
    per_country = defaultdict(int)
    for r in rows:
        per_country[r["country"]] += 1
    assert per_country == {"Japan": 60, "Czech": 30, "United_States": 45}

    # Un label par image, contenu cohérent (image 1 : D00 gardé, D43 ignoré).
    for r in rows:
        assert (out / r["label"]).exists() and (out / r["image"]).exists()
    assert (out / "labels/Japan/Japan_000001.txt").read_text() == "0 0.300000 0.312500 0.400000 0.375000\n"
    assert (out / "labels/Japan/Japan_000004.txt").read_text() == ""  # image sans dégât
    total_unknown = sum(v.get("D43", 0) for v in res["stats"]["unknown_labels"].values())
    assert total_unknown == sum(int(r["stem"].split("_")[-1]) % 4 == 1 for r in rows)  # un D43 par image i%4==1

    # Image 3 n'a pas de <size> : taille lue depuis l'image.
    j3 = next(r for r in rows if r["stem"] == "Japan_000003")
    assert (j3["width"], j3["height"]) == ("100", "80")

    # Redimensionnement : 300x240 > max_side=200 -> 200x160, labels normalisés inchangés.
    us = [r for r in rows if r["country"] == "United_States"]
    assert all(r["resized"] == "1" and (r["out_width"], r["out_height"]) == ("200", "160") for r in us)
    with Image.open(out / us[0]["image"]) as im:
        assert im.size == (200, 160)
    # Petite image : lien symbolique, ou copie si l'OS refuse (Windows sans mode développeur).
    small = out / "images/Japan/Japan_000001.jpg"
    assert small.is_symlink() or small.read_bytes() == (raw_root / "Japan/train/images/Japan_000001.jpg").read_bytes()
    # Fins de ligne Unix quel que soit l'OS : fichiers identiques Windows/Linux.
    assert b"\r" not in (out / "labels/Japan/Japan_000001.txt").read_bytes()
    assert b"\r" not in (out / "manifest.csv").read_bytes()

    # Splits : disjoints, cible = 20 test + 25 pool.
    src = {s: set(_read_list(out, f"list_source_{s}.txt")) for s in ("train", "val", "test")}
    assert not (src["train"] & src["val"]) and not (src["train"] & src["test"]) and not (src["val"] & src["test"])
    assert sum(map(len, src.values())) == 90
    us_test = set(_read_list(out, "list_test_United_States.txt"))
    us_pool = set(_read_list(out, "list_pool_United_States.txt"))
    assert len(us_test) == 20 and len(us_pool) == 25 and not (us_test & us_pool)

    # Few-shot imbriqué et inclus dans le pool.
    k5, k10 = (set(_read_list(out, f"list_fewshot_United_States_k{k}_s0.txt")) for k in (5, 10))
    assert k5 <= k10 <= us_pool and len(k5) == 5

    # Les YAML Ultralytics pointent vers des listes existantes dont les chemins résolvent.
    d = yaml.safe_load((out / "data_source.yaml").read_text())
    assert d["names"] == {0: "D00", 1: "D10", 2: "D20", 3: "D40"} and "path" not in d
    for key in ("train", "val", "test"):
        for line in _read_list(out, d[key]):
            assert line.startswith("./") and (out / line[2:]).exists()
    assert (out / "data_eval_United_States.yaml").exists()

    # Liste de contrôle "_pos" : uniquement les images de test avec >= 1 boîte.
    def n_boxes(line):
        return len((out / line[2:].replace("images/", "labels/", 1)).with_suffix(".txt").read_text().split("\n")) - 1

    jp_test, jp_pos = _read_list(out, "list_test_Japan.txt"), _read_list(out, "list_test_Japan_pos.txt")
    assert set(jp_pos) == {ln for ln in jp_test if n_boxes(ln) > 0} and len(jp_pos) < len(jp_test)
    assert (out / "data_eval_Japan_pos.yaml").exists()

    info = json.loads((out / "dataset_info.json").read_text())
    assert len(info["manifest_sha256"]) == 64 and info["n_images"] == 135


def test_deterministic_and_overwrite_guard(raw_root, tmp_path, cfg):
    out = tmp_path / "out"
    h1 = run(cfg, raw_root, out)["info"]["manifest_sha256"]
    with pytest.raises(FileExistsError):
        run(cfg, raw_root, out)
    h2 = run(cfg, raw_root, out, overwrite=True)["info"]["manifest_sha256"]
    assert h1 == h2
    cfg["project"]["seed"] = 7
    assert run(cfg, raw_root, out, overwrite=True)["info"]["manifest_sha256"] != h1


def test_refuses_to_delete_foreign_directory(raw_root, tmp_path, cfg):
    foreign = tmp_path / "precious"
    foreign.mkdir()
    (foreign / "thesis.docx").write_text("x")
    with pytest.raises(FileExistsError, match=MARKER):
        run(cfg, raw_root, foreign, overwrite=True)
    assert (foreign / "thesis.docx").exists()


def test_missing_required_country_raises(raw_root, tmp_path, cfg):
    cfg["data"]["countries"]["Norway"]["required"] = True
    with pytest.raises(FileNotFoundError, match="Norway"):
        run(cfg, raw_root, tmp_path / "out")


def test_background_subsampling_only_affects_source_train(raw_root, tmp_path, cfg):
    cfg["split"]["train_background_keep"] = 0.0
    out = tmp_path / "out"
    run(cfg, raw_root, out)

    def label_text(line):
        return (out / line[2:].replace("images/", "labels/", 1)).with_suffix(".txt").read_text()

    assert all(label_text(ln).strip() for ln in _read_list(out, "list_source_train.txt"))
    assert any(label_text(ln) == "" for ln in _read_list(out, "list_source_val.txt"))


def test_empty_split_raises_before_conversion(raw_root, tmp_path, cfg):
    cfg["split"]["block_size"] = 50  # Japan (60 images) -> 2 blocs : impossible de remplir 3 splits
    with pytest.raises(ValueError, match="vide"):
        run(cfg, raw_root, tmp_path / "out")


def test_pool_too_small_raises(raw_root, tmp_path, cfg):
    cfg["fewshot"]["sizes"] = [5, 100]
    with pytest.raises(ValueError, match="few-shot"):
        run(cfg, raw_root, tmp_path / "out")
