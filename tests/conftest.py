"""Fabrique un mini-RDD2022 synthétique (images + XML VOC) pour les tests."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from PIL import Image

BASE_CFG = {
    "project": {"seed": 42},
    "data": {
        "classes": ["D00", "D10", "D20", "D40"],
        "label_aliases": {},
        "min_box_px": 2,
        "max_side": 200,
        "image_mode": "symlink",
        "jpeg_quality": 90,
        "workers": 2,
        "countries": {
            "Japan": {"role": "source", "max_images": None},
            "Czech": {"role": "source", "max_images": 30},
            "United_States": {"role": "target", "max_images": None},
            "Norway": {"role": "target", "max_images": None, "required": False},
        },
    },
    "split": {
        "block_size": 5,
        "source": {"train": 0.7, "val": 0.15, "test": 0.15},
        "target": {"test": 20, "pool": "rest"},
        "train_background_keep": 1.0,
    },
    "fewshot": {"sizes": [5, 10], "seeds": [0, 1]},
}


def voc_xml(filename: str, w: int, h: int, objects: list[tuple], write_size: bool = True) -> str:
    size = f"<size><width>{w}</width><height>{h}</height><depth>3</depth></size>" if write_size else ""
    objs = "".join(
        f"<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin>"
        f"<xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"
        for n, a, b, c, d in objects
    )
    return f"<annotation><filename>{filename}</filename>{size}{objs}</annotation>"


def make_country(root: Path, country: str, n: int, w: int = 100, h: int = 80, prefix_dir: str | None = None):
    base = root / prefix_dir / country if prefix_dir else root / country
    img_dir = base / "train" / "images"
    xml_dir = base / "train" / "annotations" / "xmls"
    img_dir.mkdir(parents=True)
    xml_dir.mkdir(parents=True)
    (base / "test" / "images").mkdir(parents=True)  # le test officiel n'a pas d'annotations
    for i in range(1, n + 1):
        stem = f"{country}_{i:06d}"
        Image.new("RGB", (w, h), (i % 255, 100, 50)).save(img_dir / f"{stem}.jpg")
        if i % 4 == 0:
            objs = []  # image sans dégât
        elif i % 4 == 1:
            objs = [("D00", 10, 10, 50, 40), ("D43", 1, 1, 5, 5)]  # D43 hors classes
        elif i % 4 == 2:
            objs = [("D40", 20.5, 15.5, 60.5, 70.5), ("D10", -5, 0, 30, 20)]  # flottants + clipping
        else:
            objs = [("D20", 0, 0, w, h)]
        (xml_dir / f"{stem}.xml").write_text(voc_xml(f"{stem}.jpg", w, h, objs, write_size=(i != 3)))
    return base


@pytest.fixture
def cfg():
    return copy.deepcopy(BASE_CFG)


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    make_country(root, "Japan", 60)
    make_country(root, "Czech", 40, prefix_dir="RDD2022_Czech")  # extraction imbriquée
    make_country(root, "United_States", 45, w=300, h=240)  # > max_side -> redimensionnée
    return root
