"""Lecture des annotations Pascal VOC de RDD2022 et conversion en lignes YOLO.

Fonctions pures (sans effet de bord) : elles sont testées unitairement.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class VocObject:
    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class VocAnnotation:
    filename: str
    width: int  # 0 si absent ou invalide dans le XML
    height: int
    objects: list[VocObject] = field(default_factory=list)


@dataclass
class ConversionStats:
    kept: Counter = field(default_factory=Counter)  # classe -> nb de boîtes gardées
    dropped: Counter = field(default_factory=Counter)  # raison -> nb de boîtes ignorées
    unknown_labels: Counter = field(default_factory=Counter)  # étiquette hors classes -> nb

    def update(self, other: ConversionStats) -> None:
        self.kept.update(other.kept)
        self.dropped.update(other.dropped)
        self.unknown_labels.update(other.unknown_labels)


def _num(node: ET.Element | None, default: float | None = None) -> float | None:
    if node is None or node.text is None or not node.text.strip():
        return default
    try:
        return float(node.text.strip())
    except ValueError:
        return default


def parse_voc(xml_path: str | Path) -> VocAnnotation:
    """Parse un fichier VOC. Les objets sans boîte complète sont ignorés."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(_num(size.find("width"), 0) or 0) if size is not None else 0
    height = int(_num(size.find("height"), 0) or 0) if size is not None else 0
    filename = (root.findtext("filename") or "").strip()

    objects = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if box is None:
            continue
        coords = [_num(box.find(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        if any(c is None for c in coords):
            continue
        objects.append(VocObject(name, *coords))  # type: ignore[arg-type]
    return VocAnnotation(filename, width, height, objects)


def voc_box_to_yolo(
    xmin: float, ymin: float, xmax: float, ymax: float, width: int, height: int
) -> tuple[float, float, float, float]:
    """Boîte VOC (pixels, coins) -> YOLO (normalisé, centre + taille). Suppose une boîte valide."""
    return (
        (xmin + xmax) / 2 / width,
        (ymin + ymax) / 2 / height,
        (xmax - xmin) / width,
        (ymax - ymin) / height,
    )


def annotation_to_yolo_lines(
    ann: VocAnnotation,
    classes: list[str],
    aliases: dict[str, str] | None = None,
    min_box_px: float = 2.0,
    width: int | None = None,
    height: int | None = None,
) -> tuple[list[str], ConversionStats]:
    """Convertit une annotation en lignes YOLO 'cls cx cy w h'.

    - les étiquettes hors `classes` (après alias) sont ignorées et comptées ;
    - les boîtes sont rognées à l'image ; dégénérées / inversées / trop petites -> ignorées ;
    - les doublons exacts sont supprimés.
    `width`/`height` remplacent la taille du XML (utile quand elle vaut 0).
    """
    aliases = aliases or {}
    w = width or ann.width
    h = height or ann.height
    if w <= 0 or h <= 0:
        raise ValueError(f"Taille d'image invalide pour {ann.filename!r}: {w}x{h}")
    class_to_id = {c: i for i, c in enumerate(classes)}
    stats = ConversionStats()
    lines: list[str] = []
    seen: set[str] = set()

    for o in ann.objects:
        name = aliases.get(o.name, o.name)
        if name not in class_to_id:
            stats.unknown_labels[o.name] += 1
            continue
        if o.xmax <= o.xmin or o.ymax <= o.ymin:
            stats.dropped["inverted_or_empty"] += 1
            continue
        x0, x1 = max(0.0, o.xmin), min(float(w), o.xmax)
        y0, y1 = max(0.0, o.ymin), min(float(h), o.ymax)
        if x1 <= x0 or y1 <= y0:
            stats.dropped["outside_image"] += 1
            continue
        if (x1 - x0) < min_box_px or (y1 - y0) < min_box_px:
            stats.dropped["too_small"] += 1
            continue
        cx, cy, bw, bh = voc_box_to_yolo(x0, y0, x1, y1, w, h)
        line = f"{class_to_id[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        if line in seen:
            stats.dropped["duplicate"] += 1
            continue
        seen.add(line)
        lines.append(line)
        stats.kept[name] += 1
    return lines, stats
