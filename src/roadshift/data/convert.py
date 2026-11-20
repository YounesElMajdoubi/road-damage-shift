"""RDD2022 (Pascal VOC) -> format YOLO, avec split par pays et sous-échantillonnage.

Usage :
    python -m roadshift.data.convert --config configs/default.yaml \
        [--raw-root /kaggle/input/...] [--out /kaggle/temp/rdd_yolo] [--overwrite]

Sortie (dans --out) :
    images/<Pays>/<stem>.jpg        lien symbolique, copie, ou copie redimensionnée
    labels/<Pays>/<stem>.txt        une ligne YOLO par boîte (fichier vide = image sans dégât)
    list_*.txt                      listes d'images par split (chemins relatifs "./images/...")
    data_source.yaml                train/val/test sources pour Ultralytics
    data_eval_<Pays>.yaml           évaluation sur le test d'un pays
    manifest.csv                    une ligne par image (pays, split, tailles, nb de boîtes)
    stats.json                      comptes par pays/split/classe + boîtes ignorées
    dataset_info.json               config utilisée + empreinte sha256 du manifest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image

from roadshift.data.splits import (
    natural_key,
    nested_fewshot,
    shuffled_blocks,
    split_blocks,
    subsample_backgrounds,
    take_blocks,
)
from roadshift.data.voc import ConversionStats, annotation_to_yolo_lines, parse_voc

log = logging.getLogger("roadshift.convert")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MARKER = ".roadshift_dataset"


# --------------------------------------------------------------------------- découverte
@dataclass(frozen=True)
class CountrySource:
    country: str
    xml_dir: Path
    img_dir: Path


def discover_countries(raw_root: Path, countries: list[str]) -> dict[str, list[CountrySource]]:
    """Trouve les dossiers `.../<Pays>/.../train/annotations/xmls` sous raw_root.

    Indépendant de la façon dont l'archive a été extraite (miroir Kaggle, figshare...).
    Le pays est le premier composant du chemin qui correspond à un nom de la config.
    """
    lower = {c.lower(): c for c in countries}
    found: dict[str, list[CountrySource]] = defaultdict(list)
    for dirpath, dirnames, _ in os.walk(raw_root, followlinks=True):
        p = Path(dirpath)
        if p.name != "xmls" or p.parent.name != "annotations":
            continue
        train_dir = p.parent.parent
        img_dir = train_dir / "images"
        if not img_dir.is_dir():
            log.warning("xmls sans dossier images voisin : %s", p)
            continue
        rel = p.relative_to(raw_root).parts
        country = next((lower[part.lower()] for part in rel if part.lower() in lower), None)
        if country is None:
            log.info("Dossier ignoré (pays absent de la config) : %s", p)
            continue
        found[country].append(CountrySource(country, p, img_dir))
        dirnames.clear()  # inutile de descendre plus bas
    return dict(found)


def index_images(img_dir: Path) -> dict[str, Path]:
    return {f.stem: f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS}


# --------------------------------------------------------------------------- conversion
@dataclass
class ImageResult:
    country: str
    stem: str
    split: str
    src: Path
    dst: Path
    width: int
    height: int
    out_width: int
    out_height: int
    resized: bool
    n_boxes: int
    per_class: Counter
    stats: ConversionStats


def _place_image(src: Path, dst: Path, w: int, h: int, max_side: int | None, mode: str, quality: int):
    """Lien/copie, ou copie redimensionnée si le grand côté dépasse max_side."""
    if max_side and max(w, h) > max_side:
        scale = max_side / max(w, h)
        new = (max(1, round(w * scale)), max(1, round(h * scale)))
        with Image.open(src) as im:
            im = im.convert("RGB").resize(new, Image.Resampling.LANCZOS)
            dst = dst.with_suffix(".jpg")
            im.save(dst, quality=quality)
        return dst, new[0], new[1], True
    if mode == "symlink":
        try:
            os.symlink(src.resolve(), dst)
            return dst, w, h, False
        except OSError:  # ex. Windows sans droits : on copie
            pass
    shutil.copy2(src, dst)
    return dst, w, h, False


def convert_one(
    country: str,
    stem: str,
    split: str,
    xml_path: Path,
    img_path: Path,
    out_root: Path,
    cfg_data: dict,
) -> ImageResult:
    ann = parse_voc(xml_path)
    w, h = ann.width, ann.height
    if w <= 0 or h <= 0:  # taille absente du XML : on lit l'en-tête de l'image
        with Image.open(img_path) as im:
            w, h = im.size
    lines, stats = annotation_to_yolo_lines(
        ann,
        cfg_data["classes"],
        cfg_data.get("label_aliases") or {},
        cfg_data.get("min_box_px", 2),
        width=w,
        height=h,
    )
    img_dst = out_root / "images" / country / f"{stem}{img_path.suffix.lower()}"
    lbl_dst = out_root / "labels" / country / f"{stem}.txt"
    img_dst, ow, oh, resized = _place_image(
        img_path,
        img_dst,
        w,
        h,
        cfg_data.get("max_side"),
        cfg_data.get("image_mode", "symlink"),
        cfg_data.get("jpeg_quality", 95),
    )
    lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
    return ImageResult(
        country, stem, split, img_path, img_dst, w, h, ow, oh, resized, len(lines), stats.kept.copy(), stats
    )


# --------------------------------------------------------------------------- sorties
def _write_list(out_root: Path, name: str, results: list[ImageResult]) -> str:
    fname = f"list_{name}.txt"
    rels = sorted((f"./{r.dst.relative_to(out_root).as_posix()}" for r in results), key=natural_key)
    (out_root / fname).write_text("\n".join(rels) + "\n", encoding="utf-8", newline="\n")
    return fname


def _write_data_yaml(out_root: Path, name: str, train: str, val: str, test: str | None, classes: list[str]):
    # Pas de clé `path` : Ultralytics prend alors le dossier du YAML -> dataset déplaçable.
    d = {"train": train, "val": val, "names": dict(enumerate(classes))}
    if test:
        d["test"] = test
    (out_root / f"data_{name}.yaml").write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8", newline="\n")


MANIFEST_COLS = [
    "country", "stem", "split", "image", "label", "width", "height",
    "out_width", "out_height", "resized", "n_boxes",
]  # fmt: skip


def _manifest_csv(results: list[ImageResult], out_root: Path, classes: list[str]) -> str:
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator="\n")
    wr.writerow(MANIFEST_COLS + [f"n_{c}" for c in classes])
    for r in sorted(results, key=lambda r: (r.country, natural_key(r.stem))):
        wr.writerow(
            [r.country, r.stem, r.split, r.dst.relative_to(out_root).as_posix(),
             f"labels/{r.country}/{r.stem}.txt", r.width, r.height, r.out_width, r.out_height,
             int(r.resized), r.n_boxes] + [r.per_class.get(c, 0) for c in classes]
        )  # fmt: skip
    return buf.getvalue()


def prepare_output(out_root: Path, overwrite: bool) -> None:
    if out_root.exists() and any(out_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out_root} existe et n'est pas vide (utiliser --overwrite)")
        if not (out_root / MARKER).exists():
            raise FileExistsError(f"{out_root} ne vient pas de ce script ({MARKER} absent) : refus de supprimer")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / MARKER).write_text("generated by roadshift.data.convert\n", encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- pipeline
def run(cfg: dict, raw_root: Path, out_root: Path, overwrite: bool = False) -> dict:
    data_cfg, split_cfg = cfg["data"], cfg["split"]
    seed = int(cfg["project"]["seed"])
    classes = list(data_cfg["classes"])
    countries_cfg: dict = data_cfg["countries"]

    prepare_output(out_root, overwrite)
    sources = discover_countries(raw_root, list(countries_cfg))

    jobs: list[tuple] = []
    missing_images: Counter = Counter()
    for country, ccfg in countries_cfg.items():
        role = ccfg["role"]
        if role not in {"source", "target"}:
            raise ValueError(f"{country}: role inconnu {role!r}")
        srcs = sources.get(country, [])
        if not srcs:
            if ccfg.get("required", True):
                raise FileNotFoundError(f"{country}: aucun dossier train/annotations/xmls sous {raw_root}")
            log.warning("%s introuvable, ignoré (required: false)", country)
            continue
        if len(srcs) > 1:
            raise RuntimeError(f"{country}: plusieurs dossiers trouvés {[str(s.xml_dir) for s in srcs]}")
        src = srcs[0]
        imgs = index_images(src.img_dir)
        stems = []
        for x in src.xml_dir.glob("*.xml"):
            if x.stem in imgs:
                stems.append(x.stem)
            else:
                missing_images[country] += 1
        blocks = take_blocks(shuffled_blocks(stems, split_cfg["block_size"], seed), ccfg.get("max_images"))
        assignment = split_blocks(blocks, split_cfg[role])
        empty = [k for k, v in assignment.items() if not v]
        if empty:
            raise ValueError(f"{country}: split(s) vide(s) {empty} — réduire split.block_size ou augmenter max_images")
        max_k = max((cfg.get("fewshot") or {}).get("sizes") or [0])
        if role == "target" and len(assignment.get("pool", [])) < max_k:
            raise ValueError(
                f"{country}: pool de {len(assignment.get('pool', []))} images < few-shot max {max_k} "
                f"(augmenter max_images ou réduire split.target.test)"
            )
        (out_root / "images" / country).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / country).mkdir(parents=True, exist_ok=True)
        for split, split_stems in assignment.items():
            for s in split_stems:
                jobs.append((country, s, split, src.xml_dir / f"{s}.xml", imgs[s], out_root, data_cfg))
        log.info("%s (%s) : %d images -> %s", country, role, sum(map(len, assignment.values())),
                 {k: len(v) for k, v in assignment.items()})  # fmt: skip

    with ThreadPoolExecutor(max_workers=int(data_cfg.get("workers", 8))) as ex:
        results = list(ex.map(lambda j: convert_one(*j), jobs))

    by = defaultdict(list)  # (country, split) -> résultats
    for r in results:
        by[(r.country, r.split)].append(r)
    roles = {c: countries_cfg[c]["role"] for c, _ in by}

    # --- listes et YAML
    lists: dict[str, str] = {}
    src_countries = sorted({c for c in roles if roles[c] == "source"})
    tgt_countries = sorted({c for c in roles if roles[c] == "target"})
    for split in ("train", "val", "test"):
        rs = [r for c in src_countries for r in by[(c, split)]]
        if split == "train":
            bg = {r.stem: r.n_boxes == 0 for r in rs}
            keep_frac = split_cfg.get("train_background_keep", 1.0)
            keep = set(subsample_backgrounds([r.stem for r in rs], bg, keep_frac, seed))
            rs = [r for r in rs if r.stem in keep]
        lists[f"source_{split}"] = _write_list(out_root, f"source_{split}", rs)
    for c in src_countries + tgt_countries:
        lists[f"test_{c}"] = _write_list(out_root, f"test_{c}", by[(c, "test")])
    fewshot_cfg = cfg.get("fewshot") or {}
    for c in tgt_countries:
        pool = by[(c, "pool")]
        lists[f"pool_{c}"] = _write_list(out_root, f"pool_{c}", pool)
        pool_by_stem = {r.stem: r for r in pool}
        for s in fewshot_cfg.get("seeds", []):
            for k, sub in nested_fewshot(list(pool_by_stem), fewshot_cfg.get("sizes", []), s).items():
                name = f"fewshot_{c}_k{k}_s{s}"
                lists[name] = _write_list(out_root, name, [pool_by_stem[x] for x in sub])

    _write_data_yaml(out_root, "source", lists["source_train"], lists["source_val"], lists["source_test"], classes)
    for c in src_countries + tgt_countries:
        _write_data_yaml(out_root, f"eval_{c}", lists["source_train"], lists[f"test_{c}"], None, classes)

    # --- manifest, stats, empreinte
    manifest = _manifest_csv(results, out_root, classes)
    (out_root / "manifest.csv").write_text(manifest, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(manifest.encode()).hexdigest()

    stats: dict = {
        "per_country_split": {},
        "dropped_boxes": {},
        "unknown_labels": {},
        "missing_images": dict(missing_images),
    }
    for (c, split), rs in sorted(by.items()):
        agg = ConversionStats()
        for r in rs:
            agg.update(r.stats)
        stats["per_country_split"][f"{c}/{split}"] = {
            "images": len(rs),
            "background_images": sum(r.n_boxes == 0 for r in rs),
            "resized_images": sum(r.resized for r in rs),
            "boxes": {k: agg.kept.get(k, 0) for k in classes},
        }
        stats["dropped_boxes"][f"{c}/{split}"] = dict(agg.dropped)
        stats["unknown_labels"][f"{c}/{split}"] = dict(agg.unknown_labels)
    stats["lists"] = {k: len((out_root / v).read_text().split()) for k, v in lists.items()}
    (out_root / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )

    info = {
        "manifest_sha256": digest,
        "n_images": len(results),
        "raw_root": str(raw_root),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {k: cfg[k] for k in ("project", "data", "split", "fewshot") if k in cfg},
    }
    (out_root / "dataset_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    log.info("Terminé : %d images, manifest sha256=%s", len(results), digest[:12])
    return {"stats": stats, "info": info, "lists": lists}


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8", newline="\n") as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--raw-root", help="remplace paths.raw_root")
    ap.add_argument("--out", help="remplace paths.processed_root")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    run(
        cfg,
        Path(args.raw_root or cfg["paths"]["raw_root"]),
        Path(args.out or cfg["paths"]["processed_root"]),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
