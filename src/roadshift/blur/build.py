"""Construit les jeux de test floutés (Q2) à partir du dataset converti, sans toucher aux originaux.

Usage :
    python -m roadshift.blur.build --config configs/default.yaml --data /tmp/rdd_yolo [--workers 4]

Pour chaque jeu d'évaluation S (ex. source_test) et chaque longueur L > 0 de la config :
    blur/L{L}/images/<Pays>/<stem>.jpg     image floutée (angle tiré par image) + bruit
    wiener/L{L}/images/<Pays>/<stem>.jpg   la même, défloutée par Wiener avec le noyau exact
    .../labels/<Pays>/<stem>.txt           labels copiés (le flou ne déplace pas les objets)
    list_blur_L{L}_{S}.txt, list_wiener_L{L}_{S}.txt, data_blur_L{L}_{S}.yaml, data_wiener_L{L}_{S}.yaml
L = 0 : list_blur_L0_{S}.txt = copie de list_{S}.txt (images d'origine).
blur_manifest.csv : angle et temps de Wiener (ms) par image ; blur_info.json : config et durées.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from roadshift.blur.ops import blur_image, image_rng, motion_kernel, wiener_deconv


def image_angle(stem: str, bcfg: dict) -> float:
    """Angle de bougé d'une image, tiré une fois pour toutes (indépendant de la longueur)."""
    return float(image_rng(stem, bcfg["seed"]).choice(bcfg["angles_deg"]))


def label_paths(data: Path, img_rel: str) -> tuple[Path, Path]:
    """'./blur/L15/images/Pays/stem.jpg' -> (label d'origine, label à écrire à côté de l'image floutée)."""
    kind, lvl, rest = img_rel[2:].split("/", 2)  # rest = images/<Pays>/<stem>.jpg
    src = (data / rest.replace("images/", "labels/", 1)).with_suffix(".txt")
    dst = (data / kind / lvl / rest.replace("images/", "labels/", 1)).with_suffix(".txt")
    return src, dst


def _process(job: tuple) -> dict:
    """Une image, une longueur : flou + Wiener. Fonction de module (sérialisable pour les processus)."""
    src, rel, stem, length, cfg, out_root = job
    angle = image_angle(stem, cfg)  # même direction de bougé pour une image à toutes les longueurs
    rng = image_rng(f"{stem}|L{length}", cfg["seed"])  # bruit : propre à (image, longueur)
    k = motion_kernel(length, angle)
    img = np.asarray(Image.open(src).convert("RGB"))
    blurred = blur_image(img, k, cfg["noise_sigma"], rng)
    t0 = time.perf_counter()
    restored = wiener_deconv(blurred, k, cfg["wiener_k"])
    wiener_ms = (time.perf_counter() - t0) * 1000
    for kind, arr in (("blur", blurred), ("wiener", restored)):
        dst = Path(out_root) / kind / f"L{length}" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(dst.with_suffix(".jpg"), quality=cfg["jpeg_quality"])
    return {"stem": stem, "length": length, "angle": angle, "wiener_ms": round(wiener_ms, 2)}


def _write_yaml(root: Path, tag: str, classes: list[str]) -> None:
    d = {"train": "list_source_train.txt", "val": f"list_{tag}.txt", "names": dict(enumerate(classes))}
    (root / f"data_{tag}.yaml").write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8", newline="\n")


def build(cfg: dict, data: Path, workers: int = 4) -> dict:
    bcfg, classes = cfg["blur"], cfg["data"]["classes"]
    t_start = time.time()
    jobs, lists = [], {}
    for s in bcfg["eval_sets"]:
        lines = (data / f"list_{s}.txt").read_text().split()
        (data / f"list_blur_L0_{s}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        _write_yaml(data, f"blur_L0_{s}", classes)
        for length in [x for x in bcfg["kernel_lengths"] if x > 0]:
            for kind in ("blur", "wiener"):
                tag = f"{kind}_L{length}_{s}"
                out = [f"./{kind}/L{length}/{ln[2:].rsplit('.', 1)[0]}.jpg" for ln in lines]
                lists[tag] = out
            for ln in lines:
                rel = ln[2:]  # images/<Pays>/<stem>.<ext>
                jobs.append((str((data / rel).resolve()), rel, Path(rel).stem, length, bcfg, str(data)))
    # labels : copiés une fois par longueur et par type (Ultralytics cherche .../labels/ à côté de .../images/)
    for tag, out in lists.items():
        for img_rel in out:
            src, dst = label_paths(data, img_rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy(src, dst)
        (data / f"list_{tag}.txt").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
        _write_yaml(data, tag, classes)
    # spawn : sûr même si le processus parent a déjà des threads (numpy, torch), et identique sous Windows
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        rows = list(ex.map(_process, jobs, chunksize=16))
    rows.sort(key=lambda r: (r["length"], r["stem"]))
    with open(data / "blur_manifest.csv", "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    wiener = np.array([r["wiener_ms"] for r in rows])
    info = {
        "n_images_processed": len(rows),
        "lists": sorted(lists) + [f"blur_L0_{s}" for s in bcfg["eval_sets"]],
        "wiener_ms_median": float(np.median(wiener)),
        "wiener_ms_p90": float(np.quantile(wiener, 0.9)),
        "build_minutes": round((time.time() - t_start) / 60, 2),
        "config": bcfg,
    }
    (data / "blur_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lengths", type=int, nargs="+", help="remplace blur.kernel_lengths (ex. test rapide : 0 15)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.lengths:
        cfg["blur"]["kernel_lengths"] = args.lengths
    info = build(cfg, args.data, args.workers)
    print(json.dumps({k: v for k, v in info.items() if k != "config"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
