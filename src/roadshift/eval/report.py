"""Tableau de résultats avec IC bootstrap à partir des predictions.json d'un run.

Usage :
    python -m roadshift.eval.report --runs-dir <dossier du run Ultralytics> --data <dataset converti> \
        --model yolo11s --out results/ci_yolo11s.csv

--runs-dir contient eval_<tag>/predictions.json (tag = source_val, source_test, test_<Pays>[_pos]).
--data contient list_<tag>.txt, manifest.csv et labels/ (sortie du convertisseur, ou dataset_meta exporté).

Le seuil de confiance du F1 est choisi sur source_val (maximisation du F1 micro), puis appliqué tel quel
à tous les tests.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from roadshift.eval.metrics import (
    best_f1_threshold,
    bootstrap,
    compute,
    load_ground_truth,
    load_predictions,
    load_sizes,
    match,
    read_image_list,
    to_agnostic,
)

CLASSES = ["D00", "D10", "D20", "D40"]


def tag_to_list(tag: str) -> str:
    """source_val -> list_source_val.txt ; test_Japan_pos -> list_test_Japan_pos.txt."""
    return f"list_{tag}.txt"


AGNOSTIC_NMS_IOU = 0.7


def build_table(runs_dir: Path, data: Path, tag: str, sizes: dict, agnostic: bool = False):
    """agnostic=True : une seule classe « dégradation » (voir metrics.to_agnostic)."""
    stems = read_image_list(data / tag_to_list(tag))
    gt = load_ground_truth(stems, data / "labels", sizes)
    preds = load_predictions(runs_dir / f"eval_{tag}" / "predictions.json")
    unknown = set(preds) - set(stems)
    if unknown:
        raise ValueError(f"{tag}: {len(unknown)} images prédites absentes de la liste (ex. {sorted(unknown)[:3]})")
    if agnostic:
        gt, preds = to_agnostic(gt, preds, AGNOSTIC_NMS_IOU)
        return match(stems, gt, preds, 1)
    return match(stems, gt, preds, len(CLASSES))


def report(runs_dir: Path, data: Path, model: str, n_boot: int = 1000, seed: int = 0, agnostic: bool = False):
    sizes = load_sizes(data / "manifest.csv")
    tags = sorted(p.name.removeprefix("eval_") for p in runs_dir.glob("eval_*") if (p / "predictions.json").exists())
    if "source_val" not in tags:
        raise FileNotFoundError("eval_source_val/predictions.json requis pour fixer le seuil du F1")
    conf = best_f1_threshold(build_table(runs_dir, data, "source_val", sizes, agnostic))
    rows = []
    for tag in ["source_val", "source_test"] + [t for t in tags if t.startswith("test_")]:
        t = build_table(runs_dir, data, tag, sizes, agnostic)
        b = bootstrap(t, conf, n=n_boot, seed=seed)
        n_gt = t.n_gt.sum(0)
        row = {"model": model, "eval": tag, "agnostic": agnostic, "n_images": len(t.stems), "n_boxes": int(n_gt.sum()),
               "conf_F1": conf}  # fmt: skip
        for k in ("mAP50", "F1", "precision", "recall"):
            row[k], row[f"{k}_lo"], row[f"{k}_hi"] = (round(v, 4) for v in b[k])
        for i, c in enumerate([] if agnostic else CLASSES):
            row[f"n_{c}"] = int(n_gt[i])
            row[f"AP50_{c}"], row[f"AP50_{c}_lo"], row[f"AP50_{c}_hi"] = (round(v, 4) for v in b[f"AP50_{i}"])
        row["mAP50_ultralytics_method"] = round(compute(t, conf, method="ultralytics")["mAP50"], 4)
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agnostic", action="store_true", help="une seule classe « dégradation »")
    args = ap.parse_args(argv)
    rows = report(args.runs_dir, args.data, args.model, args.n_boot, args.seed, args.agnostic)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(
            f"{r['eval']:24s} n={r['n_images']:5d}  mAP50 {r['mAP50']:.3f} [{r['mAP50_lo']:.3f}, {r['mAP50_hi']:.3f}]"
            f"  F1@{r['conf_F1']:.2f} {r['F1']:.3f} [{r['F1_lo']:.3f}, {r['F1_hi']:.3f}]"
        )
    print(json.dumps({"out": str(args.out), "n_boot": args.n_boot, "seed": args.seed}))


if __name__ == "__main__":
    main()
