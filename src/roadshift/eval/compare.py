"""Comparaisons avec IC bootstrap : modèle A - modèle B (apparié) et pays cible - test source (indépendant).

Usage :
    python -m roadshift.eval.compare --run yolo11s=<runs_dir> --run rtdetr_l=<runs_dir> --data <dataset_meta> \
        --target test_United_States --out results/comparisons_s1.csv

Chaque modèle garde son propre seuil de F1, choisi sur sa validation source (comme dans report.py).
Une différence est dite « significative » quand son IC à 95 % exclut 0. L'IC ne couvre que l'aléa
de l'échantillon de test : la variance d'entraînement (seed) n'est pas incluse.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

from roadshift.eval.metrics import best_f1_threshold, load_sizes, paired_bootstrap_diff, unpaired_bootstrap_diff
from roadshift.eval.report import build_table

METRICS = ("mAP50", "F1")


def compare(runs: dict[str, Path], data: Path, tags: list[str], target: str, n_boot: int = 1000, seed: int = 0):
    sizes = load_sizes(data / "manifest.csv")
    conf = {m: best_f1_threshold(build_table(d, data, "source_val", sizes)) for m, d in runs.items()}
    tables = {(m, t): build_table(d, data, t, sizes) for m, d in runs.items() for t in tags}
    rows = []

    def add(kind, ev, metric, a, b, res):
        p, lo, hi = res
        rows.append({"comparison": kind, "eval": ev, "metric": metric, "a": a, "b": b, "diff": round(p, 4),
                     "lo": round(lo, 4), "hi": round(hi, 4), "significant": lo > 0 or hi < 0})  # fmt: skip

    for ma, mb in combinations(runs, 2):
        for t in tags:
            for met in METRICS:
                r = paired_bootstrap_diff(tables[(ma, t)], tables[(mb, t)], conf[ma], conf[mb], met, n_boot, seed)
                add("model_paired", t, met, ma, mb, r)
    for m in runs:
        for met in METRICS:
            r = unpaired_bootstrap_diff(tables[(m, target)], tables[(m, "source_test")], conf[m], met, n_boot, seed)
            add("target_minus_source", f"{target} - source_test", met, m, m, r)
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="modèle=dossier_du_run (répéter)")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--target", default="test_United_States")
    ap.add_argument("--tags", nargs="+", default=["source_test", "test_Japan", "test_India", "test_Czech"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    runs = dict(r.split("=", 1) for r in args.run)
    runs = {k: Path(v) for k, v in runs.items()}
    tags = list(dict.fromkeys([*args.tags, args.target]))
    rows = compare(runs, args.data, tags, args.target, args.n_boot, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        star = "  *" if r["significant"] else ""
        print(f"{r['comparison']:20s} {r['eval']:36s} {r['metric']:5s} {r['a']}-{r['b']}: "
              f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]{star}")  # fmt: skip


if __name__ == "__main__":
    main()
