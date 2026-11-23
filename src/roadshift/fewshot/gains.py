"""Gains appariés de chaque fine-tuning few-shot par rapport à k = 0 (même modèle S1, mêmes images de test).

Usage (sur la sortie s2_outputs.zip décompressée) :
    python -m roadshift.fewshot.gains --mlflow-dir <dossier contenant mlflow.db et mlruns> \
        --data <dataset_meta> --model yolo11s --out results/fewshot_gains.csv

Pour chaque run k > 0 et chaque éval (test cible, test source) : différence (run - k0) en mAP@50 et F1,
IC à 95 % par bootstrap apparié (mêmes images rééchantillonnées pour les deux modèles).
Chaque modèle garde son seuil de F1 choisi sur sa propre validation source.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from roadshift.eval.metrics import (
    DetectionTable,
    best_f1_threshold,
    load_ground_truth,
    load_predictions,
    load_sizes,
    match,
    paired_bootstrap_diff,
    read_image_list,
)

NC = 4


def gains_vs_k0(
    runs: dict[str, dict], tables: dict[tuple[str, str], DetectionTable], confs: dict[str, float],
    evals: list[str], n_boot: int = 1000, seed: int = 0,
) -> list[dict]:  # fmt: skip
    """runs : nom -> {"k": int, "seed": int | None} ; tables : (nom, éval) -> table ; confs : nom -> seuil F1."""
    base = next(n for n, r in runs.items() if r["k"] == 0)
    rows = []
    for name, r in sorted(runs.items(), key=lambda x: (x[1]["k"], str(x[1]["seed"]))):
        if name == base:
            continue
        for ev in evals:
            for met in ("mAP50", "F1"):
                p, lo, hi = paired_bootstrap_diff(
                    tables[(name, ev)], tables[(base, ev)], confs[name], confs[base], met, n_boot, seed
                )
                rows.append({"run": name, "k": r["k"], "seed": r["seed"], "eval": ev, "metric": met,
                             "gain_vs_k0": round(p, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                             "significant": lo > 0 or hi < 0})  # fmt: skip
    return rows


def load_from_mlflow(mlflow_dir: Path, model: str) -> dict[str, dict]:
    """Retrouve, pour chaque run few-shot du modèle, k, tirage et dossier d'artefacts de prédictions."""
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{(mlflow_dir / 'mlflow.db').resolve()}")
    df = mlflow.search_runs(search_all_experiments=True)
    df = df[(df["params.model"] == model) & df["params.k"].notna()]
    out = {}
    for _, r in df.iterrows():
        art = Path(mlflow_dir) / "mlruns" / r["experiment_id"] / r["run_id"] / "artifacts" / "predictions"
        seed = r["params.subset_seed"]
        out[r["tags.mlflow.runName"]] = {
            "k": int(r["params.k"]),
            "seed": None if seed in (None, "None") else int(seed),
            "pred_dir": art,
        }
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mlflow-dir", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--model", default="yolo11s")
    ap.add_argument("--evals", nargs="+", default=["test_United_States", "source_test"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)

    runs = load_from_mlflow(args.mlflow_dir, args.model)
    sizes = load_sizes(args.data / "manifest.csv")
    tables, confs = {}, {}
    for ev in ["source_val", *args.evals]:
        stems = read_image_list(args.data / f"list_{ev}.txt")
        gt = load_ground_truth(stems, args.data / "labels", sizes)
        for name, r in runs.items():
            tables[(name, ev)] = match(stems, gt, load_predictions(r["pred_dir"] / ev / "predictions.json"), NC)
    for name in runs:
        confs[name] = best_f1_threshold(tables[(name, "source_val")])
    rows = gains_vs_k0(runs, tables, confs, args.evals, args.n_boot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        star = "  *" if r["significant"] else ""
        print(
            f"{r['run']:38s} {r['eval']:20s} {r['metric']:5s} {r['gain_vs_k0']:+.3f} "
            f"[{r['lo']:+.3f}, {r['hi']:+.3f}]{star}"
        )


if __name__ == "__main__":
    main()
