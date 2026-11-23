"""Gains appariés des fine-tunings few-shot (mêmes images de test, IC 95 % par bootstrap apparié).

Usage (sur une sortie s2_outputs.zip ou s2b_outputs.zip décompressée) :
    # chaque run k > 0 contre k = 0 (le modèle S1)
    python -m roadshift.fewshot.gains --mlflow-dir <dossier avec mlflow.db et mlruns> \
        --data <dataset_meta> --out results/fewshot_gains.csv
    # chaque run contre son jumeau d'une autre série (même k, même tirage), ex. rejeu - cible seule
    python -m roadshift.fewshot.gains --mlflow-dir <s2b> --reference-mlflow-dir <s2> \
        --data <dataset_meta> --out results/replay_vs_target_only.csv

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


def paired_gains(
    pairs: list[tuple[str, str]], runs: dict[str, dict], tables: dict[tuple[str, str], DetectionTable],
    confs: dict[str, float], evals: list[str], n_boot: int = 1000, seed: int = 0,
) -> list[dict]:  # fmt: skip
    """pairs : [(run, référence)] ; renvoie run - référence pour chaque éval et métrique."""
    rows = []
    for a, b in pairs:
        for ev in evals:
            for met in ("mAP50", "F1"):
                p, lo, hi = paired_bootstrap_diff(
                    tables[(a, ev)], tables[(b, ev)], confs[a], confs[b], met, n_boot, seed
                )
                rows.append({"run": a, "reference": b, "k": runs[a]["k"], "seed": runs[a]["seed"], "eval": ev,
                             "metric": met, "gain": round(p, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                             "significant": lo > 0 or hi < 0})  # fmt: skip
    return rows


def pairs_vs_k0(runs: dict[str, dict]) -> list[tuple[str, str]]:
    base = next(n for n, r in runs.items() if r["k"] == 0)
    order = sorted((n for n in runs if n != base), key=lambda n: (runs[n]["k"], str(runs[n]["seed"])))
    return [(n, base) for n in order]


def pairs_vs_reference(runs: dict[str, dict], ref: dict[str, dict]) -> list[tuple[str, str]]:
    """Associe chaque run k > 0 au run de référence de même (k, tirage)."""
    by_key = {(r["k"], r["seed"]): n for n, r in ref.items()}
    out = []
    for n in sorted(runs, key=lambda n: (runs[n]["k"], str(runs[n]["seed"]))):
        if runs[n]["k"] == 0:
            continue
        key = (runs[n]["k"], runs[n]["seed"])
        if key not in by_key:
            raise KeyError(f"pas de run de référence pour k={key[0]}, tirage={key[1]}")
        out.append((n, by_key[key]))
    return out


def load_from_mlflow(mlflow_dir: Path, model: str, prefix: str = "") -> dict[str, dict]:
    """Pour chaque run few-shot du modèle : k, tirage et dossier d'artefacts de prédictions."""
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{(Path(mlflow_dir) / 'mlflow.db').resolve()}")
    df = mlflow.search_runs(search_all_experiments=True)
    df = df[(df["params.model"] == model) & df["params.k"].notna()]
    out = {}
    for _, r in df.iterrows():
        seed = r["params.subset_seed"]
        out[prefix + r["tags.mlflow.runName"]] = {
            "k": int(r["params.k"]),
            "seed": None if seed in (None, "None") else int(seed),
            "pred_dir": Path(mlflow_dir) / "mlruns" / r["experiment_id"] / r["run_id"] / "artifacts" / "predictions",
        }
    return out


def build_tables(runs: dict[str, dict], data: Path, evals: list[str]):
    sizes = load_sizes(data / "manifest.csv")
    tables = {}
    for ev in ["source_val", *evals]:
        stems = read_image_list(data / f"list_{ev}.txt")
        gt = load_ground_truth(stems, data / "labels", sizes)
        for name, r in runs.items():
            tables[(name, ev)] = match(stems, gt, load_predictions(r["pred_dir"] / ev / "predictions.json"), NC)
    confs = {name: best_f1_threshold(tables[(name, "source_val")]) for name in runs}
    return tables, confs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mlflow-dir", required=True, type=Path)
    ap.add_argument("--reference-mlflow-dir", type=Path, help="si donné : run - run de référence de même (k, tirage)")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--model", default="yolo11s")
    ap.add_argument("--evals", nargs="+", default=["test_United_States", "source_test"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)

    runs = load_from_mlflow(args.mlflow_dir, args.model)
    if args.reference_mlflow_dir:
        ref = load_from_mlflow(args.reference_mlflow_dir, args.model, prefix="ref:")
        pairs = pairs_vs_reference(runs, ref)
        runs = {**runs, **ref}
    else:
        pairs = pairs_vs_k0(runs)
    tables, confs = build_tables(runs, args.data, args.evals)
    rows = paired_gains(pairs, runs, tables, confs, args.evals, args.n_boot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        star = "  *" if r["significant"] else ""
        print(
            f"{r['run']:44s} {r['eval']:20s} {r['metric']:5s} {r['gain']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]{star}"
        )


if __name__ == "__main__":
    main()
