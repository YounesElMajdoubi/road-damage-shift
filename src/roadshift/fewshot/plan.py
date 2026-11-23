"""Q1 : plan des fine-tunings few-shot, coût d'annotation estimé et agrégation de la courbe.

Le coût d'annotation n'est PAS mesuré : il applique un modèle publié (temps par image + temps par boîte)
aux nombres exacts d'images et de boîtes de chaque sous-ensemble, pour donner une fourchette.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FewShotRun:
    name: str  # ex. "United_States_k50_s0", "United_States_full"
    k: int  # nombre d'images cibles annotées (0 = modèle source seul)
    seed: int | None  # tirage du sous-ensemble ; None pour k=0 et pour la réserve complète
    list_name: str | None  # liste d'images d'entraînement (None pour k=0)


def plan_runs(fewshot_cfg: dict, pool_size: int) -> list[FewShotRun]:
    """k=0 (évaluation seule), puis chaque taille x chaque tirage, puis la réserve complète (plafond)."""
    t = fewshot_cfg["target"]
    runs = [FewShotRun(f"{t}_k0", 0, None, None)]
    for k in sorted(fewshot_cfg["sizes"]):
        if k > pool_size:
            raise ValueError(f"k={k} > réserve cible de {pool_size} images")
        for s in fewshot_cfg["seeds"]:
            runs.append(FewShotRun(f"{t}_k{k}_s{s}", k, s, f"list_fewshot_{t}_k{k}_s{s}.txt"))
    if fewshot_cfg.get("include_full_pool"):
        runs.append(FewShotRun(f"{t}_full", pool_size, None, f"list_pool_{t}.txt"))
    return runs


def write_replay_list(data_root: Path, run: FewShotRun, ratio: float, seed: int) -> tuple[str, int]:
    """Liste d'entraînement S2b : les k images cibles + round(ratio * k) images du TRAIN source (tirage seedé).

    Retourne (nom de la liste écrite, nombre d'images sources ajoutées).
    """
    if run.list_name is None:
        raise ValueError("k=0 : pas d'entraînement, donc pas de liste de rejeu")
    root = Path(data_root)
    target = root.joinpath(run.list_name).read_text().split()
    source = sorted(root.joinpath("list_source_train.txt").read_text().split())
    n = round(ratio * len(target))
    if n > len(source):
        raise ValueError(f"rejeu de {n} images demandé, {len(source)} disponibles dans le train source")
    replay = sorted(random.Random(seed).sample(source, n))
    name = f"list_replay_{run.name}.txt"
    root.joinpath(name).write_text("\n".join(target + replay) + "\n", encoding="utf-8", newline="\n")
    return name, n


def write_fewshot_yaml(data_root: Path, run: FewShotRun, classes: list[str], train_list: str | None = None) -> Path:
    """Data YAML Ultralytics : train = k images cibles (ou `train_list`, ex. liste de rejeu) ;
    val = validation SOURCE (aucune image cible en plus)."""
    if run.list_name is None:
        raise ValueError("k=0 : pas d'entraînement, donc pas de YAML")
    train = train_list or run.list_name
    d = {"train": train, "val": "list_source_val.txt", "names": dict(enumerate(classes))}
    suffix = "_replay" if train_list else ""
    path = Path(data_root) / f"data_fewshot_{run.name}{suffix}.yaml"
    path.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8", newline="\n")
    return path


def finetune_lr0(model_spec: dict, lr_scale: float, nc: int) -> float:
    """lr de fine-tuning = lr effectif de l'entraînement S1 x lr_scale.

    En `optimizer: auto` (YOLO), Ultralytics choisit AdamW avec lr = 0.002 * 5 / (4 + nc) quand le nombre
    d'itérations est faible : 0.00125 pour 4 classes en S1 (results.csv du run YOLO11s : lr max 0.0012
    juste après le warmup, puis décroissance linéaire).
    """
    base = model_spec.get("lr0") or round(0.002 * 5 / (4 + nc), 6)
    return base * lr_scale


def count_annotations(list_txt: Path, labels_root: Path, classes: list[str]) -> dict:
    """Nombre d'images et de boîtes (total et par classe) d'une liste Ultralytics './images/Pays/stem.jpg'."""
    per = dict.fromkeys(classes, 0)
    stems = [Path(x).stem for x in Path(list_txt).read_text().split()]
    for s in stems:
        for line in (Path(labels_root) / s.rsplit("_", 1)[0] / f"{s}.txt").read_text().splitlines():
            if line.strip():
                per[classes[int(line.split()[0])]] += 1
    return {"n_images": len(stems), "n_boxes": sum(per.values()), **{f"n_{c}": v for c, v in per.items()}}


def annotation_cost_hours(n_images: int, n_boxes: int, cost_cfg: dict) -> tuple[float, float]:
    """Fourchette (basse, haute) en heures : s_par_image * images + s_par_boîte * boîtes."""
    base = cost_cfg["s_per_image"] * n_images
    return (
        (base + cost_cfg["s_per_box_low"] * n_boxes) / 3600,
        (base + cost_cfg["s_per_box_high"] * n_boxes) / 3600,
    )


def aggregate_curve(rows: list[dict], metric: str = "mAP50") -> list[dict]:
    """Agrège les runs par (modèle, éval, k) : moyenne et étendue sur les tirages, et IC bootstrap.

    rows : une ligne par (run, éval) avec model, eval, k, seed, <metric>, <metric>_lo, <metric>_hi.
    Avec un seul tirage (k=0, plafond), l'IC bootstrap du run est rapporté ; avec plusieurs tirages,
    on rapporte aussi la moyenne des bornes bootstrap (incertitude due au test) et l'étendue entre
    tirages (incertitude due au choix des images annotées).
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["eval"], int(r["k"]))].append(r)
    out = []
    for (model, ev, k), rs in sorted(groups.items()):
        v = [float(r[metric]) for r in rs]
        mean = sum(v) / len(v)
        sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
        out.append(
            {
                "model": model,
                "eval": ev,
                "k": k,
                "n_seeds": len(rs),
                metric: round(mean, 4),
                f"{metric}_seed_sd": round(sd, 4),
                f"{metric}_seed_min": round(min(v), 4),
                f"{metric}_seed_max": round(max(v), 4),
                f"{metric}_boot_lo": round(sum(float(r[f"{metric}_lo"]) for r in rs) / len(rs), 4),
                f"{metric}_boot_hi": round(sum(float(r[f"{metric}_hi"]) for r in rs) / len(rs), 4),
            }
        )
    return out
