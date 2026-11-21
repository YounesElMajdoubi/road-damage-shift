"""Métriques de détection avec intervalles de confiance bootstrap.

- Appariement prédiction / vérité terrain par image et par classe, glouton par score décroissant
  (convention COCO), IoU >= 0.5.
- AP@50 par classe : 101 seuils de rappel, définition COCO par défaut (voir `_ap` pour l'écart avec
  Ultralytics), mAP@50 = moyenne des classes présentes dans la vérité terrain.
- F1 « CRDDC » : F1 micro toutes classes (classe correcte + IoU > 0.5) à un seuil de confiance unique,
  choisi sur la validation source puis appliqué tel quel aux tests (aucun réglage sur un test).
- Boîtes évaluées telles qu'exportées dans predictions.json, donc rognées au cadre de l'image (convention
  COCO : on évalue ce qu'on livre). Ultralytics, lui, apparie dans l'espace letterbox SANS rogner : une boîte
  qui déborde sur le remplissage y perd de l'IoU. Vérifié en patchant son validateur pour rogner : son mAP@50
  rejoint alors celui de ce module (écart résiduel dû aux scores ex æquo).
- Bootstrap par rééchantillonnage d'images : chaque image tirée reçoit un poids (son nombre de tirages),
  ce qui évite de recopier les détections et permet B = 1000 en quelques secondes.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IOU_THR = 0.5
_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2 / < 2


# --------------------------------------------------------------------------- chargement
def load_sizes(manifest_csv: str | Path) -> dict[str, tuple[int, int]]:
    """stem -> (largeur, hauteur) de l'image réellement vue par le modèle."""
    with open(manifest_csv, encoding="utf-8") as f:
        return {r["stem"]: (int(r["out_width"]), int(r["out_height"])) for r in csv.DictReader(f)}


def read_image_list(list_txt: str | Path) -> list[str]:
    """Liste Ultralytics ('./images/Pays/stem.jpg') -> stems, dans l'ordre."""
    return [Path(line).stem for line in Path(list_txt).read_text().split()]


def load_ground_truth(stems: list[str], labels_root: str | Path, sizes: dict[str, tuple[int, int]]):
    """Labels YOLO normalisés -> {stem: (classes int[n], boîtes xyxy pixels float[n,4])}."""
    labels_root = Path(labels_root)
    gt = {}
    for s in stems:
        country = s.rsplit("_", 1)[0]
        w, h = sizes[s]
        f = labels_root / country / f"{s}.txt"
        rows = np.loadtxt(f, ndmin=2) if f.stat().st_size else np.zeros((0, 5))
        cls = rows[:, 0].astype(int)
        cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
        gt[s] = (cls, np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1))
    return gt


def load_predictions(json_path: str | Path, class_offset: int = 1):
    """predictions.json d'Ultralytics (bbox xywh coin haut-gauche, category_id = classe + 1)
    -> {stem: (classes, boîtes xyxy, scores)}."""
    by: dict[str, list] = {}
    for d in json.loads(Path(json_path).read_text()):
        by.setdefault(str(d["image_id"]), []).append(d)
    out = {}
    for s, ds in by.items():
        b = np.array([d["bbox"] for d in ds], dtype=float)
        out[s] = (
            np.array([d["category_id"] - class_offset for d in ds], dtype=int),
            np.concatenate([b[:, :2], b[:, :2] + b[:, 2:]], axis=1),
            np.array([d["score"] for d in ds], dtype=float),
        )
    return out


# --------------------------------------------------------------------------- appariement
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre boîtes xyxy : a[n,4], b[m,4] -> [n,m]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-12)


@dataclass
class DetectionTable:
    """Résultat d'appariement, indexé par image pour le bootstrap."""

    stems: list[str]
    img: np.ndarray  # [N] index d'image de chaque détection
    cls: np.ndarray  # [N]
    score: np.ndarray  # [N] trié par ordre décroissant
    tp: np.ndarray  # [N] bool
    n_gt: np.ndarray  # [n_images, nc] nombre de boîtes vraies
    nc: int


def match(stems: list[str], gt: dict, preds: dict, nc: int, iou_thr: float = IOU_THR) -> DetectionTable:
    """Glouton par score décroissant : une boîte vraie ne peut valider qu'une prédiction de même classe."""
    img, cls_l, sc, tp = [], [], [], []
    n_gt = np.zeros((len(stems), nc), dtype=int)
    for i, s in enumerate(stems):
        gc, gb = gt[s]
        n_gt[i] = np.bincount(gc, minlength=nc)[:nc]
        if s not in preds:
            continue
        pc, pb, ps = preds[s]
        order = np.argsort(-ps, kind="stable")
        pc, pb, ps = pc[order], pb[order], ps[order]
        ious = box_iou(pb, gb)
        used = np.zeros(len(gb), dtype=bool)
        for k in range(len(ps)):
            cand = np.where((gc == pc[k]) & ~used)[0]
            hit = False
            if len(cand):
                j = cand[np.argmax(ious[k, cand])]
                if ious[k, j] >= iou_thr:
                    used[j] = hit = True
            img.append(i)
            cls_l.append(pc[k])
            sc.append(ps[k])
            tp.append(hit)
    img_a, cls_a, sc_a, tp_a = np.array(img, dtype=int), np.array(cls_l, dtype=int), np.array(sc), np.array(tp, bool)
    o = np.argsort(-sc_a, kind="stable")  # trié une fois : l'ordre ne dépend pas des poids bootstrap
    return DetectionTable(stems, img_a[o], cls_a[o], sc_a[o], tp_a[o], n_gt, nc)


# --------------------------------------------------------------------------- métriques pondérées
def _ap(score: np.ndarray, tp: np.ndarray, w: np.ndarray, n_gt: float, method: str = "coco") -> float:
    """AP sur 101 seuils de rappel ; w = poids bootstrap de chaque détection.

    - "coco" (défaut) : précision interpolée (max à rappel >= r) échantillonnée en r = 0, 0.01, ..., 1 ;
      0 au-delà du rappel maximal atteint. C'est la définition de pycocotools.
    - "ultralytics" : intégrale trapézoïdale de l'enveloppe prolongée linéairement jusqu'à rappel 1
      (reproduit `ultralytics.utils.metrics.compute_ap`). Optimiste quand le rappel maximal < 1
      (ex. 0.75 au lieu de ~0.5 si la moitié des objets est ratée) et plafonnée à 0.995.
      Sert uniquement à valider ce module contre les chiffres d'Ultralytics.
    """
    if n_gt <= 0:
        return float("nan")
    if len(score) == 0:
        return 0.0
    # Détections déjà triées par score décroissant (voir `match`).
    ctp = np.cumsum(w * tp)
    cfp = np.cumsum(w * ~tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    x = np.linspace(0, 1, 101)
    if method == "coco":
        env = np.flip(np.maximum.accumulate(np.flip(prec)))
        idx = np.searchsorted(rec, x - 1e-12, side="left")
        return float(np.where(idx < len(rec), env[np.minimum(idx, len(rec) - 1)], 0.0).mean())
    if method == "ultralytics":
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate(([1.0], prec, [0.0])))))
        return float(_trapz(np.interp(x, mrec, mpre), x))
    raise ValueError(f"méthode d'AP inconnue : {method!r}")


def compute(t: DetectionTable, conf: float, img_w: np.ndarray | None = None, method: str = "coco") -> dict[str, float]:
    """mAP@50, AP@50 par classe, et P / R / F1 micro au seuil `conf`. img_w = poids par image."""
    img_w = np.ones(len(t.stems)) if img_w is None else img_w
    w = img_w[t.img]
    n_gt = (img_w[:, None] * t.n_gt).sum(0)
    out = {}
    aps = []
    for c in range(t.nc):
        m = t.cls == c
        ap = _ap(t.score[m], t.tp[m], w[m], n_gt[c], method)
        out[f"AP50_{c}"] = ap
        if not np.isnan(ap):
            aps.append(ap)
    out["mAP50"] = float(np.mean(aps)) if aps else float("nan")
    keep = t.score >= conf
    tp_w = float((w * (t.tp & keep)).sum())
    fp_w = float((w * (~t.tp & keep)).sum())
    fn_w = float(n_gt.sum() - tp_w)
    p = tp_w / (tp_w + fp_w) if tp_w + fp_w else 0.0
    r = tp_w / (tp_w + fn_w) if tp_w + fn_w else 0.0
    out.update(precision=p, recall=r, F1=2 * p * r / (p + r) if p + r else 0.0)
    return out


def best_f1_threshold(t: DetectionTable, grid: np.ndarray | None = None) -> float:
    """Seuil de confiance qui maximise le F1 micro (à utiliser sur la validation uniquement)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else grid
    f1 = [compute(t, c)["F1"] for c in grid]
    return float(grid[int(np.argmax(f1))])


def bootstrap(t: DetectionTable, conf: float, n: int = 1000, ci: float = 0.95, seed: int = 0, method: str = "coco"):
    """Point estimate + IC percentile par rééchantillonnage d'images. -> {métrique: (val, bas, haut)}."""
    rng = np.random.default_rng(seed)
    point = compute(t, conf, method=method)
    n_img = len(t.stems)
    draws = {k: [] for k in point}
    for _ in range(n):
        w = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img).astype(float)
        for k, v in compute(t, conf, w, method).items():
            draws[k].append(v)
    a = (1 - ci) / 2
    return {k: (point[k], float(np.nanquantile(draws[k], a)), float(np.nanquantile(draws[k], 1 - a))) for k in point}


def paired_bootstrap_diff(
    a: DetectionTable,
    b: DetectionTable,
    conf_a: float,
    conf_b: float,
    metric: str = "mAP50",
    n: int = 1000,
    seed: int = 0,
):
    """IC de la différence a - b quand les deux modèles sont évalués sur LES MÊMES images (rééchantillonnage commun)."""
    if a.stems != b.stems:
        raise ValueError(
            "comparaison appariée : les deux tables doivent porter sur les mêmes images, dans le même ordre"
        )
    rng = np.random.default_rng(seed)
    n_img = len(a.stems)
    d = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img).astype(float)
        d.append(compute(a, conf_a, w)[metric] - compute(b, conf_b, w)[metric])
    point = compute(a, conf_a)[metric] - compute(b, conf_b)[metric]
    return point, float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
