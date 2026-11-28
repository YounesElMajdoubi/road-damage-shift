"""Sorties des détecteurs zero-shot -> predictions.json au format Ultralytics (fonctions pures, sans torch).

Grounding DINO renvoie, pour chaque requête (boîte), une probabilité par token du texte. Ses étiquettes texte
peuvent être partielles (« crack » seul), donc ambiguës entre classes. On suit plutôt l'évaluation COCO de
Grounding DINO : score d'une classe = moyenne des probabilités des tokens de sa phrase (« positive map »),
puis chaque requête prend la classe de score maximal.

Florence-2 renvoie des boîtes sans score par boîte : on leur donne le score de la séquence générée
(un score par image et par classe).
"""

from __future__ import annotations

import numpy as np


def build_caption(phrases: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """« a . b . c . » (format de Grounding DINO) et la position (début, fin) de chaque phrase dans le texte."""
    text, spans = "", []
    for p in phrases:
        p = p.strip().lower()
        if not p or "." in p:
            raise ValueError(f"phrase invalide : {p!r}")
        spans.append((len(text), len(text) + len(p)))
        text += p + " . "
    return text.strip(), spans


def positive_map(offsets: list[tuple[int, int]], spans: list[tuple[int, int]], n_tokens: int) -> np.ndarray:
    """[n_classes, n_tokens] : 1/k sur les k tokens de chaque phrase (offsets du tokenizer, (0, 0) = spécial)."""
    m = np.zeros((len(spans), n_tokens))
    for c, (a, b) in enumerate(spans):
        idx = [t for t, (s, e) in enumerate(offsets[:n_tokens]) if e > s and s >= a and e <= b]
        if not idx:
            raise ValueError(f"aucun token pour la phrase {c} ({a}, {b})")
        m[c, idx] = 1.0 / len(idx)
    return m


def class_scores(prob: np.ndarray, pmap: np.ndarray) -> np.ndarray:
    """prob [Q, T] (sigmoïde des logits) x positive map [C, T] -> [Q, C] = moyenne sur les tokens de la phrase."""
    if prob.shape[1] != pmap.shape[1]:
        raise ValueError(f"tokens : {prob.shape[1]} (logits) != {pmap.shape[1]} (positive map)")
    return np.nan_to_num(prob, nan=0.0) @ pmap.T


def select_detections(boxes_cxcywh: np.ndarray, scores: np.ndarray, w: int, h: int, top_k: int, min_score: float):
    """Boîtes normalisées [Q, 4] + scores par classe [Q, C] -> (classes, boîtes xyxy pixels bornées, scores),
    une classe par requête (la meilleure), top_k requêtes au plus, score >= min_score."""
    cls = scores.argmax(1)
    sc = scores[np.arange(len(scores)), cls]
    keep = np.where(sc >= min_score)[0]
    keep = keep[np.argsort(-sc[keep], kind="stable")][:top_k]
    cx, cy, bw, bh = (boxes_cxcywh[keep] * [w, h, w, h]).T
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1) if len(keep) else np.zeros((0, 4))
    return cls[keep], clip_boxes(xyxy, w, h), sc[keep]


def clip_boxes(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    out = np.asarray(xyxy, dtype=float).reshape(-1, 4).copy()
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out


def to_records(stem: str, cls: np.ndarray, xyxy: np.ndarray, score: np.ndarray) -> list[dict]:
    """Format predictions.json d'Ultralytics : image_id = nom de fichier sans extension, category_id = classe + 1,
    bbox = [x, y, largeur, hauteur] en pixels. Les boîtes vides (après bornage) sont retirées."""
    out = []
    for c, b, s in zip(cls, xyxy, score, strict=True):
        x1, y1, x2, y2 = (float(v) for v in b)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        out.append({"image_id": stem, "category_id": int(c) + 1, "bbox": [round(x1, 2), round(y1, 2),
                    round(x2 - x1, 2), round(y2 - y1, 2)], "score": round(float(s), 5)})  # fmt: skip
    return out


def florence_boxes(parsed: dict) -> np.ndarray:
    """Résultat de post_process_generation (<OPEN_VOCABULARY_DETECTION>) -> boîtes xyxy pixels.
    Florence peut répondre par des boîtes ou des polygones : un polygone devient sa boîte englobante."""
    boxes = [list(map(float, b)) for b in parsed.get("bboxes", [])]
    for poly in parsed.get("polygons", []):
        pts = np.asarray([v for part in poly for v in part] if poly and isinstance(poly[0], list | tuple) else poly,
                         dtype=float).reshape(-1, 2)  # fmt: skip
        if len(pts):
            boxes.append([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
    return np.asarray(boxes, dtype=float).reshape(-1, 4)
