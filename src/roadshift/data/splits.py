"""Découpage reproductible par blocs et sous-ensembles few-shot imbriqués."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence

_DIGITS = re.compile(r"(\d+)")


def natural_key(s: str) -> list:
    """Japan_2 < Japan_10 (tri humain)."""
    return [int(t) if t.isdigit() else t.lower() for t in _DIGITS.split(s)]


def make_blocks(stems: Sequence[str], block_size: int) -> list[list[str]]:
    """Regroupe les images consécutives (tri naturel) en blocs de `block_size`."""
    if block_size < 1:
        raise ValueError("block_size doit être >= 1")
    ordered = sorted(stems, key=natural_key)
    return [ordered[i : i + block_size] for i in range(0, len(ordered), block_size)]


def shuffled_blocks(stems: Sequence[str], block_size: int, seed: int) -> list[list[str]]:
    blocks = make_blocks(stems, block_size)
    random.Random(seed).shuffle(blocks)
    return blocks


def take_blocks(blocks: list[list[str]], max_images: int | None) -> list[list[str]]:
    """Sous-échantillonnage : garde les premiers blocs (déjà mélangés) jusqu'à max_images.

    Le dernier bloc est tronqué pour atteindre exactement max_images.
    """
    if max_images is None:
        return blocks
    out, n = [], 0
    for b in blocks:
        if n >= max_images:
            break
        b = b[: max_images - n]
        out.append(b)
        n += len(b)
    return out


def split_blocks(blocks: list[list[str]], spec: dict[str, float | int | str]) -> dict[str, list[str]]:
    """Répartit des blocs entre splits sans jamais couper un bloc.

    spec : {nom: fraction (0<f<1) | nombre d'images (int>=1) | "rest"}, traité dans l'ordre.
    Un seul "rest" autorisé, forcément en dernier. Sans "rest", le dernier split
    reçoit le reliquat (les arrondis ne perdent aucune image).
    """
    names = list(spec)
    if not names:
        raise ValueError("spec vide")
    rest_names = [k for k, v in spec.items() if v == "rest"]
    if len(rest_names) > 1 or (rest_names and rest_names[0] != names[-1]):
        raise ValueError("'rest' doit être unique et en dernière position")

    total = sum(len(b) for b in blocks)
    targets: dict[str, int] = {}
    for k, v in spec.items():
        if v == "rest":
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"valeur invalide pour {k}: {v!r}")
        if isinstance(v, float) and 0 < v < 1:
            targets[k] = round(v * total)
        elif isinstance(v, int) and v >= 1:
            targets[k] = v
        else:
            raise ValueError(f"valeur invalide pour {k}: {v!r} (fraction dans ]0,1[ ou entier >= 1)")
    if sum(targets.values()) > total:
        raise ValueError(f"spec demande {sum(targets.values())} images, seulement {total} disponibles")

    out: dict[str, list[str]] = {k: [] for k in names}
    i = 0
    for k in names[:-1]:
        # On ajoute un bloc tant qu'il rapproche du nombre visé (dépassement <= déficit) :
        # chaque split est juste à un demi-bloc près, sans biais systématique vers le premier.
        while i < len(blocks):
            deficit = targets[k] - len(out[k])
            if deficit <= 0 or len(blocks[i]) - deficit > deficit:
                break
            out[k].extend(blocks[i])
            i += 1
    for b in blocks[i:]:  # le reliquat va au dernier split
        out[names[-1]].extend(b)
    return out


def nested_fewshot(pool: Sequence[str], sizes: Sequence[int], seed: int) -> dict[int, list[str]]:
    """Tire des sous-ensembles imbriqués (50 ⊂ 200 ⊂ 500) : une permutation, des préfixes."""
    sizes = sorted(sizes)
    if sizes and sizes[-1] > len(pool):
        raise ValueError(f"pool de {len(pool)} images < taille demandée {sizes[-1]}")
    perm = sorted(pool, key=natural_key)
    random.Random(seed).shuffle(perm)
    return {k: sorted(perm[:k], key=natural_key) for k in sizes}


def subsample_backgrounds(stems: Sequence[str], is_background: dict[str, bool], keep: float, seed: int) -> list[str]:
    """Garde toutes les images positives et une fraction `keep` des images sans objet."""
    if not 0.0 <= keep <= 1.0:
        raise ValueError("keep doit être dans [0, 1]")
    rng = random.Random(seed)
    return [s for s in stems if not is_background[s] or rng.random() < keep]
