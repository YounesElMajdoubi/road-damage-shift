"""Q2 : performance sous flou de bougé, avec IC bootstrap et écarts appariés entre approches.

Usage :
    python -m roadshift.blur.report --base <run du modèle S1> --aug <run du modèle entraîné avec flou> \
        --data <dataset converti + jeux floutés> --out-dir results/

Chaque dossier de run contient eval_<tag>/predictions.json (tag = source_val, blur_L<L>_<jeu>, wiener_L<L>_<jeu>).
Approches comparées à chaque niveau de flou L, sur les mêmes images :
    aucun traitement : modèle S1 sur images floutées
    Wiener           : modèle S1 sur images floutées puis défloutées (PSF connue)
    augmentation     : modèle entraîné avec flou aléatoire, sur images floutées
Seuil du F1 : choisi sur la validation source PROPRE de chaque modèle. La vérité terrain est inchangée par le flou.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from roadshift.eval.metrics import (
    best_f1_threshold,
    bootstrap,
    load_ground_truth,
    load_predictions,
    load_sizes,
    match,
    paired_bootstrap_diff,
    read_image_list,
)

NC = 4
METHODS = {"none": "Aucun traitement", "wiener": "Défloutage Wiener (PSF connue)", "aug": "Augmentation par flou"}
TAG = re.compile(r"^(blur|wiener)_L(\d+)_(.+)$")


class Evaluator:
    def __init__(self, data: Path):
        self.data, self.sizes, self._gt = data, load_sizes(data / "manifest.csv"), {}

    def table(self, run_dir: Path, tag: str):
        stems = read_image_list(self.data / f"list_{tag}.txt")
        key = tuple(stems)
        if key not in self._gt:
            self._gt[key] = load_ground_truth(stems, self.data / "labels", self.sizes)
        return match(stems, self._gt[key], load_predictions(run_dir / f"eval_{tag}" / "predictions.json"), NC)


def _tags(run_dir: Path) -> list[tuple[str, int, str]]:
    out = []
    for p in run_dir.glob("eval_*"):
        m = TAG.match(p.name.removeprefix("eval_"))
        if m and (p / "predictions.json").exists():
            out.append((m.group(1), int(m.group(2)), m.group(3)))
    return sorted(out, key=lambda x: (x[2], x[1], x[0]))


def blur_report(base: Path, aug: Path | None, data: Path, n_boot: int = 1000, seed: int = 0):
    ev = Evaluator(data)
    conf = {"base": best_f1_threshold(ev.table(base, "source_val"))}
    if aug:
        conf["aug"] = best_f1_threshold(ev.table(aug, "source_val"))
    tables = {}
    for kind, length, s in _tags(base):
        tag = f"{kind}_L{length}_{s}"
        tables[("wiener" if kind == "wiener" else "none", s, length)] = ev.table(base, tag)
    for kind, length, s in _tags(aug) if aug else []:
        if kind == "blur":
            tables[("aug", s, length)] = ev.table(aug, f"blur_L{length}_{s}")
    for m, s, length in list(tables):  # L = 0 : Wiener avec le noyau identité = aucun traitement
        if m == "none" and length == 0:
            tables[("wiener", s, 0)] = tables[(m, s, 0)]
    scores, gains = [], []
    for (m, s, length), t in sorted(tables.items(), key=lambda x: (x[0][1], x[0][2], x[0][0])):
        c = conf["aug" if m == "aug" else "base"]
        b = bootstrap(t, c, n=n_boot, seed=seed)
        row = {"method": m, "eval_set": s, "blur_length": length, "n_images": len(t.stems), "conf_F1": c}
        for k in ("mAP50", "F1", "precision", "recall"):
            row[k], row[f"{k}_lo"], row[f"{k}_hi"] = (round(v, 4) for v in b[k])
        scores.append(row)
        if m != "none" and ("none", s, length) in tables and not (m == "wiener" and length == 0):
            for met in ("mAP50", "F1"):
                p, lo, hi = paired_bootstrap_diff(t, tables[("none", s, length)], c, conf["base"], met, n_boot, seed)
                gains.append({"comparison": f"{m} - none", "eval_set": s, "blur_length": length, "metric": met,
                              "diff": round(p, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                              "significant": lo > 0 or hi < 0})  # fmt: skip
    return scores, gains


def plot_blur(scores: list[dict], out_png: Path, metric: str = "mAP50") -> Path:
    """Petits multiples (un panneau par jeu), 3 séries en couleurs validées + marqueurs distincts + étiquettes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from roadshift.fewshot.plot import GRID, INK, INK2, SURFACE

    colors = {"none": "#2a78d6", "wiener": "#eb6834", "aug": "#1baf7a"}  # palette validée, ordre fixe
    markers = {"none": "o", "wiener": "s", "aug": "D"}  # encodage secondaire (turquoise peu contrasté)
    titles = {"source_test": "Test des pays sources", "test_United_States": "Pays jamais vu (États-Unis)"}
    sets = [s for s in ("source_test", "test_United_States") if any(r["eval_set"] == s for r in scores)]
    fig, axes = plt.subplots(1, len(sets), figsize=(5.4 * len(sets), 4.2), facecolor=SURFACE)
    axes = [axes] if len(sets) == 1 else list(axes)
    for ax, s in zip(axes, sets, strict=True):
        ax.set_facecolor(SURFACE)
        lengths = sorted({int(r["blur_length"]) for r in scores if r["eval_set"] == s})
        pos = {length: i for i, length in enumerate(lengths)}
        present = [m for m in METHODS if any(r["method"] == m and r["eval_set"] == s for r in scores)]
        for j, m in enumerate(present):
            pts = sorted((r for r in scores if r["method"] == m and r["eval_set"] == s), key=lambda r: r["blur_length"])
            dx = (j - (len(present) - 1) / 2) * 0.09
            x = [pos[int(r["blur_length"])] + dx for r in pts]
            y = [float(r[metric]) for r in pts]
            err = [[y_ - float(r[f"{metric}_lo"]) for y_, r in zip(y, pts, strict=True)],
                   [float(r[f"{metric}_hi"]) - y_ for y_, r in zip(y, pts, strict=True)]]  # fmt: skip
            ax.errorbar(x, y, yerr=err, color=colors[m], lw=2, marker=markers[m], ms=8, capsize=0, elinewidth=1.5,
                        markeredgecolor=SURFACE, markeredgewidth=2, label=METHODS[m], zorder=3)  # fmt: skip
            dy = (j - (len(present) - 1) / 2) * -13
            ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(10, dy), textcoords="offset points", va="center",
                        fontsize=9, color=INK)  # fmt: skip
        ax.set_title(titles.get(s, s), loc="left", fontsize=11, color=INK)
        ax.set_xticks(range(len(lengths)), [f"{v} px" if v else "net" for v in lengths], fontsize=8, color=INK2)
        ax.set_xlabel("longueur du flou de bougé", fontsize=9, color=INK2)
        ax.set_ylabel(f"{metric} (IC 95 % bootstrap)", fontsize=9, color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.set_xlim(-0.4, len(lengths) - 0.4)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out_png


def _write(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--aug", type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    scores, gains = blur_report(args.base, args.aug, args.data, args.n_boot)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write(scores, args.out_dir / f"blur_scores{args.suffix}.csv")
    if gains:
        _write(gains, args.out_dir / f"blur_gains{args.suffix}.csv")
    plot_blur(scores, args.out_dir / f"blur_curve{args.suffix}.png")
    for g in gains:
        star = "  *" if g["significant"] else ""
        print(f"{g['comparison']:14s} {g['eval_set']:20s} L={g['blur_length']:<3d} {g['metric']:5s} "
              f"{g['diff']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]{star}")  # fmt: skip


if __name__ == "__main__":
    main()
