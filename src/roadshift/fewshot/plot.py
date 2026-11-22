"""Figure Q1 : performance vs nombre d'images cibles annotées, avec le coût estimé sous chaque k.

Deux panneaux (petits multiples, jamais deux axes y) : le pays cible, et le test source pour mesurer l'oubli.
Barres verticales = IC bootstrap à 95 % (moyenne des bornes sur les tirages) ; petits points = chaque tirage.
"""

from __future__ import annotations

from pathlib import Path

SERIES = {"yolo11s": "#2a78d6", "rtdetr_l": "#eb6834"}  # palette catégorielle validée (ordre fixe)
LABELS = {"yolo11s": "YOLO11s", "rtdetr_l": "RT-DETR-l"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _xlabels(ks: list[int], cost: dict[int, tuple[float, float]], full_k: int | None) -> list[str]:
    out = []
    for k in ks:
        name = f"réserve\n({k})" if k == full_k else str(k)
        lo, hi = cost.get(k, (0.0, 0.0))
        out.append(name if k == 0 else f"{name}\n{lo:.1f}–{hi:.1f} h")
    return out


def plot_curve(
    curve: list[dict],
    runs: list[dict],
    cost: dict[int, tuple[float, float]],
    panels: list[tuple[str, str]],
    out_png: str | Path,
    metric: str = "mAP50",
    full_k: int | None = None,
) -> Path:
    """curve : sortie de aggregate_curve ; runs : lignes par run (pour les points de chaque tirage) ;
    cost : k -> (heures basse, haute) ; panels : [(eval_tag, titre)]."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = sorted({int(r["k"]) for r in curve})
    pos = {k: i for i, k in enumerate(ks)}
    models = [m for m in SERIES if any(r["model"] == m for r in curve)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.2), sharey=False, facecolor=SURFACE)
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (tag, title) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)
        for j, m in enumerate(models):
            pts = sorted((r for r in curve if r["model"] == m and r["eval"] == tag), key=lambda r: int(r["k"]))
            if not pts:
                continue
            dx = (j - (len(models) - 1) / 2) * 0.08  # léger décalage pour ne pas superposer les modèles
            x = [pos[int(r["k"])] + dx for r in pts]
            y = [r[metric] for r in pts]
            lo = [r[metric] - r[f"{metric}_boot_lo"] for r in pts]
            hi = [r[f"{metric}_boot_hi"] - r[metric] for r in pts]
            c = SERIES[m]
            ax.errorbar(x, y, yerr=[lo, hi], color=c, lw=2, marker="o", ms=8, capsize=0, elinewidth=1.5,
                        markeredgecolor=SURFACE, markeredgewidth=2, label=LABELS.get(m, m), zorder=3)  # fmt: skip
            seeds = [r for r in runs if r["model"] == m and r["eval"] == tag and r.get("seed") not in (None, "", "nan")]
            ax.scatter([pos[int(r["k"])] + dx + 0.12 for r in seeds], [float(r[metric]) for r in seeds],
                       s=14, color=c, alpha=0.55, linewidths=0, zorder=2)  # fmt: skip
            ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)  # fmt: skip
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xticks(range(len(ks)), _xlabels(ks, cost, full_k), fontsize=8, color=INK2)
        ax.set_xlabel("images cibles annotées (coût estimé, modèle publié)", fontsize=9, color=INK2)
        ax.set_ylabel(f"{metric} (IC 95 % bootstrap)", fontsize=9, color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.set_xlim(-0.4, len(ks) - 0.4)
    if len(models) >= 2:  # une seule série : pas de légende, le titre de la figure la nomme
        axes[0].legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    elif models:
        fig.suptitle(LABELS.get(models[0], models[0]), x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout()
    out = Path(out_png)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out
