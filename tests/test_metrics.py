import json

import numpy as np
import pytest

from roadshift.eval.metrics import (
    best_f1_threshold,
    bootstrap,
    box_iou,
    compute,
    load_ground_truth,
    load_predictions,
    load_sizes,
    match,
    paired_bootstrap_diff,
)

NC = 4


def _gt(*boxes):
    return (np.array([b[0] for b in boxes], dtype=int), np.array([b[1:] for b in boxes], dtype=float).reshape(-1, 4))


def _pred(*boxes):
    cls, xyxy, sc = zip(*[(b[0], b[1:5], b[5]) for b in boxes], strict=True) if boxes else ((), (), ())
    return np.array(cls, dtype=int), np.array(xyxy, dtype=float).reshape(-1, 4), np.array(sc, dtype=float)


def test_box_iou():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    np.testing.assert_allclose(box_iou(a, b), [[1.0, 1 / 3, 0.0]], atol=1e-9)


def test_perfect_predictions_give_ap_and_f1_of_one():
    gt = {"a": _gt((0, 0, 0, 10, 10), (3, 20, 20, 40, 40)), "b": _gt((1, 0, 0, 5, 5))}
    pr = {"a": _pred((0, 0, 0, 10, 10, 0.9), (3, 20, 20, 40, 40, 0.8)), "b": _pred((1, 0, 0, 5, 5, 0.7))}
    t = match(["a", "b"], gt, pr, NC)
    m = compute(t, conf=0.5)
    assert m["mAP50"] == pytest.approx(1.0) and m["F1"] == pytest.approx(1.0)
    assert compute(t, 0.5, method="ultralytics")["mAP50"] == pytest.approx(0.995)  # plafond connu d'Ultralytics
    assert np.isnan(m["AP50_2"])  # classe absente de la vérité terrain : exclue de la moyenne


def test_wrong_class_and_duplicates_are_false_positives():
    gt = {"a": _gt((0, 0, 0, 10, 10))}
    pr = {"a": _pred((0, 0, 0, 10, 10, 0.9), (0, 0, 0, 10, 10, 0.8), (1, 0, 0, 10, 10, 0.95))}
    t = match(["a"], gt, pr, NC)
    assert t.tp.tolist() == [False, True, False]  # trié par score : D10 (faux), D00 (vrai), doublon D00 (faux)
    m = compute(t, conf=0.0)
    assert m["precision"] == pytest.approx(1 / 3) and m["recall"] == pytest.approx(1.0)


def test_missed_images_count_as_false_negatives():
    gt = {"a": _gt((0, 0, 0, 10, 10)), "b": _gt((0, 0, 0, 10, 10))}
    pr = {"a": _pred((0, 0, 0, 10, 10, 0.9))}  # aucune prédiction sur b
    t = match(["a", "b"], gt, pr, NC)
    m = compute(t, conf=0.5)
    assert m["recall"] == pytest.approx(0.5) and m["AP50_0"] == pytest.approx(51 / 101)  # rappels 0..0.5 à précision 1
    # Ultralytics prolonge la courbe jusqu'à rappel 1 : AP optimiste.
    assert compute(t, 0.5, method="ultralytics")["AP50_0"] == pytest.approx(0.75)


def test_low_iou_is_not_a_match():
    gt = {"a": _gt((0, 0, 0, 10, 10))}
    pr = {"a": _pred((0, 6, 0, 16, 10, 0.9))}  # IoU = 4/16 = 0.25
    assert match(["a"], gt, pr, NC).tp.tolist() == [False]


def test_weights_equal_duplicating_images():
    rng = np.random.default_rng(0)
    gt, pr = {}, {}
    for s in "abcd":
        k = rng.integers(1, 4)
        boxes = rng.uniform(0, 50, (k, 2))
        g = [(int(rng.integers(0, NC)), x, y, x + 20, y + 20) for x, y in boxes]
        gt[s] = _gt(*g)
        pr[s] = _pred(
            *[(c, x0 + rng.uniform(-3, 3), y0, x1, y1, float(rng.uniform(0.1, 1))) for c, x0, y0, x1, y1 in g]
        )
    t = match(list("abcd"), gt, pr, NC)
    weighted = compute(t, 0.3, np.array([2.0, 1.0, 0.0, 1.0]))
    dup = compute(match(["a", "a2", "b", "d"], {**gt, "a2": gt["a"]}, {**pr, "a2": pr["a"]}, NC), 0.3)
    for k in ("mAP50", "precision", "recall", "F1"):
        assert weighted[k] == pytest.approx(dup[k])


def _random_table(seed=0, n_img=60):
    rng = np.random.default_rng(seed)
    gt, pr = {}, {}
    stems = [f"x_{i}" for i in range(n_img)]
    for s in stems:
        g = [(int(rng.integers(0, NC)), *(xy := rng.uniform(0, 100, 2)), *(xy + 20)) for _ in range(rng.integers(0, 4))]
        gt[s] = _gt(*g) if g else (np.zeros(0, int), np.zeros((0, 4)))
        p = [(c, x0 + rng.normal(0, 4), y0, x1, y1, float(rng.uniform(0, 1))) for c, x0, y0, x1, y1 in g]
        p += [(int(rng.integers(0, NC)), *(xy := rng.uniform(0, 100, 2)), *(xy + 20), float(rng.uniform(0, 0.6)))]
        pr[s] = _pred(*p)
    return match(stems, gt, pr, NC)


def test_bootstrap_ci_brackets_point_and_is_seeded():
    t = _random_table()
    r1, r2 = bootstrap(t, 0.3, n=200, seed=1), bootstrap(t, 0.3, n=200, seed=1)
    assert r1 == r2
    for k in ("mAP50", "F1"):
        v, lo, hi = r1[k]
        assert lo <= v <= hi and hi - lo > 0


def test_paired_diff_of_identical_models_is_zero():
    t = _random_table()
    assert paired_bootstrap_diff(t, t, 0.3, 0.3, n=100) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        paired_bootstrap_diff(t, _random_table(n_img=10), 0.3, 0.3)


def test_best_threshold_in_grid():
    assert 0.05 <= best_f1_threshold(_random_table()) <= 0.95


def test_loaders_roundtrip(tmp_path):
    (tmp_path / "labels" / "Japan").mkdir(parents=True)
    (tmp_path / "labels" / "Japan" / "Japan_000001.txt").write_text("2 0.5 0.5 0.2 0.4\n")
    (tmp_path / "labels" / "Japan" / "Japan_000002.txt").write_text("")
    sizes = {"Japan_000001": (100, 50), "Japan_000002": (100, 50)}
    gt = load_ground_truth(["Japan_000001", "Japan_000002"], tmp_path / "labels", sizes)
    assert gt["Japan_000001"][0].tolist() == [2] and gt["Japan_000001"][1].tolist() == [[40, 15, 60, 35]]
    assert len(gt["Japan_000002"][0]) == 0
    pj = tmp_path / "p.json"
    pj.write_text(json.dumps([{"image_id": "Japan_000001", "category_id": 3, "bbox": [40, 15, 20, 20], "score": 0.9}]))
    pr = load_predictions(pj)
    assert pr["Japan_000001"][0].tolist() == [2] and pr["Japan_000001"][1].tolist() == [[40, 15, 60, 35]]
    assert compute(match(list(sizes), gt, pr, NC), 0.5)["mAP50"] == pytest.approx(1.0)


def test_unknown_ap_method():
    with pytest.raises(ValueError):
        compute(_random_table(), 0.3, method="voc")


def test_report_end_to_end(raw_root, tmp_path, cfg):
    """Convertisseur -> prédictions parfaites au format Ultralytics -> rapport : mAP = F1 = 1 partout."""
    import csv as _csv

    from roadshift.data.convert import run
    from roadshift.eval.report import main

    data = tmp_path / "data"
    run(cfg, raw_root, data)
    sizes = load_sizes(data / "manifest.csv")
    runs = tmp_path / "runs"
    for lst in data.glob("list_*.txt"):
        tag = lst.stem.removeprefix("list_")
        if not (tag.startswith("source_") or tag.startswith("test_")) or tag == "source_train":
            continue
        preds = []
        for stem in lst.read_text().split():
            stem = stem.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            w, h = sizes[stem]
            for line in (data / "labels" / stem.rsplit("_", 1)[0] / f"{stem}.txt").read_text().split("\n"):
                if line:
                    c, cx, cy, bw, bh = map(float, line.split())
                    box = [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h]
                    preds.append({"image_id": stem, "category_id": int(c) + 1, "bbox": box, "score": 0.9})
        (runs / f"eval_{tag}").mkdir(parents=True)
        (runs / f"eval_{tag}" / "predictions.json").write_text(json.dumps(preds))
    out = tmp_path / "ci.csv"
    main(["--runs-dir", str(runs), "--data", str(data), "--model", "oracle", "--out", str(out), "--n-boot", "50"])
    rows = list(_csv.DictReader(out.read_text().splitlines()))
    assert {r["eval"] for r in rows} >= {"source_val", "source_test", "test_United_States", "test_Japan_pos"}
    for r in rows:
        assert float(r["mAP50"]) == pytest.approx(1.0) and float(r["F1"]) == pytest.approx(1.0)
