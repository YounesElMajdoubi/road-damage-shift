from pathlib import Path

import numpy as np
import pytest

from roadshift.blur.ops import blur_image, image_rng, motion_kernel, psnr, wiener_deconv


@pytest.mark.parametrize("length", [2, 9, 15, 31])
@pytest.mark.parametrize("angle", [0, 45, 90])
def test_kernel_normalized_odd_and_oriented(length, angle):
    k = motion_kernel(length, angle)
    assert k.shape[0] == k.shape[1] and k.shape[0] % 2 == 1 and k.sum() == pytest.approx(1.0)
    c = k.shape[0] // 2
    assert k[c, c] > 0  # le segment passe par le centre
    if angle == 0:
        assert np.count_nonzero(k[c]) == np.count_nonzero(k)  # horizontal : une seule ligne
    if angle == 90:
        assert np.count_nonzero(k[:, c]) == np.count_nonzero(k)  # vertical : une seule colonne


def test_identity_kernel_and_no_noise_is_noop():
    img = np.random.default_rng(0).integers(0, 256, (20, 30, 3), dtype=np.uint8)
    assert motion_kernel(0, 45).shape == (1, 1)
    np.testing.assert_array_equal(blur_image(img, motion_kernel(0, 0), 0.0, image_rng("x", 0)), img)


def test_blur_preserves_constant_image_and_smooths():
    flat = np.full((32, 32, 3), 120, np.uint8)
    np.testing.assert_array_equal(blur_image(flat, motion_kernel(15, 45), 0.0, image_rng("x", 0)), flat)
    img = np.zeros((32, 32, 3), np.uint8)
    img[:, 16:] = 255  # bord vertical net
    out = blur_image(img, motion_kernel(9, 0), 0.0, image_rng("x", 0))  # bougé horizontal : bord adouci
    assert 0 < out[16, 14, 0] < 255 and out.shape == img.shape and out.dtype == np.uint8


def test_rng_is_per_image_and_seeded():
    a = image_rng("Japan_000001", 42).normal(size=3)
    assert np.array_equal(a, image_rng("Japan_000001", 42).normal(size=3))
    assert not np.array_equal(a, image_rng("Japan_000002", 42).normal(size=3))
    assert not np.array_equal(a, image_rng("Japan_000001", 7).normal(size=3))


def _scene(n=96):
    yy, xx = np.mgrid[0:n, 0:n]
    g = 128 + 60 * np.sin(xx / 6.0) * np.cos(yy / 9.0)
    g[30:60, 20:70] = 230  # objet net
    g[70:74, :] = 20  # « fissure »
    return np.repeat(np.clip(g, 0, 255)[..., None], 3, axis=2).astype(np.uint8)


@pytest.mark.parametrize("angle", [0, 45, 90])
def test_wiener_recovers_detail_with_known_psf(angle):
    img = _scene()
    k = motion_kernel(15, angle)
    blurred = blur_image(img, k, 0.005, image_rng("s", 0))
    restored = wiener_deconv(blurred, k, 0.01)
    assert restored.shape == img.shape and restored.dtype == np.uint8
    assert psnr(restored, img) > psnr(blurred, img) + 1.0  # au moins +1 dB


def test_wiener_identity_kernel_is_noop():
    img = _scene()
    np.testing.assert_array_equal(wiener_deconv(img, motion_kernel(0, 0), 0.01), img)


def _stem(line):
    return line.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def _ultra_label(img: Path) -> Path:
    """Règle d'Ultralytics : remplacer le dernier /images/ par /labels/ et l'extension par .txt."""
    head, tail = img.as_posix().rsplit("/images/", 1)  # as_posix : séparateurs "/" aussi sous Windows
    return Path(f"{head}/labels/{tail}").with_suffix(".txt")


def test_build_blur_sets(raw_root, tmp_path, cfg):
    import csv

    import yaml
    from PIL import Image

    from roadshift.blur.build import build, image_angle
    from roadshift.data.convert import run

    data = tmp_path / "data"
    run(cfg, raw_root, data)
    cfg["blur"] = {"kernel_lengths": [0, 9, 15], "angles_deg": [0, 45, 90], "noise_sigma": 0.01, "jpeg_quality": 95,
                   "eval_sets": ["source_test", "test_United_States"], "seed": 42, "wiener_k": 0.01}  # fmt: skip
    info = build(cfg, data, workers=2)
    orig = (data / "list_source_test.txt").read_text().split()
    n_us = len((data / "list_test_United_States.txt").read_text().split())
    for kind in ("blur", "wiener"):
        lst = (data / f"list_{kind}_L9_source_test.txt").read_text().split()
        assert [_stem(x) for x in lst] == [_stem(x) for x in orig]  # mêmes images, même ordre : appariement
        for img_rel, o in zip(lst, orig, strict=True):
            img = data / img_rel[2:]
            assert img.exists()
            assert _ultra_label(img).read_text() == _ultra_label(data / o[2:]).read_text()
        y = yaml.safe_load((data / f"data_{kind}_L9_source_test.yaml").read_text())
        assert y["val"] == f"list_{kind}_L9_source_test.txt"
    assert (data / "list_blur_L0_source_test.txt").read_text().split() == orig
    first = Path(orig[0][2:]).with_suffix(".jpg")
    b = np.asarray(Image.open(data / "blur" / "L15" / first))
    w = np.asarray(Image.open(data / "wiener" / "L15" / first))
    assert b.shape == w.shape and not np.array_equal(b, w)
    m = list(csv.DictReader((data / "blur_manifest.csv").read_text().splitlines()))
    assert len(m) == info["n_images_processed"] == 2 * (len(orig) + n_us)  # 2 longueurs > 0
    angles = {}
    for r in m:
        angles.setdefault(r["stem"], set()).add(float(r["angle"]))
    assert all(len(v) == 1 for v in angles.values())  # même angle à toutes les longueurs
    assert all(float(r["angle"]) == image_angle(r["stem"], cfg["blur"]) for r in m)
    assert info["wiener_ms_median"] > 0


def _oracle_preds(data, tag, sizes):
    preds = []
    for line in (data / f"list_{tag}.txt").read_text().split():
        stem = _stem(line)
        w, h = sizes[stem]
        for lab in (data / "labels" / stem.rsplit("_", 1)[0] / f"{stem}.txt").read_text().split("\n"):
            if lab:
                c, cx, cy, bw, bh = map(float, lab.split())
                box = [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h]
                preds.append({"image_id": stem, "category_id": int(c) + 1, "bbox": box, "score": 0.9})
    return preds


def test_blur_report_oracles(raw_root, tmp_path, cfg):
    import json

    from roadshift.blur.build import build
    from roadshift.blur.report import main
    from roadshift.data.convert import run
    from roadshift.eval.metrics import load_sizes

    data = tmp_path / "data"
    run(cfg, raw_root, data)
    cfg["blur"] = {"kernel_lengths": [0, 9], "angles_deg": [0, 90], "noise_sigma": 0.01, "jpeg_quality": 95,
                   "eval_sets": ["source_test", "test_United_States"], "seed": 42, "wiener_k": 0.01}  # fmt: skip
    build(cfg, data, workers=2)
    sizes = load_sizes(data / "manifest.csv")
    base, aug = tmp_path / "base", tmp_path / "aug"
    tags = ["source_val"] + [f"{k}_L{L}_{s}" for s in cfg["blur"]["eval_sets"] for k, L in
                             [("blur", 0), ("blur", 9), ("wiener", 9)]]  # fmt: skip
    for run_dir, run_tags in ((base, tags), (aug, [t for t in tags if not t.startswith("wiener")])):
        for t in run_tags:
            (run_dir / f"eval_{t}").mkdir(parents=True)
            (run_dir / f"eval_{t}" / "predictions.json").write_text(json.dumps(_oracle_preds(data, t, sizes)))
    out = tmp_path / "res"
    main(["--base", str(base), "--aug", str(aug), "--data", str(data), "--out-dir", str(out), "--n-boot", "30"])
    import csv as _csv

    scores = list(_csv.DictReader((out / "blur_scores.csv").read_text().splitlines()))
    assert {(r["method"], r["eval_set"], r["blur_length"]) for r in scores} == {
        (m, s, L) for m in ("none", "wiener", "aug") for s in cfg["blur"]["eval_sets"] for L in ("0", "9")
    }
    assert all(float(r["mAP50"]) == 1.0 and float(r["F1"]) == 1.0 for r in scores)
    gains = list(_csv.DictReader((out / "blur_gains.csv").read_text().splitlines()))
    # wiener - none à L>0 seulement ; aug - none à toutes les longueurs (dont l'effet sur images nettes)
    assert {(g["comparison"], g["blur_length"]) for g in gains} == {("wiener - none", "9"), ("aug - none", "0"),
                                                                    ("aug - none", "9")}  # fmt: skip
    assert all(float(g["diff"]) == 0 and g["significant"] == "False" for g in gains)
    assert (out / "blur_curve.png").read_bytes()[:4] == b"\x89PNG"
