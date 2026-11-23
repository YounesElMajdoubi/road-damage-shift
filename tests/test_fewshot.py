import numpy as np
import pytest
import yaml

from roadshift.data.convert import run
from roadshift.fewshot.plan import (
    FewShotRun,
    aggregate_curve,
    annotation_cost_hours,
    count_annotations,
    finetune_lr0,
    plan_runs,
    write_fewshot_yaml,
)

FS = {"target": "United_States", "sizes": [10, 5], "seeds": [0, 1], "include_full_pool": True}
COST = {"s_per_image": 7.8, "s_per_box_high": 34.5, "s_per_box_low": 7.0}


def test_plan_runs_order_and_content():
    runs = plan_runs(FS, pool_size=25)
    assert [r.name for r in runs] == [
        "United_States_k0",
        "United_States_k5_s0", "United_States_k5_s1",
        "United_States_k10_s0", "United_States_k10_s1",
        "United_States_full",
    ]  # fmt: skip
    assert runs[0].list_name is None and runs[-1].k == 25 and runs[-1].list_name == "list_pool_United_States.txt"
    assert runs[1].list_name == "list_fewshot_United_States_k5_s0.txt"
    with pytest.raises(ValueError):
        plan_runs({**FS, "sizes": [50]}, pool_size=25)


def test_cost_formula():
    lo, hi = annotation_cost_hours(50, 100, COST)
    assert lo == pytest.approx((7.8 * 50 + 7.0 * 100) / 3600)
    assert hi == pytest.approx((7.8 * 50 + 34.5 * 100) / 3600)
    assert annotation_cost_hours(0, 0, COST) == (0.0, 0.0)


def test_finetune_lr0():
    assert finetune_lr0({"weights": "yolo11s.pt"}, 0.1, nc=4) == pytest.approx(0.000125)  # auto S1 = 0.00125
    assert finetune_lr0({"lr0": 0.0001}, 0.1, nc=4) == pytest.approx(0.00001)


def test_yaml_and_counts_on_converted_data(raw_root, tmp_path, cfg):
    data = tmp_path / "data"
    run(cfg, raw_root, data)
    r = FewShotRun("United_States_k5_s0", 5, 0, "list_fewshot_United_States_k5_s0.txt")
    d = yaml.safe_load(write_fewshot_yaml(data, r, cfg["data"]["classes"]).read_text())
    assert d["train"] == r.list_name and d["val"] == "list_source_val.txt" and "path" not in d
    assert (data / d["train"]).exists() and (data / d["val"]).exists()
    c = count_annotations(data / r.list_name, data / "labels", cfg["data"]["classes"])
    assert c["n_images"] == 5 and c["n_boxes"] == sum(c[f"n_{k}"] for k in cfg["data"]["classes"])
    # Cohérence avec les labels écrits par le convertisseur
    n = sum(
        len(
            (data / "labels" / "United_States" / f"{x.rsplit('/', 1)[-1].rsplit('.', 1)[0]}.txt")
            .read_text()
            .split("\n")
        )
        - 1
        for x in (data / r.list_name).read_text().split()
    )
    assert c["n_boxes"] == n
    with pytest.raises(ValueError):
        write_fewshot_yaml(data, FewShotRun("x_k0", 0, None, None), cfg["data"]["classes"])


def test_aggregate_curve():
    rows = [
        {"model": "m", "eval": "t", "k": 50, "seed": s, "mAP50": v, "mAP50_lo": v - 0.1, "mAP50_hi": v + 0.1}
        for s, v in [(0, 0.4), (1, 0.5), (2, 0.6)]
    ] + [{"model": "m", "eval": "t", "k": 0, "seed": None, "mAP50": 0.3, "mAP50_lo": 0.2, "mAP50_hi": 0.35}]
    out = {r["k"]: r for r in aggregate_curve(rows)}
    assert out[50]["mAP50"] == pytest.approx(0.5) and out[50]["mAP50_seed_sd"] == pytest.approx(0.1)
    assert (out[50]["mAP50_seed_min"], out[50]["mAP50_seed_max"]) == (0.4, 0.6)
    assert out[50]["mAP50_boot_lo"] == pytest.approx(0.4) and out[0]["n_seeds"] == 1 and out[0]["mAP50_seed_sd"] == 0


def test_plot_curve_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from roadshift.fewshot.plot import plot_curve

    rows = [
        {"model": "yolo11s", "eval": ev, "k": k, "seed": s, "mAP50": 0.4, "mAP50_lo": 0.35, "mAP50_hi": 0.45}
        for ev in ("test_United_States", "source_test")
        for k, s in [(0, None), (50, 0), (50, 1), (100, None)]
    ]
    out = plot_curve(
        aggregate_curve(rows), rows, {50: (0.3, 1.0), 100: (0.6, 2.0)},
        [("test_United_States", "cible"), ("source_test", "oubli")], tmp_path / "c.png", full_k=100,
    )  # fmt: skip
    assert out.exists() and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_paired_gains_and_pairing():
    from roadshift.eval.metrics import match
    from roadshift.fewshot.gains import paired_gains, pairs_vs_k0, pairs_vs_reference

    gt = {"a": (np.array([0]), np.array([[0, 0, 10, 10.0]])), "b": (np.array([1]), np.array([[5, 5, 20, 20.0]]))}
    pr = {s: (g[0], g[1], np.array([0.9])) for s, g in gt.items()}
    t = match(["a", "b"], gt, pr, 4)
    runs = {"k0": {"k": 0, "seed": None}, "k50_s1": {"k": 50, "seed": 1}, "k50_s0": {"k": 50, "seed": 0}}
    assert pairs_vs_k0(runs) == [("k50_s0", "k0"), ("k50_s1", "k0")]
    ref = {"ref:k0": {"k": 0, "seed": None}, "ref:x": {"k": 50, "seed": 0}, "ref:y": {"k": 50, "seed": 1}}
    assert pairs_vs_reference(runs, ref) == [("k50_s0", "ref:x"), ("k50_s1", "ref:y")]
    with pytest.raises(KeyError):
        pairs_vs_reference({"k9": {"k": 9, "seed": 0}}, ref)
    allr = {**runs, **ref}
    rows = paired_gains(
        [("k50_s0", "k0")], allr, {(n, "test"): t for n in allr}, dict.fromkeys(allr, 0.5), ["test"], 50
    )
    assert [r["metric"] for r in rows] == ["mAP50", "F1"]
    assert all(r["gain"] == 0 and not r["significant"] and r["reference"] == "k0" for r in rows)


def test_replay_list(raw_root, tmp_path, cfg):
    from roadshift.fewshot.plan import write_replay_list

    data = tmp_path / "data"
    run(cfg, raw_root, data)
    r = FewShotRun("United_States_k5_s0", 5, 0, "list_fewshot_United_States_k5_s0.txt")
    name, n = write_replay_list(data, r, ratio=1.0, seed=0)
    lines = (data / name).read_text().split()
    target = (data / r.list_name).read_text().split()
    source_train = set((data / "list_source_train.txt").read_text().split())
    assert n == 5 and len(lines) == 10 and lines[:5] == target
    assert set(lines[5:]) <= source_train  # jamais val/test source
    assert write_replay_list(data, r, 1.0, 0)[0] == name and (data / name).read_text().split() == lines  # seedé
    write_replay_list(data, r, 1.0, seed=1)
    assert (data / name).read_text().split()[5:] != lines[5:]  # autre seed : autres images sources
    y = yaml.safe_load(write_fewshot_yaml(data, r, cfg["data"]["classes"], train_list=name).read_text())
    assert y["train"] == name and y["val"] == "list_source_val.txt"
    with pytest.raises(ValueError):
        write_replay_list(data, r, ratio=1000.0, seed=0)


def test_plot_overlay_cli(tmp_path):
    pytest.importorskip("matplotlib")
    import csv as _csv

    from roadshift.fewshot.plot import main

    fields = ["model", "eval", "k", "seed", "mAP50", "mAP50_lo", "mAP50_hi"]
    for name, v in (("a", 0.4), ("b", 0.45)):
        with open(tmp_path / f"{name}.csv", "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for ev in ("test_United_States", "source_test"):
                for k, s in [(0, ""), (50, 0), (50, 1), (100, "")]:
                    w.writerow({"model": "yolo11s", "eval": ev, "k": k, "seed": s, "mAP50": v, "mAP50_lo": v - 0.05,
                                "mAP50_hi": v + 0.05})  # fmt: skip
    (tmp_path / "cost.csv").write_text(
        "run,k,seed,h_low,h_high\nk0,0,,0,0\nk50a,50,0,0.3,1.0\nk50b,50,1,0.4,1.2\nf,100,,1,3\n"
    )
    out = tmp_path / "o.png"
    a, b, c = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "cost.csv"
    main(["--runs", f"A={a}", "--runs", f"B={b}", "--cost", str(c), "--out", str(out)])
    assert out.read_bytes()[:4] == b"\x89PNG"
    with pytest.raises(SystemExit):
        main(["--runs", "A=x", "--runs", "B=x", "--runs", "C=x", "--cost", "c", "--out", str(out)])
