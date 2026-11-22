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
