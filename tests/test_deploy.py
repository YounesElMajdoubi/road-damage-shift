"""Quantification et latence, testées sur un petit réseau ONNX construit à la main (sans torch ni Ultralytics)."""

import numpy as np
import pytest
from PIL import Image

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from roadshift.deploy.bench import bench  # noqa: E402
from roadshift.deploy.quantize import (  # noqa: E402
    calibration_images,
    calibration_mix,
    copy_metadata,
    letterbox,
    quantize_int8,
    quantized_convs,
)

SIZE = 32


def _tiny_model(path):
    """Conv(3->8) -> Relu -> Conv(8->4) -> Sigmoid, entrée 1x3x32x32, avec métadonnées façon Ultralytics."""
    rng = np.random.default_rng(0)
    w1 = numpy_helper.from_array(rng.normal(0, 0.3, (8, 3, 3, 3)).astype(np.float32), "w1")
    w2 = numpy_helper.from_array(rng.normal(0, 0.3, (4, 8, 1, 1)).astype(np.float32), "w2")
    nodes = [
        helper.make_node("Conv", ["images", "w1"], ["c1"], name="/model.0/conv/Conv", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"], name="/model.0/act/Relu"),
        helper.make_node("Conv", ["r1", "w2"], ["c2"], name="/model.23/cv2/Conv"),
        helper.make_node("Sigmoid", ["c2"], ["output0"]),
    ]
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, SIZE, SIZE])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4, SIZE, SIZE])
    g = helper.make_graph(nodes, "tiny", [inp], [out], [w1, w2])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    m.metadata_props.add(key="names", value="{0: 'D00', 1: 'D10', 2: 'D20', 3: 'D40'}")
    m.metadata_props.add(key="stride", value="32")
    onnx.save(m, str(path))
    return path


def _images(tmp_path, n=8):
    rng = np.random.default_rng(1)
    paths = []
    for i in range(n):
        p = tmp_path / f"img_{i}.jpg"
        Image.fromarray(rng.integers(0, 256, (24, 40, 3), dtype=np.uint8)).save(p)
        paths.append(p)
    return paths


def test_letterbox_shape_padding_and_range():
    img = np.full((24, 40, 3), 200, np.uint8)
    x = letterbox(img, SIZE)
    assert x.shape == (1, 3, SIZE, SIZE) and x.dtype == np.float32 and x.min() >= 0 and x.max() <= 1
    # 40x24 -> 32x19 (ratio 0.8) centré : bandes grises 114 en haut et en bas
    assert np.allclose(x[0, :, 0, :], 114 / 255) and np.allclose(x[0, :, SIZE // 2, :], 200 / 255)


def test_calibration_images_from_train_only(raw_root, tmp_path, cfg):
    from roadshift.data.convert import run

    data = tmp_path / "data"
    run(cfg, raw_root, data)
    imgs = calibration_images(data, 10, seed=42)
    train = {str(data / x[2:]) for x in (data / "list_source_train.txt").read_text().split()}
    assert len(imgs) == 10 and all(str(p) in train for p in imgs)
    assert imgs == calibration_images(data, 10, seed=42)


def test_quantize_int8_close_to_fp32_and_keeps_metadata(tmp_path):
    fp32 = _tiny_model(tmp_path / "fp32.onnx")
    cfg = {"per_channel": True, "activation_type": "QUInt8", "weight_type": "QInt8", "op_types": ["Conv"],
           "calibrate_method": "MinMax"}  # fmt: skip
    int8 = quantize_int8(fp32, tmp_path / "int8.onnx", _images(tmp_path), cfg, SIZE)
    m = onnx.load(str(int8))
    ops = {n.op_type for n in m.graph.node}
    assert "QuantizeLinear" in ops and "DequantizeLinear" in ops  # format QDQ
    assert {p.key for p in m.metadata_props} == {"names", "stride"}  # métadonnées Ultralytics recopiées
    x = letterbox(np.asarray(Image.open(_images(tmp_path, 1)[0])), SIZE)
    y32 = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]).run(None, {"images": x})[0]
    y8 = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"]).run(None, {"images": x})[0]
    assert y8.shape == y32.shape and np.abs(y8 - y32).max() < 0.05


def test_copy_metadata_replaces(tmp_path):
    a = _tiny_model(tmp_path / "a.onnx")
    b = _tiny_model(tmp_path / "b.onnx")
    mb = onnx.load(str(b))
    del mb.metadata_props[:]
    mb.metadata_props.add(key="old", value="x")
    onnx.save(mb, str(b))
    copy_metadata(a, b)
    assert {p.key for p in onnx.load(str(b)).metadata_props} == {"names", "stride"}


def test_bench_onnx(tmp_path):
    rows = bench([_tiny_model(tmp_path / "m.onnx")], imgsz=SIZE, threads=1, warmup=2, runs=5)
    r = rows[0]
    assert r["model"] == "m.onnx" and r["runs"] == 5 and 0 < r["ms_median"] <= r["ms_p90"] + 1e-9
    assert r["backend"].startswith("onnxruntime") and r["size_bytes"] > 0


def test_quantize_excludes_head_by_name(tmp_path):
    fp32 = _tiny_model(tmp_path / "fp32.onnx")
    cfg = {"per_channel": True, "activation_type": "QUInt8", "weight_type": "QInt8", "op_types": ["Conv"],
           "calibrate_method": "MinMax"}  # fmt: skip
    head = {**cfg, "exclude_name_contains": ["/model.23/"]}
    q_all = quantize_int8(fp32, tmp_path / "all.onnx", _images(tmp_path), cfg, SIZE)
    q_head = quantize_int8(fp32, tmp_path / "nohead.onnx", _images(tmp_path), head, SIZE)
    assert quantized_convs(q_all)[:2] == (2, 2)
    n, q, fp = quantized_convs(q_head)
    assert (n, q) == (2, 1) and fp == ["/model.23/cv2/Conv"]
    bad = {**cfg, "exclude_name_contains": ["/model.99/"]}
    with pytest.raises(ValueError, match="aucun nœud"):  # motif mal écrit : erreur, pas une copie silencieuse de Q3
        quantize_int8(fp32, tmp_path / "x.onnx", _images(tmp_path), bad, SIZE)


def test_calibration_mix_counts_and_refuses_eval_lists(tmp_path):
    for name, n in (("list_source_train.txt", 30), ("list_pool_United_States.txt", 20)):
        (tmp_path / name).write_text("\n".join(f"./images/X/{name[5:9]}_{i}.jpg" for i in range(n)))
    spec = {"list_source_train.txt": 5, "list_pool_United_States.txt": 4}
    imgs = calibration_mix(tmp_path, spec, seed=42)
    assert len(imgs) == 9 and len(set(imgs)) == 9
    assert sum(p.name.startswith("pool") for p in imgs) == 4
    assert imgs == calibration_mix(tmp_path, spec, seed=42)  # reproductible
    for forbidden in ("list_test_United_States.txt", "list_source_val.txt"):
        with pytest.raises(ValueError):
            calibration_mix(tmp_path, {forbidden: 5}, seed=42)
