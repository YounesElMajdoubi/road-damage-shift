"""Quantification INT8 statique d'un modèle ONNX (onnxruntime, format QDQ), avec calibration sur images réelles.

- `letterbox` reproduit le prétraitement d'Ultralytics pour une entrée carrée fixe : redimensionnement en
  gardant les proportions, bandes grises (114) centrées, RGB, [0, 1], CHW.
- Les métadonnées d'Ultralytics (classes, stride, imgsz) sont recopiées du modèle FP32 vers le modèle INT8,
  sinon Ultralytics ne sait plus l'évaluer.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image


def letterbox(img: np.ndarray, size: int = 640, fill: int = 114) -> np.ndarray:
    """img uint8 HxWx3 (RGB) -> float32 1x3xSxS dans [0, 1]."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = (
        np.asarray(Image.fromarray(img).resize((nw, nh), Image.Resampling.BILINEAR)) if (nh, nw) != (h, w) else img
    )
    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def calibration_images(data_root: Path, n: int, seed: int, list_name: str = "list_source_train.txt") -> list[Path]:
    """n images tirées d'une liste Ultralytics (par défaut le TRAIN source), reproductible. Aucun label n'est lu."""
    lines = sorted((Path(data_root) / list_name).read_text().split())
    return [Path(data_root) / x[2:] for x in random.Random(seed).sample(lines, min(n, len(lines)))]


def calibration_mix(data_root: Path, spec: dict[str, int], seed: int) -> list[Path]:
    """Calibration mêlant plusieurs listes, ex. {"list_source_train.txt": 100, "list_pool_United_States.txt": 100}.

    Les listes de test sont refusées : la calibration ne doit jamais voir une image évaluée."""
    out: list[Path] = []
    for name, n in spec.items():
        if "test" in name or "val" in name:
            raise ValueError(f"liste interdite pour la calibration : {name}")
        out += calibration_images(data_root, n, seed, name)
    return out


def excluded_nodes(model_path: Path, op_types: list[str], name_contains: list[str]) -> list[str]:
    """Noms des nœuds des types quantifiés dont le nom contient un des motifs (ex. "/model.23/" = tête YOLO11)."""
    import onnx

    if not name_contains:
        return []
    g = onnx.load(str(model_path), load_external_data=False).graph
    return [n.name for n in g.node if n.op_type in op_types and any(s in n.name for s in name_contains)]


class LetterboxReader:
    """CalibrationDataReader d'onnxruntime : une image prétraitée par appel."""

    def __init__(self, images: list[Path], input_name: str, size: int):
        self._it = iter(images)
        self.input_name, self.size = input_name, size

    def get_next(self):
        p = next(self._it, None)
        if p is None:
            return None
        return {self.input_name: letterbox(np.asarray(Image.open(p).convert("RGB")), self.size)}


def copy_metadata(src: Path, dst: Path) -> None:
    import onnx

    m_src, m_dst = onnx.load(str(src)), onnx.load(str(dst))
    keep = {p.key: p.value for p in m_src.metadata_props}
    del m_dst.metadata_props[:]
    for k, v in keep.items():
        m_dst.metadata_props.add(key=k, value=v)
    onnx.save(m_dst, str(dst))


def quantize_int8(fp32: Path, out: Path, images: list[Path], cfg: dict, size: int) -> Path:
    """Quantification statique QDQ ; seuls les types d'opérations de cfg['op_types'] sont quantifiés.

    cfg['exclude_name_contains'] (optionnel) : les nœuds dont le nom contient un de ces motifs restent en FP32.
    Les noms sont lus sur le modèle prétraité, celui qui est réellement quantifié."""
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = Path(out).with_suffix(".pre.onnx")
    quant_pre_process(str(fp32), str(pre))
    input_name = ort.InferenceSession(str(pre), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    exclude = excluded_nodes(pre, list(cfg["op_types"]), list(cfg.get("exclude_name_contains") or []))
    if cfg.get("exclude_name_contains") and not exclude:
        raise ValueError(f"aucun nœud ne correspond à {cfg['exclude_name_contains']}")
    quantize_static(
        str(pre),
        str(out),
        LetterboxReader(images, input_name, size),
        quant_format=QuantFormat.QDQ,
        per_channel=cfg["per_channel"],
        activation_type=getattr(QuantType, cfg["activation_type"]),
        weight_type=getattr(QuantType, cfg["weight_type"]),
        op_types_to_quantize=list(cfg["op_types"]),
        calibrate_method=getattr(CalibrationMethod, cfg["calibrate_method"]),
        nodes_to_exclude=exclude,
    )
    pre.unlink(missing_ok=True)
    copy_metadata(fp32, out)
    return Path(out)


def quantized_convs(model_path: Path) -> tuple[int, int, list[str]]:
    """(nombre de Conv, nombre de Conv dont les poids passent par DequantizeLinear, noms des Conv restées FP32)."""
    import onnx

    g = onnx.load(str(model_path), load_external_data=False).graph
    prod = {o: n for n in g.node for o in n.output}
    convs = [n for n in g.node if n.op_type == "Conv"]
    fp32 = [n.name for n in convs if (prod.get(n.input[1]) is None or prod[n.input[1]].op_type != "DequantizeLinear")]
    return len(convs), len(convs) - len(fp32), fp32
