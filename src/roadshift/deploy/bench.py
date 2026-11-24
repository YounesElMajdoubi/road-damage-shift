"""Latence CPU du modèle seul (sans pré/post-traitement), batch 1, entrée fixe.

Usage (aussi sur ton propre PC, pour comparer au CPU de Kaggle) :
    python -m roadshift.deploy.bench --models best.pt model_fp32.onnx model_int8.onnx --threads 4 \
        --out results/latency_<machine>.csv

Protocole : même tenseur aléatoire seedé pour tous les modèles, `warmup` passes ignorées, puis `runs` passes
chronométrées une à une ; on rapporte la médiane et le 90e centile (plus robustes que la moyenne).
"""

from __future__ import annotations

import argparse
import csv
import platform
import time
from pathlib import Path

import numpy as np


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _onnx_runner(path: Path, threads: int):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return lambda x: sess.run(None, {name: x}), f"onnxruntime {ort.__version__}"


def _torch_runner(path: Path, threads: int):
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(threads)
    model = YOLO(str(path)).model.float().eval()

    def run(x):
        with torch.inference_mode():
            return model(torch.from_numpy(x))

    return run, f"torch {torch.__version__}"


def measure(run, x: np.ndarray, warmup: int, runs: int) -> dict:
    for _ in range(warmup):
        run(x)
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        run(x)
        t.append((time.perf_counter() - t0) * 1000)
    t = np.array(t)
    return {"ms_median": round(float(np.median(t)), 2), "ms_p90": round(float(np.quantile(t, 0.9)), 2),
            "ms_mean": round(float(t.mean()), 2), "fps_median": round(1000 / float(np.median(t)), 1)}  # fmt: skip


def bench(models: list[Path], imgsz: int = 640, threads: int = 4, warmup: int = 20, runs: int = 200, seed: int = 0):
    import os
    import warnings

    if threads > (os.cpu_count() or threads):
        warnings.warn(f"{threads} threads demandés pour {os.cpu_count()} cœurs : latences pessimistes", stacklevel=2)
    x = np.random.default_rng(seed).random((1, 3, imgsz, imgsz), dtype=np.float32)
    rows = []
    for p in models:
        p = Path(p)
        run, backend = (_torch_runner if p.suffix == ".pt" else _onnx_runner)(p, threads)
        size = p.stat().st_size
        rows.append({"model": p.name, "backend": backend, "size_MB": round(size / 1e6, 2), "size_bytes": size,
                     "threads": threads, "imgsz": imgsz, "warmup": warmup, "runs": runs,
                     **measure(run, x, warmup, runs), "cpu": cpu_name(), "os": platform.platform()})  # fmt: skip
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    rows = bench(args.models, args.imgsz, args.threads, args.warmup, args.runs)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="\n", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    for r in rows:
        print(f"{r['model']:28s} {r['size_MB']:7.1f} Mo  médiane {r['ms_median']:8.1f} ms  p90 {r['ms_p90']:8.1f} ms  "
              f"({r['fps_median']} img/s, {r['threads']} threads, {r['cpu']})")  # fmt: skip


if __name__ == "__main__":
    main()
