"""Inférence zero-shot (Grounding DINO, Florence-2) -> <out>/eval_<tag>/predictions.json, lisible par
roadshift.eval.report / compare comme un run Ultralytics.

Usage (GPU conseillé) :
    python -m roadshift.zeroshot.run --model grounding_dino --data <dataset converti> \
        --tags source_val source_test test_United_States --out runs_zeroshot/grounding_dino [--limit 50]

Prompts et réglages : configs/default.yaml, section zeroshot (fixés avant tout résultat).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from roadshift.eval.metrics import read_image_list
from roadshift.eval.report import tag_to_list
from roadshift.zeroshot.convert import (
    build_caption,
    class_scores,
    clip_boxes,
    florence_boxes,
    positive_map,
    select_detections,
    to_records,
)


def _batches(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _open(paths):
    from PIL import Image

    return [Image.open(p).convert("RGB") for p in paths]


def load_model(name: str, mcfg: dict, device: str):
    """(processor, modèle, révision du checkpoint) depuis le Hub Hugging Face."""
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Florence2ForConditionalGeneration

    proc = AutoProcessor.from_pretrained(mcfg["hf_id"])
    if name == "grounding_dino":
        model = AutoModelForZeroShotObjectDetection.from_pretrained(mcfg["hf_id"])
    elif name == "florence2":
        model = Florence2ForConditionalGeneration.from_pretrained(mcfg["hf_id"], dtype=torch.float16)
    else:
        raise ValueError(name)
    return proc, model.to(device).eval(), getattr(model.config, "_commit_hash", None)


def grounding_dino_predict(proc, model, paths: list[Path], phrases: list[str], mcfg: dict, device: str) -> list[dict]:
    import torch

    text, spans = build_caption(phrases)
    enc = proc.tokenizer(text, return_offsets_mapping=True)
    records = []
    for chunk in _batches(paths, int(mcfg["batch"])):
        imgs = _open(chunk)
        inputs = proc(images=imgs, text=[text] * len(imgs), return_tensors="pt")
        if not bool(inputs["pixel_mask"].all()) and len(imgs) > 1:  # tailles mélangées : image par image
            for p in chunk:
                records += grounding_dino_predict(proc, model, [p], phrases, {**mcfg, "batch": 1}, device)
            continue
        ids = inputs["input_ids"][0].tolist()
        if ids[: len(enc["input_ids"])] != enc["input_ids"]:
            raise RuntimeError("tokens du processeur différents de ceux de la positive map")
        with torch.no_grad():
            out = model(**inputs.to(device))
        prob = out.logits.sigmoid().float().cpu().numpy()  # [B, Q, T]
        pmap = positive_map(enc["offset_mapping"], spans, prob.shape[-1])
        boxes = out.pred_boxes.float().cpu().numpy()  # [B, Q, 4] cx cy w h normalisés
        for p, img, pr, bx in zip(chunk, imgs, prob, boxes, strict=True):
            cls, xyxy, sc = select_detections(bx, class_scores(pr, pmap), *img.size, mcfg["top_k"], mcfg["min_score"])
            records += to_records(p.stem, cls, xyxy, sc)
    return records


def florence2_predict(proc, model, paths: list[Path], phrases: list[str], mcfg: dict, device: str) -> list[dict]:
    """Une génération par image et par classe (tâche <OPEN_VOCABULARY_DETECTION>). Florence ne donne pas de score
    par boîte : chaque boîte reçoit exp(score de la séquence), commun à toutes les boîtes de cette génération."""
    import torch

    task, records = mcfg["task"], []
    dtype = next(model.parameters()).dtype
    for c, phrase in enumerate(phrases):
        for chunk in _batches(paths, int(mcfg["batch"])):
            imgs = _open(chunk)
            inputs = proc(text=[task + phrase] * len(imgs), images=imgs, return_tensors="pt").to(device, dtype)
            with torch.no_grad():
                gen = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                     max_new_tokens=mcfg["max_new_tokens"], num_beams=mcfg["num_beams"],
                                     do_sample=False, return_dict_in_generate=True, output_scores=True)  # fmt: skip
            texts = proc.batch_decode(gen.sequences, skip_special_tokens=False)
            seq = getattr(gen, "sequences_scores", None)
            scores = np.exp(seq.float().cpu().numpy()) if seq is not None else np.ones(len(imgs))
            for p, img, t, s in zip(chunk, imgs, texts, scores, strict=True):
                parsed = proc.post_process_generation(text=t, task=task, image_size=img.size)[task]
                xyxy = clip_boxes(florence_boxes(parsed), *img.size)
                records += to_records(p.stem, np.full(len(xyxy), c), xyxy, np.full(len(xyxy), s))
    return records


PREDICT = {"grounding_dino": grounding_dino_predict, "florence2": florence2_predict}


def run(name: str, cfg: dict, data: Path, tags: list[str], out: Path, device: str, limit: int | None = None,
        loaded=None) -> dict:  # fmt: skip
    """Écrit out/eval_<tag>/predictions.json ; renvoie les durées et la révision du modèle."""
    zs = cfg["zeroshot"]
    mcfg = zs["models"][name]
    phrases = [zs["prompts"][c] for c in cfg["data"]["classes"]]
    proc, model, revision = loaded or load_model(name, mcfg, device)
    info = {"model": name, "hf_id": mcfg["hf_id"], "revision": revision, "phrases": phrases, "tags": {}}
    for tag in tags:
        stems = read_image_list(data / tag_to_list(tag))[:limit]
        paths = [next((data / "images" / s.rsplit("_", 1)[0]).glob(f"{s}.*")) for s in stems]
        t0 = time.time()
        records = PREDICT[name](proc, model, paths, phrases, mcfg, device)
        d = out / f"eval_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "predictions.json").write_text(json.dumps(records))
        info["tags"][tag] = {"n_images": len(paths), "n_boxes": len(records),
                             "s_per_image": round((time.time() - t0) / max(1, len(paths)), 4)}  # fmt: skip
        print(f"{name} {tag}: {len(paths)} images, {len(records)} boîtes, {info['tags'][tag]['s_per_image']} s/image")
    (out / "zeroshot_info.json").write_text(json.dumps(info, indent=2))
    return info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(PREDICT))
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--tags", nargs="+", default=["source_val", "source_test", "test_United_States"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="n premières images par jeu (essai rapide)")
    a = ap.parse_args(argv)
    run(a.model, yaml.safe_load(a.config.read_text()), a.data, a.tags, a.out, a.device, a.limit)


if __name__ == "__main__":
    main()
