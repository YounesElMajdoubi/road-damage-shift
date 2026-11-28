import json

import numpy as np
import pytest

from roadshift.eval.metrics import load_predictions, nms, to_agnostic
from roadshift.zeroshot.convert import (
    build_caption,
    class_scores,
    florence_boxes,
    positive_map,
    select_detections,
    to_records,
)

PHRASES = ["longitudinal crack", "transverse crack", "alligator crack", "pothole"]


def test_build_caption_spans_point_to_each_phrase():
    text, spans = build_caption(PHRASES)
    assert text == "longitudinal crack . transverse crack . alligator crack . pothole ."
    assert [text[a:b] for a, b in spans] == PHRASES
    with pytest.raises(ValueError):
        build_caption(["a . b"])


def test_positive_map_averages_the_tokens_of_each_phrase():
    text, spans = build_caption(["alligator crack", "pothole"])  # "alligator crack . pothole ."
    # [CLS] alligator crack . pot ##hole . [SEP], puis padding
    offsets = [(0, 0), (0, 9), (10, 15), (16, 17), (18, 21), (21, 25), (26, 27), (0, 0)]
    m = positive_map(offsets, spans, n_tokens=10)
    assert m.shape == (2, 10)
    np.testing.assert_allclose(m[0, [1, 2]], 0.5)
    np.testing.assert_allclose(m[1, [4, 5]], 0.5)
    assert m[:, [0, 3, 6, 7, 8, 9]].sum() == 0  # ni tokens spéciaux, ni « . », ni padding
    prob = np.zeros((3, 10))
    prob[0, [1, 2]] = [0.8, 0.6]  # requête 0 : alligator crack
    prob[1, [4, 5]] = [0.9, 0.1]  # requête 1 : pothole (moyenne 0,5)
    np.testing.assert_allclose(class_scores(prob, m), [[0.7, 0], [0, 0.5], [0, 0]])
    with pytest.raises(ValueError):
        class_scores(prob[:, :9], m)


def test_select_detections_pixels_topk_threshold_and_clipping():
    boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.95, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]])
    scores = np.array([[0.1, 0.6], [0.9, 0.2], [0.005, 0.001]])
    cls, xyxy, sc = select_detections(boxes, scores, w=200, h=100, top_k=5, min_score=0.01)
    assert cls.tolist() == [0, 1] and sc.tolist() == [0.9, 0.6]  # tri décroissant, 3e requête sous le seuil
    np.testing.assert_allclose(xyxy[1], [80, 30, 120, 70])
    np.testing.assert_allclose(xyxy[0], [170, 40, 200, 60])  # bornée au bord droit
    assert len(select_detections(boxes, scores, 200, 100, top_k=1, min_score=0)[0]) == 1


def test_records_roundtrip_through_the_evaluation_loader(tmp_path):
    recs = to_records(
        "Japan_000001", np.array([2, 0]), np.array([[10, 20, 50, 80], [5, 5, 5, 9]]), np.array([0.7, 0.3])
    )
    assert len(recs) == 1  # boîte de largeur nulle retirée
    assert recs[0] == {"image_id": "Japan_000001", "category_id": 3, "bbox": [10.0, 20.0, 40.0, 60.0], "score": 0.7}
    (tmp_path / "p.json").write_text(json.dumps(recs))
    cls, xyxy, sc = load_predictions(tmp_path / "p.json")["Japan_000001"]
    assert cls.tolist() == [2] and xyxy.tolist() == [[10, 20, 50, 80]] and sc.tolist() == [0.7]


def test_florence_boxes_and_polygons():
    parsed = {"bboxes": [[60, 120, 180, 240]], "polygons": [[[60, 60, 180, 72, 120, 240]]]}
    np.testing.assert_allclose(florence_boxes(parsed), [[60, 120, 180, 240], [60, 60, 180, 240]])
    assert florence_boxes({"bboxes": [], "polygons": []}).shape == (0, 4)


def test_agnostic_merges_classes_and_removes_cross_class_duplicates():
    b = np.array([[0, 0, 10, 10], [0, 0, 10, 10.5], [50, 50, 60, 60]], dtype=float)
    sc = np.array([0.9, 0.8, 0.7])
    assert nms(b, sc, 0.7).tolist() == [0, 2]
    gt = {"a": (np.array([2, 3]), b[[0, 2]])}
    g, p = to_agnostic(gt, {"a": (np.array([0, 1, 3]), b, sc)})
    assert g["a"][0].tolist() == [0, 0]
    assert p["a"][0].tolist() == [0, 0] and p["a"][2].tolist() == [0.9, 0.7]


def _oracle_runs(data, runs, shift_class):
    from roadshift.eval.metrics import load_sizes

    sizes = load_sizes(data / "manifest.csv")
    for tag in ("source_val", "source_test", "test_United_States"):
        preds = []
        for stem in (data / f"list_{tag}.txt").read_text().split():
            stem = stem.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            w, h = sizes[stem]
            for line in (data / "labels" / stem.rsplit("_", 1)[0] / f"{stem}.txt").read_text().split("\n"):
                if line:
                    c, cx, cy, bw, bh = map(float, line.split())
                    box = [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h]
                    preds.append({"image_id": stem, "category_id": (int(c) + shift_class) % 4 + 1, "bbox": box,
                                  "score": 0.9})  # fmt: skip
                    preds.append({"image_id": stem, "category_id": (int(c) + 2) % 4 + 1, "bbox": box, "score": 0.5})
        (runs / f"eval_{tag}").mkdir(parents=True)
        (runs / f"eval_{tag}" / "predictions.json").write_text(json.dumps(preds))


def test_agnostic_report_scores_localisation_without_the_class(raw_root, tmp_path, cfg):
    """Bonnes boîtes, mauvaise classe (+ un doublon d'une autre classe) : mAP par classe nulle, mAP « dégradation »
    = 1 grâce à la NMS entre classes."""
    from roadshift.data.convert import run
    from roadshift.eval.compare import compare
    from roadshift.eval.report import report

    data = tmp_path / "data"
    run(cfg, raw_root, data)
    _oracle_runs(data, tmp_path / "wrong_class", shift_class=1)
    _oracle_runs(data, tmp_path / "right_class", shift_class=0)
    by_class = {r["eval"]: r for r in report(tmp_path / "wrong_class", data, "zs", n_boot=20)}
    agn = {r["eval"]: r for r in report(tmp_path / "wrong_class", data, "zs", n_boot=20, agnostic=True)}
    for tag in ("source_test", "test_United_States"):
        assert by_class[tag]["mAP50"] == pytest.approx(0.0)
        assert agn[tag]["mAP50"] == pytest.approx(1.0) and agn[tag]["F1"] == pytest.approx(1.0)
        assert agn[tag]["agnostic"] is True and "AP50_D00" not in agn[tag]
    rows = compare({"right": tmp_path / "right_class", "wrong": tmp_path / "wrong_class"}, data,
                   ["source_test", "test_United_States"], "test_United_States", n_boot=20, agnostic=True)  # fmt: skip
    paired = [r for r in rows if r["comparison"] == "model_paired"]
    assert paired and all(r["diff"] == pytest.approx(0.0) and r["agnostic"] for r in paired)


def test_grounding_dino_glue_on_a_tiny_random_model(raw_root, tmp_path, cfg):
    """Mécanique seulement (poids aléatoires, rien n'est téléchargé) : tokens, lots, format de sortie."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import torch

    from roadshift.data.convert import run as convert
    from roadshift.zeroshot.run import run

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", "longitudinal", "transverse", "alligator", "crack",
             "pot", "##hole"]  # fmt: skip
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tok = transformers.BertTokenizerFast(vocab_file=str(tmp_path / "vocab.txt"))
    proc = transformers.GroundingDinoProcessor(
        image_processor=transformers.GroundingDinoImageProcessor(size={"shortest_edge": 64, "longest_edge": 64}),
        tokenizer=tok,
    )
    torch.manual_seed(0)
    mcfg = transformers.GroundingDinoConfig(
        backbone_config=transformers.SwinConfig(embed_dim=8, depths=[1] * 4, num_heads=[1] * 4, window_size=2,
                                                image_size=64, out_features=["stage2", "stage3", "stage4"]),
        text_config=transformers.BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1,
                                            num_attention_heads=2, intermediate_size=16, max_position_embeddings=64),
        d_model=32, num_queries=12, encoder_layers=1, decoder_layers=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
        encoder_attention_heads=2, decoder_attention_heads=2, num_feature_levels=3, encoder_n_points=2,
        decoder_n_points=2, max_text_len=32, use_timm_backbone=False, use_pretrained_backbone=False,
    )  # fmt: skip
    model = transformers.GroundingDinoForObjectDetection(mcfg).eval()
    data = tmp_path / "data"
    convert(cfg, raw_root, data)
    gd = {"hf_id": "tiny", "batch": 3, "top_k": 5, "min_score": 0.0}
    cfg["zeroshot"] = {
        "prompts": dict(zip(cfg["data"]["classes"], PHRASES, strict=True)),
        "models": {"grounding_dino": gd},
    }
    out = tmp_path / "zs"
    info = run("grounding_dino", cfg, data, ["source_test", "test_United_States"], out, "cpu", limit=4,
               loaded=(proc, model, "x"))  # fmt: skip
    assert info["tags"]["source_test"]["n_images"] == 4
    for tag in ("source_test", "test_United_States"):
        preds = load_predictions(out / f"eval_{tag}" / "predictions.json")
        assert len(preds) == info["tags"][tag]["n_images"]  # top_k > 0 et seuil 0 : chaque image a des boîtes
        for cls, xyxy, sc in preds.values():
            assert len(cls) <= 5 and set(cls) <= {0, 1, 2, 3} and (xyxy[:, 2:] >= xyxy[:, :2]).all()
            assert (np.diff(sc) <= 0).all() or len(sc) == 1
